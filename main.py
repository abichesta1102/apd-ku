import cv2
import requests
import time
import os
import re
import io
import threading
import json
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO
from deepface import DeepFace

# Muat variabel dari file .env (jika ada)
load_dotenv()

# --- 1. KONFIGURASI UTAMA (dibaca dari .env) ---
DISCORD_WEBHOOK_URL  = os.getenv("DISCORD_WEBHOOK_URL", "")
FOTO_PEGAWAI_DIR     = os.getenv("FOTO_PEGAWAI_DIR",    "foto_pegawai")
LOKASI_RUANGAN       = os.getenv("LOKASI_RUANGAN",      "Tidak Diketahui")
YOLO_MODEL_PATH      = os.getenv("YOLO_MODEL_PATH",     "ppe_best.pt")
FORCE_REBUILD_FACE_DB = os.getenv("FORCE_REBUILD_FACE_DB", "false").lower() == "true"

CAMERA_INDEX  = int(os.getenv("CAMERA_INDEX",  "0"))
CAMERA_WIDTH  = int(os.getenv("CAMERA_WIDTH",  "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
FRAME_SKIP    = int(os.getenv("FRAME_SKIP",    "3"))

COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "15"))

if not DISCORD_WEBHOOK_URL:
    print("[WARNING] DISCORD_WEBHOOK_URL tidak diset di .env — laporan Discord tidak akan terkirim!")

# Load Model AI YOLOv8
# PENTING: file model di-rename jadi 'ppe_best.pt' supaya tidak tertukar dengan model helm-only lama
model = YOLO(YOLO_MODEL_PATH)

# Sesuai urutan kelas di data.yaml dataset "Construction Site Safety" v27 (SUDAH DIKONFIRMASI):
CLASS_HARDHAT     = 0   # helm terpasang dengan benar -> aman
CLASS_NO_HARDHAT  = 2   # kepala terdeteksi TANPA helm -> pelanggaran
CLASS_NO_SAFETY_VEST = 4  # terdeteksi TANPA rompi keselamatan -> pelanggaran

# --- 2. KONFIGURASI PENGENALAN WAJAH (dibaca dari .env) ---
FACE_MODEL_NAME      = os.getenv("FACE_MODEL_NAME",      "SFace")
FACE_DETECTOR_BACKEND = os.getenv("FACE_DETECTOR_BACKEND", "retinaface")
MIN_CROP_SIZE        = int(os.getenv("MIN_CROP_SIZE",     "40"))

# Hapus PKL cache lama agar dibangun ulang dengan backend yang benar
# (PKL diindeks per model+backend, PKL lama "skip" tidak kompatibel dengan "retinaface")
if os.path.isdir(FOTO_PEGAWAI_DIR):
    for pkl_file in os.listdir(FOTO_PEGAWAI_DIR):
        if pkl_file.endswith('.pkl') and 'skip' in pkl_file:
            os.remove(os.path.join(FOTO_PEGAWAI_DIR, pkl_file))
            print(f"[INFO] PKL cache lama dihapus: {pkl_file}")

if FORCE_REBUILD_FACE_DB and os.path.isdir(FOTO_PEGAWAI_DIR):
    for pkl_file in os.listdir(FOTO_PEGAWAI_DIR):
        if pkl_file.endswith('.pkl'):
            os.remove(os.path.join(FOTO_PEGAWAI_DIR, pkl_file))
    print("[INFO] Cache wajah lama dihapus, akan dibangun ulang sekali di awal.")

print("[INFO] Menyiapkan model pengenalan wajah (hanya sekali di awal, mohon tunggu)...")
try:
    DeepFace.build_model(FACE_MODEL_NAME)
    daftar_foto = [f for f in os.listdir(FOTO_PEGAWAI_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if daftar_foto:
        DeepFace.find(
            img_path=os.path.join(FOTO_PEGAWAI_DIR, daftar_foto[0]),
            db_path=FOTO_PEGAWAI_DIR,
            model_name=FACE_MODEL_NAME,
            detector_backend=FACE_DETECTOR_BACKEND,
            enforce_detection=False,
            silent=True
        )
    print("[INFO] Model wajah siap.")
except Exception as e:
    print(f"[WARNING] Pemanasan model wajah dilewati: {e}")

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

last_sent_times = {}   # key: box_key -> last alert timestamp per orang
threads_aktif = set()  # set of box_keys yang sedang diproses
lock = threading.Lock()


def send_discord_alert(nama, waktu, gambar_bytes, pelanggaran_text):
    """Fungsi mengirim teks laporan & foto bukti ke Discord (tanpa menyimpan ke disk)"""
    pesan = (
        f"🚨 **LAPORAN PELANGGARAN APD** 🚨\n"
        f"👤 **Nama Pegawai:** {nama}\n"
        f"⚠️ **Pelanggaran:** {pelanggaran_text}\n"
        f"📍 **Lokasi:** {LOKASI_RUANGAN}\n"
        f"⏰ **Waktu:** {waktu}"
    )
    payload = {"content": pesan}
    try:
        files = {"file": ("bukti_pelanggaran.jpeg", gambar_bytes, "image/jpeg")}
        response = requests.post(DISCORD_WEBHOOK_URL, data={"payload_json": json.dumps(payload)}, files=files)
        if response.status_code in (200, 204):
            print(f"[SUCCESS] Laporan berhasil terkirim ke Discord: {nama}")
        else:
            print(f"[ERROR] Gagal kirim ke Discord. Status Code: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Terjadi kendala koneksi: {e}")


MIN_CROP_SIZE = 40   # Minimal lebar/tinggi crop (piksel) agar layak diidentifikasi


def identifikasi_wajah(frame, head_box):
    """
    Crop area kepala dari frame, lalu kenali wajah dengan DeepFace.
    Menggunakan backend 'retinaface' agar wajah terdeteksi dengan benar di dalam crop
    (crop bisa mengandung helm di atasnya — retinaface akan menemukan wajah di bawahnya).
    Kembalikan nama pegawai sebagai string.
    """
    nama_pegawai = "Tidak Dikenal / Pegawai Baru"
    if head_box is None:
        # Tidak ada kepala terdeteksi — gunakan seluruh frame sebagai fallback
        crop_wajah = frame
    else:
        x1, y1, x2, y2 = head_box
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.5)
        pad_y = int((y2 - y1) * 0.7)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        crop_wajah = frame[cy1:cy2, cx1:cx2]

        # Validasi: crop terlalu kecil → retinaface tidak bisa bekerja, skip
        ch, cw = crop_wajah.shape[:2]
        if cw < MIN_CROP_SIZE or ch < MIN_CROP_SIZE:
            print(f"[WARNING] Crop wajah terlalu kecil ({cw}x{ch}px), identifikasi dilewati.")
            return nama_pegawai

    try:
        df_res = DeepFace.find(
            img_path=crop_wajah,
            db_path=FOTO_PEGAWAI_DIR,
            model_name=FACE_MODEL_NAME,
            detector_backend=FACE_DETECTOR_BACKEND,
            enforce_detection=False,
            silent=True
        )
        if len(df_res) > 0 and not df_res[0].empty:
            file_path = df_res[0].iloc[0]['identity']
            nama_file = os.path.basename(file_path)
            nama_tanpa_ext = os.path.splitext(nama_file)[0]
            nama_bersih = re.sub(r'\s*\(\d+\)\s*', '', nama_tanpa_ext)
            nama_bersih = nama_bersih.split('_')[0]
            nama_pegawai = nama_bersih.strip().title()
            print(f"[INFO] Wajah teridentifikasi: {nama_pegawai}")
        else:
            print("[INFO] Wajah tidak cocok dengan database pegawai.")
    except Exception as e:
        print(f"[WARNING] Gagal identifikasi wajah: {e}")
    return nama_pegawai


def proses_pelanggaran(frame, head_box, jenis_pelanggaran, box_key):
    """
    Jalan di THREAD TERPISAH supaya video utama tetap lancar.
    head_box    : koordinat box kepala pelanggar (NO-Hardhat atau Hardhat)
    jenis_pelanggaran: list pelanggaran orang ini
    box_key     : tuple rounded koordinat kepala, dipakai sebagai ID unik orang
    """
    try:
        print(f"[INFO] Memproses pelanggar {box_key}: {', '.join(jenis_pelanggaran)}")
        nama_pegawai = identifikasi_wajah(frame, head_box)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        gambar_bytes = io.BytesIO(img_encoded.tobytes())

        pelanggaran_text = " & ".join(jenis_pelanggaran)
        send_discord_alert(nama_pegawai, now_str, gambar_bytes, pelanggaran_text)
    finally:
        with lock:
            threads_aktif.discard(box_key)


print("Sistem Deteksi APD Berjalan... Tekan 'q' pada jendela video untuk berhenti.")

frame_skip = FRAME_SKIP
frame_count = 0
annotated_frame = None


def box_center_y(box):
    """Pusat Y dari sebuah box (x1,y1,x2,y2)."""
    return (box[1] + box[3]) / 2


def box_overlap_x(a, b):
    """Hitung overlap horizontal (0.0–1.0) antara dua box — untuk pasangkan kepala & vest."""
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1))
    union = max(ax2, bx2) - min(ax1, bx1)
    return inter / union if union > 0 else 0.0


def rounded_key(box, grid=30):
    """Bulatkan koordinat box ke grid agar posisi sedikit bergeser tetap dianggap orang sama."""
    return tuple((v // grid) * grid for v in box)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip == 0:
        results = model.predict(frame, imgsz=320, verbose=False)
        annotated_frame = results[0].plot()

        # Kumpulkan semua deteksi per kelas
        no_helm_list = []   # list of (x1,y1,x2,y2) kepala tanpa helm
        helm_list    = []   # list of (x1,y1,x2,y2) kepala dengan helm
        no_vest_list = []   # list of (x1,y1,x2,y2) badan tanpa rompi

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            coords = tuple(map(int, box.xyxy[0]))
            if cls_id == CLASS_NO_HARDHAT:
                no_helm_list.append(coords)
            elif cls_id == CLASS_HARDHAT:
                helm_list.append(coords)
            elif cls_id == CLASS_NO_SAFETY_VEST:
                no_vest_list.append(coords)

        # Pasangkan setiap vest ke kepala yang paling overlap secara horizontal
        semua_kepala = no_helm_list + helm_list
        vest_ke_kepala = {}  # vest_box -> kepala_box terdekat
        for vest in no_vest_list:
            best_kepala = None
            best_overlap = 0.0
            for kepala in semua_kepala:
                if kepala[3] <= vest[3]:   # kepala harus di atas vest
                    ov = box_overlap_x(vest, kepala)
                    if ov > best_overlap:
                        best_overlap = ov
                        best_kepala = kepala
            if best_kepala is not None and best_overlap > 0.2:
                vest_ke_kepala[vest] = best_kepala

        # --- Bangun daftar pelanggar sebagai list of (head_box, jenis_list, box_key) ---
        # Menggunakan list (bukan dict) agar vest-only tanpa kepala juga punya key unik.
        pelanggar_map = {}   # head_box_tuple -> index di pelanggar list
        pelanggar = []       # list of [head_box, [jenis], box_key]

        def tambah_pelanggaran(hbox, jenis_baru, kunci):
            """Tambahkan jenis_baru ke pelanggar dengan kunci, atau buat entri baru."""
            if kunci in pelanggar_map:
                pelanggar[pelanggar_map[kunci]][1].append(jenis_baru)
            else:
                pelanggar_map[kunci] = len(pelanggar)
                pelanggar.append([hbox, [jenis_baru], kunci])

        # Pelanggaran helm — key = rounded koordinat kepala
        for hbox in no_helm_list:
            tambah_pelanggaran(hbox, "Tidak Menggunakan Helm Proyek", rounded_key(hbox))

        # Pelanggaran vest — pasangkan ke kepala, key = rounded koordinat kepala
        for vest, kepala in vest_ke_kepala.items():
            tambah_pelanggaran(kepala, "Tidak Menggunakan Rompi Keselamatan", rounded_key(kepala))

        # Vest tanpa kepala — tiap vest dapat key unik dari koordinat vest-nya sendiri
        for vest in no_vest_list:
            if vest not in vest_ke_kepala:
                vest_key = ("vest",) + rounded_key(vest)
                tambah_pelanggaran(None, "Tidak Menggunakan Rompi Keselamatan", vest_key)

        # Proses tiap pelanggar secara independen di thread terpisah
        current_time = time.time()
        frame_copy = frame.copy()
        for head_box, jenis, box_key in pelanggar:
            with lock:
                sudah_diproses = box_key in threads_aktif
                cooldown_ok = (current_time - last_sent_times.get(box_key, 0)) > COOLDOWN_SECONDS

            if not sudah_diproses and cooldown_ok:
                with lock:
                    threads_aktif.add(box_key)
                    last_sent_times[box_key] = current_time
                threading.Thread(
                    target=proses_pelanggaran,
                    args=(frame_copy, head_box, list(jenis), box_key),
                    daemon=True
                ).start()

    if annotated_frame is not None:
        cv2.imshow("Kamera Pengawas APD", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()