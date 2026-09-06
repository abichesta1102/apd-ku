import cv2
import requests
import time
import os
import re
import threading
import mimetypes
import json
from datetime import datetime
from ultralytics import YOLO
from deepface import DeepFace

# --- 1. KONFIGURASI UTAMA ---
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1538194531701489675/oT8T_0mCmXXeOQHLZcVPtsjmfhHa2U4vURY1hP-a9rUqs19A1tVOTzN7yMQVgdbBpoRD"

FOTO_PEGAWAI_DIR = "foto_pegawai"
LOKASI_RUANGAN = "Ruangan Genset"

FORCE_REBUILD_FACE_DB = True

# Load Model AI YOLOv8 - model PPE (helm + vest) hasil training dataset Construction Site Safety
model = YOLO('ppe_best.pt')

# Sesuai urutan kelas di data.yaml dataset "Construction Site Safety" v27:
CLASS_HARDHAT = 0          # helm terpasang dengan benar -> aman
CLASS_NO_HARDHAT = 2       # kepala terdeteksi TANPA helm -> pelanggaran
CLASS_NO_SAFETY_VEST = 4   # terdeteksi TANPA rompi keselamatan -> pelanggaran
CLASS_PERSON = 5           # dipakai sebagai "jangkar" identitas untuk tracking, BUKAN untuk pelanggaran

# --- 2. KONFIGURASI PENGENALAN WAJAH ---
FACE_MODEL_NAME = "SFace"
# "mtcnn" tidak bergantung pada file haarcascade sistem Python,
# sehingga bekerja sempurna di dalam virtual environment (venv).
FACE_DETECTOR_BACKEND = "mtcnn"

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

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

COOLDOWN_SECONDS = 15

# --- STATE UNTUK TRACKING MULTI-ORANG ---
# Kunci semua dictionary ini adalah TRACK ID (angka unik per orang selama dia ada di frame)
track_id_to_nama = {}      # {track_id: "Nama Pegawai"} -- cache identitas, sekali kenal, dipakai terus
last_report_time = {}      # {track_id: waktu laporan terakhir} -- cooldown PER ORANG, bukan global
sedang_lapor = set()       # track_id yang laporannya (kirim Discord) sedang diproses

# --- STATE UNTUK IDENTIFIKASI WAJAH PROAKTIF & GRACE PERIOD ---
sedang_identifikasi = set()  # track_id yang identifikasi wajahnya sedang diproses di thread lain
last_id_attempt = {}         # {track_id: waktu percobaan terakhir}
ID_ATTEMPT_COOLDOWN = 3      # detik antar percobaan identifikasi wajah per orang

first_seen_time = {}         # {track_id: waktu pertama kali terlihat}
GRACE_PERIOD_SECONDS = 15     # waktu tunggu (detik) sebelum lapor pelanggaran jika identitas belum dikenali


def send_discord_alert(nama, waktu, frame, pelanggaran_text):
    """Fungsi mengirim teks laporan & foto bukti ke Discord langsung dari memori tanpa disimpan ke folder"""
    pesan = (
        f"🚨 **LAPORAN PELANGGARAN APD** 🚨\n"
        f"👤 **Nama Pegawai:** {nama}\n"
        f"⚠️ **Pelanggaran:** {pelanggaran_text}\n"
        f"📍 **Lokasi:** {LOKASI_RUANGAN}\n"
        f"⏰ **Waktu:** {waktu}"
    )
    payload = {"content": pesan}
    try:
        # Encode frame langsung ke memori (JPEG buffer) tanpa disimpan ke folder
        success, img_encoded = cv2.imencode('.jpg', frame)
        if not success:
            print("[ERROR] Gagal meng-encode frame gambar untuk Discord.")
            return

        files = {"file": ("bukti_pelanggaran.jpg", img_encoded.tobytes(), "image/jpeg")}
        response = requests.post(DISCORD_WEBHOOK_URL, data={"payload_json": json.dumps(payload)}, files=files)
        if response.status_code in (200, 204):
            print(f"[SUCCESS] Laporan berhasil terkirim ke Discord: {nama}")
        else:
            print(f"[ERROR] Gagal kirim ke Discord. Status Code: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Terjadi kendala koneksi: {e}")


def box_center_inside(inner_box, outer_box):
    """Cek apakah titik tengah 'inner_box' (mis. box helm) ada di dalam 'outer_box' (box orang).
    Ini cara sederhana mengaitkan box helm/vest ke ORANG mana pemiliknya."""
    cx = (inner_box[0] + inner_box[2]) / 2
    cy = (inner_box[1] + inner_box[3]) / 2
    return outer_box[0] <= cx <= outer_box[2] and outer_box[1] <= cy <= outer_box[3]


def _ekstrak_nama_dari_path(file_path):
    """Helper: ambil nama pegawai dari path file foto referensi.
    Contoh: 'foto_pegawai/abi (3).jpeg' -> 'Abi'"""
    nama_file = os.path.basename(file_path)
    nama_tanpa_ext = os.path.splitext(nama_file)[0]
    nama_bersih = re.sub(r'\s*\(\d+\)\s*', '', nama_tanpa_ext)
    nama_bersih = nama_bersih.split('_')[0]
    return nama_bersih.strip().title()


def identifikasi_wajah_proaktif(frame, person_box, person_tid):
    """
    THREAD TERPISAH — dijalankan PROAKTIF setiap kali orang baru terdeteksi di frame,
    BUKAN hanya saat pelanggaran. Tujuannya: begitu orang menghadap kamera sesaat,
    sistem langsung tahu siapa dia dan menyimpan identitasnya ke cache.
    Setelah dikenali SEKALI, tidak akan pernah dipanggil lagi untuk track ID ini.
    """
    try:
        if person_box is None:
            return

        x1, y1, x2, y2 = person_box
        
        h, w = frame.shape[:2]
        cx1, cy1 = max(0, x1), max(0, y1)
        # Menggunakan seluruh kotak (y2) agar wajah (terutama dagu) tidak terpotong!
        cx2, cy2 = min(w, x2), min(h, y2)
        crop_wajah = frame[cy1:cy2, cx1:cx2]

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
                nama_pegawai = _ekstrak_nama_dari_path(df_res[0].iloc[0]['identity'])
                track_id_to_nama[person_tid] = nama_pegawai
                print(f"[INFO] ✅ Wajah dikenali: ID#{person_tid} = {nama_pegawai}")
            else:
                print(f"[INFO] ID#{person_tid}: wajah belum cocok (mungkin belum menghadap kamera).")
        except Exception as e:
            print(f"[WARNING] Gagal identifikasi wajah ID#{person_tid}: Face could not be detected atau buram.")
            
    finally:
        # SANGAT PENTING: Hapus dari daftar sedang diproses supaya main loop
        # bisa MENCOBA LAGI beberapa detik kemudian jika dia belum dikenali!
        if person_tid not in track_id_to_nama:
            sedang_identifikasi.discard(person_tid)
def proses_pelanggaran(frame, jenis_pelanggaran, person_tid):
    """
    THREAD TERPISAH — dipanggil saat pelanggaran terdeteksi.
    Identitas diambil dari cache yang sudah diisi oleh identifikasi_wajah_proaktif.
    Tidak perlu wajah terlihat lagi saat pelanggaran terjadi.
    """
    try:
        nama_pegawai = track_id_to_nama.get(person_tid, "Tidak Dikenal / Pegawai Baru")
        print(f"[INFO] 🚨 ID#{person_tid} ({nama_pegawai}) melanggar: {', '.join(jenis_pelanggaran)}")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pelanggaran_text = " & ".join(jenis_pelanggaran)
        send_discord_alert(nama_pegawai, now_str, frame, pelanggaran_text)
    finally:
        sedang_lapor.discard(person_tid)


print("Sistem Deteksi APD Berjalan... Tekan 'q' pada jendela video untuk berhenti.")

frame_skip = 3
frame_count = 0
annotated_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip == 0:
        # model.track() dipakai (bukan model.predict()) supaya tiap orang dapat TRACK ID
        # yang persisten antar-frame. persist=True wajib, supaya ID tidak reset tiap panggilan.
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=320, verbose=False)
        annotated_frame = results[0].plot()

        boxes = results[0].boxes
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(boxes)

        person_list = []       # [(track_id, box), ...] -- ini "jangkar" identitas tiap orang
        hardhat_list = []
        no_hardhat_list = []
        no_vest_list = []

        for box, tid in zip(boxes, track_ids):
            cls_id = int(box.cls[0])
            xyxy = tuple(map(int, box.xyxy[0]))
            if cls_id == CLASS_PERSON and tid is not None:
                person_list.append((int(tid), xyxy))
            elif cls_id == CLASS_HARDHAT:
                hardhat_list.append(xyxy)
            elif cls_id == CLASS_NO_HARDHAT:
                no_hardhat_list.append(xyxy)
            elif cls_id == CLASS_NO_SAFETY_VEST:
                no_vest_list.append(xyxy)

        # Catat waktu kemunculan pertama kali untuk setiap orang
        current_time = time.time()
        for person_tid, _ in person_list:
            if person_tid not in first_seen_time:
                first_seen_time[person_tid] = current_time

        # --- FASE 1: IDENTIFIKASI WAJAH PROAKTIF ---
        # Untuk setiap orang yang BELUM dikenali, coba identifikasi wajahnya
        # secara terus-menerus (dengan cooldown). Begitu dikenali SEKALI,
        # identitas disimpan permanen di cache dan tidak dicoba lagi.
        for person_tid, person_box in person_list:
            if person_tid not in track_id_to_nama and person_tid not in sedang_identifikasi:
                if current_time - last_id_attempt.get(person_tid, 0) > ID_ATTEMPT_COOLDOWN:
                    last_id_attempt[person_tid] = current_time
                    sedang_identifikasi.add(person_tid)
                    threading.Thread(
                        target=identifikasi_wajah_proaktif,
                        args=(frame.copy(), person_box, person_tid),
                        daemon=True
                    ).start()

        # --- FASE 2: DETEKSI PELANGGARAN ---
        # Untuk SETIAP orang yang terlacak, cek pelanggaran miliknya sendiri-sendiri.
        # Jika identitas sudah dikenali (dari fase 1), laporan langsung pakai nama itu.
        for person_tid, person_box in person_list:
            no_helm_box = next((b for b in no_hardhat_list if box_center_inside(b, person_box)), None)
            vest_pelanggaran = any(box_center_inside(b, person_box) for b in no_vest_list)

            jenis_pelanggaran = []
            if no_helm_box is not None:
                jenis_pelanggaran.append("Tidak Menggunakan Helm Proyek")
            if vest_pelanggaran:
                jenis_pelanggaran.append("Tidak Menggunakan Rompi Keselamatan")

            if jenis_pelanggaran:
                last_time = last_report_time.get(person_tid, 0)
                if current_time - last_time > COOLDOWN_SECONDS and person_tid not in sedang_lapor:
                    # CEK GRACE PERIOD (Toleransi Waktu)
                    # Jika belum dikenali, tapi belum melewati grace period, tunggu dulu
                    waktu_berlalu = current_time - first_seen_time[person_tid]
                    if person_tid not in track_id_to_nama and waktu_berlalu < GRACE_PERIOD_SECONDS:
                        continue  # Abaikan dulu supaya AI punya waktu identifikasi wajah

                    sedang_lapor.add(person_tid)
                    last_report_time[person_tid] = current_time
                    threading.Thread(
                        target=proses_pelanggaran,
                        args=(frame.copy(), jenis_pelanggaran, person_tid),
                        daemon=True
                    ).start()

    if annotated_frame is not None:
        cv2.imshow("Kamera Pengawas APD", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()