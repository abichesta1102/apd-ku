"""Tes cepat RetinaFace + DeepFace"""
import cv2
import os
import time
from deepface import DeepFace

FOTO_PEGAWAI_DIR = "foto_pegawai"

# Hapus semua cache pkl
for f in os.listdir(FOTO_PEGAWAI_DIR):
    if f.endswith('.pkl'):
        os.remove(os.path.join(FOTO_PEGAWAI_DIR, f))
        print(f"Cache dihapus: {f}")

# Ambil frame dari kamera
print("Mengambil gambar dari kamera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1)
ret, frame = cap.read()
cap.release()

if not ret:
    print("GAGAL ambil frame!")
    exit(1)

cv2.imwrite("test_capture.jpg", frame)
print(f"Frame disimpan: test_capture.jpg {frame.shape}")

# Tes dengan RetinaFace
print("\nMenjalankan DeepFace.find dengan RetinaFace...")
try:
    result = DeepFace.find(
        img_path=frame,
        db_path=FOTO_PEGAWAI_DIR,
        model_name="VGG-Face",
        detector_backend="retinaface",
        enforce_detection=False,
        silent=True
    )
    
    if len(result) > 0 and not result[0].empty:
        print("\n=== WAJAH TERIDENTIFIKASI ===")
        identity = result[0]['identity'][0]
        print(f"File cocok: {os.path.basename(identity)}")
        # Tampilkan semua kolom untuk debug
        print(f"Kolom: {list(result[0].columns)}")
        print(result[0].head())
    else:
        print("\nTidak ada kecocokan ditemukan.")
        if len(result) > 0:
            print(f"DataFrame kosong. Kolom: {list(result[0].columns)}")
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
