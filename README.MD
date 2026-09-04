# Sistem Deteksi APD (Alat Pelindung Diri)

Sistem computer vision real-time yang mendeteksi kepatuhan penggunaan Alat Pelindung Diri (APD) pada pegawai — saat ini mendeteksi **helm proyek** dan **rompi keselamatan (safety vest)** — mengenali identitas pegawai yang melanggar lewat face recognition, dan mengirim laporan otomatis (nama, foto bukti, waktu, lokasi) ke Discord.

Dibangun untuk kompetisi **HOLOGY 9.0 (Universitas Brawijaya) — kategori HoloDev (Software Development)**.

## Fitur

- **Deteksi helm** — mengenali apakah pegawai memakai helm proyek (`Hardhat`) atau tidak (`NO-Hardhat`)
- **Deteksi rompi keselamatan** — mengenali apakah rompi terpasang (`Safety Vest`) atau tidak (`NO-Safety Vest`)
- **Pengenalan wajah** — mencocokkan wajah pegawai yang melanggar dengan database foto pegawai, menampilkan nama di laporan
- **Laporan otomatis ke Discord** — teks (nama, jenis pelanggaran, lokasi, waktu) + foto bukti, terkirim lewat Discord Webhook
- **Multi-pelanggaran sekaligus** — kalau helm dan vest sama-sama tidak terpasang, satu laporan menyebutkan keduanya
- **Dioptimalkan untuk laptop tanpa GPU** — diuji lancar di ThinkPad T480 (Intel i5 gen-8, tanpa GPU diskrit)

### Belum diimplementasikan (rencana pengembangan)
- Deteksi APD lain: body harness, sepatu safety
- Dukungan multi-orang dalam satu frame secara bersamaan (saat ini asumsi 1 orang di depan kamera)
- Sumber video dari CCTV (RTSP stream) — saat ini masih webcam laptop (`cv2.VideoCapture(0)`)

## Cara Kerja (Alur Pipeline)

```
Webcam → YOLOv8 (deteksi helm & vest) → ada pelanggaran?
                                              │
                                              ▼ (thread terpisah, non-blocking)
                                    Crop area kepala → DeepFace (SFace)
                                    cocokkan dengan foto_pegawai/
                                              │
                                              ▼
                                    Susun pesan + screenshot → kirim ke Discord Webhook
```

Poin penting arsitektur:
- **Threading**: proses pengenalan wajah + kirim Discord dijalankan di *background thread* (`threading.Thread`), supaya jendela video (`cv2.imshow`) tidak freeze/"Not Responding" selagi menunggu proses AI yang lebih berat.
- **Frame skipping**: deteksi YOLO cuma dijalankan tiap 3 frame (`frame_skip = 3`) pada resolusi inferensi 320px, bukan tiap frame — menghemat CPU tanpa mengorbankan responsivitas.
- **Cooldown**: jeda minimal 15 detik (`COOLDOWN_SECONDS`) antar laporan, supaya Discord tidak di-spam saat satu pelanggaran terdeteksi berkali-kali di frame berurutan.
- **Model warm-up**: model wajah (SFace) dan cache database wajah (`.pkl`) dibangun **sekali di awal** program (sebelum loop kamera dimulai), bukan di tiap pelanggaran — ini adalah fix dari bug awal yang membuat sistem sangat berat/freeze.
- **Face detector `"skip"`**: karena area kepala sudah diketahui presisi dari box YOLO, pengenalan wajah langsung dijalankan di crop tersebut tanpa langkah deteksi wajah terpisah (`detector_backend="skip"`) — lebih cepat, dan menghindari bug OpenCV Haar cascade yang hilang di sebagian environment Windows.

## Tech Stack

| Komponen | Tools | Peran |
|---|---|---|
| Deteksi objek (helm/vest) | YOLOv8n (Ultralytics), custom-trained | Deteksi real-time dari frame kamera |
| Pengenalan wajah | DeepFace, model `SFace` | Identifikasi nama pegawai dari crop wajah |
| Capture & image processing | OpenCV (`cv2`) | Ambil frame webcam, gambar bounding box, crop |
| Notifikasi | Discord Webhook (`requests`) | Kirim laporan teks + foto bukti |
| Training | Google Colab (free T4 GPU) | Training model YOLO custom |
| Dataset & anotasi | Roboflow | Sumber & manajemen dataset |

## Model Deteksi APD

Model final di-training dari dataset publik **["Construction Site Safety"](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) v27** (Roboflow Universe, lisensi CC BY 4.0) — 2.801 gambar (2.605 train / 114 valid / 82 test), base model `YOLOv8n`, 50 epoch, `imgsz=640`.

**10 kelas dalam dataset** (index sesuai `data.yaml`):
```
0: Hardhat        1: Mask           2: NO-Hardhat     3: NO-Mask        4: NO-Safety Vest
5: Person         6: Safety Cone    7: Safety Vest    8: machinery      9: vehicle
```
Sistem ini hanya memakai index **0, 2, 4** (`Hardhat`, `NO-Hardhat`, `NO-Safety Vest`) — kelas lain (Mask, Person, machinery, dll) ada di model tapi tidak dipakai di logic saat ini.

**Hasil evaluasi model (kelas yang dipakai):**

| Kelas | Precision | Recall | mAP50 |
|---|---|---|---|
| Hardhat | 0.962 | 0.759 | 0.863 |
| NO-Hardhat | 0.859 | 0.551 | 0.650 |
| Safety Vest | 0.967 | 0.708 | 0.869 |
| NO-Safety Vest | 0.866 | 0.594 | 0.720 |

> **Catatan riwayat:** versi model sebelumnya (dataset v1, hanya 398 gambar) menghasilkan mAP50 vest yang sangat buruk (~0.08 dan ~0.004) karena data vest terlalu sedikit — karena itu training diulang dengan dataset v27 yang jauh lebih besar.

## Struktur Project

```
APD/
├── main.py              # Program utama (kamera, deteksi, pengenalan wajah, laporan Discord)
├── test_face.py          # Script eksperimen/testing awal untuk face recognition
├── ppe_best.pt            # Model YOLOv8 custom (10 kelas, hasil training dataset v27)
├── foto_pegawai/          # Database foto wajah pegawai (nama file = nama pegawai, jadi acuan hasil identifikasi)
└── venv/                  # Virtual environment Python
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install ultralytics opencv-python deepface requests numpy
```

Siapkan foto pegawai di folder `foto_pegawai/`, satu atau beberapa foto wajah per pegawai (nama file dijadikan acuan nama, contoh: `sunaryo_1.jpg`, `sunaryo_2.jpg`).

Isi `DISCORD_WEBHOOK_URL` di `main.py` dengan URL webhook channel Discord tujuan.

## Konfigurasi Utama (`main.py`)

| Variabel | Fungsi |
|---|---|
| `LOKASI_RUANGAN` | Nama lokasi yang ditampilkan di laporan (saat ini di-hardcode; akan jadi per-kamera saat multi-CCTV) |
| `COOLDOWN_SECONDS` | Jeda minimal antar laporan Discord (detik) |
| `FORCE_REBUILD_FACE_DB` | Set `True` sekali kalau baru menambah/mengganti foto pegawai, lalu kembalikan ke `False` |
| `frame_skip` | Jumlah frame yang dilewati sebelum deteksi berikutnya dijalankan |

## Menjalankan

```bash
python main.py
```

Tekan `q` pada jendela video untuk menghentikan program.

## Keterbatasan Saat Ini

- Didesain untuk **1 orang dalam frame pada satu waktu** — kalau ada beberapa orang dengan status APD berbeda-beda sekaligus, sistem belum bisa mengaitkan pelanggaran ke orang yang tepat satu per satu (butuh person-level box association untuk pengembangan lanjutan).
- Sumber video masih webcam laptop, belum RTSP/CCTV.
- Foto referensi pegawai idealnya cukup fokus ke wajah (karena `detector_backend="skip"` tidak melakukan deteksi/alignment wajah ulang pada foto database).