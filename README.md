# SCSG Project

**Deteksi Social Distancing Real-Time (YOLO + Bird's-Eye View)**

## Anggota Kelompok

| No | Nama | NPM |
|----|------|-----|
| 1 | Yazid Dahren Fauzan | 140810230001 |
| 2 | Robby Azwan Saputra | 140810230008 |
| 3 | Achmad Dzaki Azhari | 140810230034 |
| 4 | Siti Nailah Eko     | 140810230059 |
| 5 | Athallah Azhar Aulia Hadi | 140810230083 |

---

## Deskripsi

Aplikasi Python untuk memantau *social distancing* secara real-time dari file video, stream, atau webcam.
Orang dideteksi memakai YOLOv8, lalu satu titik acuan per orang diproyeksikan ke bidang 2D tampak atas
(*bird's-eye view*) lewat homography. Jarak Euclidean antar orang dihitung di bidang tersebut sehingga
distorsi perspektif kamera tidak memengaruhi hasil.

Titik acuan jarak bisa dipilih lewat `--method`: `centroid` (default, baseline) atau `bottom` (titik kaki, peningkatan). Lihat bagian Catatan.

Alur per frame: **Baca frame → Deteksi → Transformasi titik → Hitung jarak → Gambar overlay → Tampilkan**.

## Struktur Project

```
social_distancing_detector/
├── config.py            # konstanta + dataclass Settings (bisa di-override via JSON)
├── src/
│   ├── centroid.py      # pemilihan titik acuan (centroid / bottom)
│   ├── detector.py      # PersonDetector (YOLO) + container Detections
│   ├── perspective.py   # PerspectiveTransformer (homography / BEV)
│   ├── distance.py      # DistanceAnalyzer (logika pelanggaran dengan cdist)
│   └── visualizer.py    # kotak, garis, HUD, inset bird's-eye
├── utils/helper.py      # FPSCounter, pemilihan ROI dengan mouse
├── main.py              # entry point
└── requirements.txt
```

## Instalasi

Membutuhkan Python 3.9 atau lebih baru.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / Mac
pip install -r requirements.txt
```

Bobot `yolov8n.pt` akan diunduh otomatis saat pertama kali dijalankan (butuh koneksi internet).

## Cara Menjalankan

Jalankan dari dalam folder `social_distancing_detector`.

```bash
# File video, pilih ROI secara interaktif
python main.py --source video_testing.mp4 --select-points

# Webcam
python main.py --source 0

# Pakai config tersimpan, hanya analisis orang di dalam ROI
python main.py --source video_testing.mp4 --config roi.json --roi-only

# Bandingkan metode titik acuan pada video dan ROI yang sama
python main.py --source video_testing.mp4 --config roi.json --method centroid
python main.py --source video_testing.mp4 --config roi.json --method bottom
```

Tekan **`q`** untuk keluar.

### Memilih ROI

Klik 4 titik di lantai dengan urutan: **kiri-atas, kanan-atas, kanan-bawah, kiri-bawah**.
Pilih area yang berbentuk persegi panjang di dunia nyata (misalnya lantai atau trotoar).

| Tombol | Fungsi |
|--------|--------|
| Klik kiri | Tambah titik |
| Klik kanan | Undo titik terakhir |
| `R` | Reset semua titik |
| `Enter` / `Space` | Konfirmasi (setelah 4 titik) |
| `Q` / `Esc` | Batal |

Jika `SRC_POINTS` masih nilai default (dan `--no-select` tidak dipakai), jendela pemilihan ROI muncul otomatis.
Titik yang dipilih dicetak sebagai JSON di terminal agar bisa disalin ke file config.

### File Config (opsional)

```json
{
  "distance_threshold_px": 150,
  "confidence_threshold": 0.45,
  "src_points": [[420, 260], [860, 255], [1150, 690], [130, 700]]
}
```

Nama key sama dengan field pada `Settings` di `config.py`.
Prioritas: **flag CLI > file JSON > `config.py`**.

### Opsi CLI

| Flag | Keterangan |
|------|------------|
| `--source` | Path file, URL stream, atau indeks webcam (default `0`) |
| `--config` | File JSON untuk override konfigurasi |
| `--select-points` / `--no-select` | Paksa / lewati pemilihan ROI |
| `--model` | Path bobot YOLO |
| `--method` | Titik acuan jarak: `centroid` (default) atau `bottom` (titik kaki) |
| `--conf` | Threshold kepercayaan deteksi (default `0.5`) |
| `--threshold` | Threshold jarak dalam piksel BEV (default `100`) |
| `--roi-only` | Abaikan orang yang kakinya di luar ROI |
| `--no-birdseye` | Sembunyikan inset bird's-eye |

## Kalibrasi Threshold Jarak

`DISTANCE_THRESHOLD_PX` bersatuan **piksel di bidang bird's-eye**, bukan meter. Dua orang dianggap melanggar
jika jaraknya **lebih kecil** dari nilai ini.

Untuk mengonversi dari meter:

```
threshold_px = jarak_aman_meter × (BEV_WIDTH / lebar_ROI_meter)
```

Contoh: ROI berukuran 4 m × 6 m dengan BEV 400×600 px berarti skala 100 px/m,
sehingga jarak aman 1,5 m ≈ **150 px** dan 2 m ≈ **200 px**.

## Catatan

- Default memakai **centroid** bounding box (`((x1+x2)/2, (y1+y2)/2)`) sebagai baseline. Mode `bottom`
  (`((x1+x2)/2, y2)`) adalah peningkatan karena titik kaki berada di bidang tanah, sedangkan centroid
  melayang di atas tanah sehingga proyeksi homography-nya bisa bias.
- Belum ada *tracking*: setiap frame diproses terpisah dan orang tidak punya ID yang konsisten antar frame.
- Detector bisa diganti: kelas apa pun yang punya method `detect(frame) -> Detections` dapat menggantikan
  `PersonDetector` tanpa mengubah modul lain.
