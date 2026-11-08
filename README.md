# Amazon Reviews Polarity — Klasifikasi Sentimen dengan PySpark

Project Big Data & AI yang mengklasifikasikan ulasan produk Amazon menjadi **negatif** atau **positif** menggunakan **PySpark**. Alur kerjanya mencakup perbandingan metode ingestion data (RDD vs DataFrame), eksplorasi dan pembersihan data, *feature engineering*, visualisasi, lalu pemodelan **TF-IDF + Linear SVM** (perbandingan unigram vs bigram).

## Daftar Isi

- [Struktur Project](#struktur-project)
- [Prasyarat](#prasyarat)
- [Cara Startup](#cara-startup)
- [Alur Notebook](#alur-notebook)
- [Tentang Dataset](#tentang-dataset)
- [Troubleshooting](#troubleshooting)

## Struktur Project

```
amazon-reviews-polarity/
├── datasets/                  # (dibuat manual, di-ignore git) letakkan train.csv & test.csv di sini
├── images/                    # Gambar hasil visualisasi (EDA, n-gram, panjang teks)
├── test/                      # Versi notebook yang dipecah per tahap
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_exploration_and_preprocessing.ipynb
│   ├── 03_model_pipelines_and_evaluation.ipynb
│   └── requirements.txt
├── main.ipynb                 # Notebook utama (end-to-end)
├── requirements.txt           # Dependensi Python
└── README.md
```

> Folder `datasets/` tercantum di `.gitignore`, jadi **tidak ikut ter-clone**. Folder ini harus dibuat sendiri dan diisi dengan dataset (lihat langkah 2).

## Prasyarat

| Kebutuhan | Keterangan |
|-----------|------------|
| **Python** | 3.10+ (notebook dikembangkan dengan Python 3.12) |
| **Java (JDK)** | JDK 17 atau 21 — wajib untuk menjalankan Spark. Pastikan `java -version` berjalan |
| **RAM** | Minimal 8 GB disarankan. Spark dikonfigurasi `spark.driver.memory = 4g` |
| **Disk** | Beberapa GB kosong untuk dataset dan direktori sementara Spark |
| **Editor notebook** | VS Code (ekstensi **Python** dan **Jupyter**) atau Jupyter Notebook/Lab |

Cek versi yang terpasang lewat terminal:

```bash
python --version
java -version
```

## Cara Startup

Seluruh project dijalankan langsung dari `main.ipynb`. Dependensi dipasang otomatis oleh sel pertama di notebook, jadi tidak perlu menjalankan `pip install` secara manual.

### 1. Dapatkan project

Clone repository:

```bash
git clone https://github.com/Admltfr/amazon-reviews-polarity.git
```

Atau unduh sebagai ZIP dari GitHub, lalu ekstrak.

### 2. Unduh dataset dari Kaggle

1. Buka halaman dataset (perlu login Kaggle):
   [Amazon reviews — Kaggle](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=train.csv)
2. Klik **Download**, lalu ekstrak file `.zip`-nya sehingga kamu mendapatkan `train.csv` dan `test.csv`.
3. Buat folder `datasets/` di dalam folder project, lalu letakkan kedua file itu di sana.

Struktur akhirnya harus seperti ini:

```
amazon-reviews-polarity/
└── datasets/
    ├── train.csv
    └── test.csv
```

### 3. Buka `main.ipynb`

- **VS Code:** buka folder project (`File → Open Folder`), lalu buka `main.ipynb`.
- **Jupyter:** jalankan `jupyter notebook` dari dalam folder project, lalu buka `main.ipynb`.

Pilih kernel Python 3.10+ di pojok kanan atas notebook. Jika VS Code meminta memasang `ipykernel`, setujui.

### 4. Jalankan semua sel

Klik **Run All**, atau jalankan sel satu per satu dari atas.

- Sel pertama menjalankan `%pip install -r requirements.txt` (PySpark, pandas, matplotlib, scikit-learn). **Restart kernel** jika notebook memintanya setelah instalasi, lalu jalankan ulang dari atas.
- Sel inisialisasi membuat `SparkSession` (`spark`) dan `SparkContext` (`sc`).

> ⚠️ Jalankan sel **secara berurutan dari atas**. Variabel `spark` dan `sc` dibuat di sel inisialisasi, sehingga sel di bawahnya akan error (`NameError`) jika sel tersebut dilewati.

Jika ingin menjalankan per tahap, gunakan notebook di folder `test/` (urutan `01` → `02` → `03`). Notebook tersebut membaca data dari `../datasets/`, jadi `datasets/` tetap harus berada di root project.

## Alur Notebook

1. **Inisialisasi**: membuat `SparkSession` (mode `local[*]`, driver memory 4 GB).
2. **Data Ingestion**: membandingkan waktu baca data dengan RDD API vs DataFrame API.
3. **Data Exploration**: skema, statistik label, pengecekan *missing value* dan duplikat.
4. **Data Cleaning**: hapus null dan duplikat, `trim`, serta filter teks tidak valid.
5. **Feature Engineering**: kolom `text_length`, `length_category` (Short/Medium/Long), `sentiment`, dan `full_text` (gabungan title + text).
6. **Visualisasi**: distribusi sentimen, panjang teks, dan n-gram (hasil tersimpan di `images/`).
7. **Model**: `Tokenizer → (NGram) → HashingTF → IDF → Normalizer → LinearSVC`, dibandingkan untuk unigram (n=1) dan bigram (n=2).
8. **Evaluasi**: accuracy, F1, precision, recall, serta accuracy per kategori panjang teks.

## Tentang Dataset

- Sumber: [Kaggle — Amazon reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) (`train.csv` dan `test.csv`).
- Ulasan produk Amazon dengan dua kelas: skor 1–2 sebagai **negatif**, skor 4–5 sebagai **positif** (skor 3 tidak disertakan).
- Umumnya terdiri dari 3.600.000 data latih dan 400.000 data uji.
- Kolom: `label` (1 = negatif, 2 = positif), `title`, dan `text`.
- Notebook memberi nama kolom secara manual lewat `.toDF("label", "title", "text")`.

> **Catatan header:** `main.ipynb` membaca `train.csv` dengan `header=True` dan `test.csv` dengan `header=False`. Pastikan file hasil unduhan sesuai dengan itu. Jika `train.csv` kamu tidak memiliki baris header, ubah menjadi `header=False` agar baris pertama tidak terbuang. Indikasinya: output notebook melaporkan 3.599.999 baris latih, padahal dataset berisi 3.600.000 baris.

## Troubleshooting

**`JAVA_HOME is not set` / Java tidak ditemukan**
Pasang JDK 17 atau 21 dan pastikan `java -version` berjalan. Di Windows, jika masih bermasalah, *uncomment* sel pertama di `main.ipynb` dan isi variabel `JAVA` dengan path JDK kamu.

**`ModuleNotFoundError: No module named 'pyspark'`**
Instalasi di sel pertama belum selesai atau kernel belum di-restart. Jalankan ulang sel pertama, *restart kernel*, lalu jalankan dari atas. Pastikan kernel yang dipilih adalah Python yang sama dengan yang dipakai saat instalasi.

**`OutOfMemoryError` / Spark berjalan sangat lambat**
- Tutup aplikasi lain yang memakan RAM.
- Naikkan `spark.driver.memory` di sel inisialisasi (misalnya `"6g"`) jika RAM mencukupi.
- Arahkan direktori sementara Spark ke disk yang lega dengan meng-*uncomment* baris `os.makedirs(...)` dan `.config("spark.local.dir", ...)` di sel inisialisasi.

**`Path does not exist: datasets/train.csv`**
Folder `datasets/` belum dibuat, atau notebook dibuka dari folder yang salah. Buka folder project (bukan folder induknya) dan cek isi `datasets/`.

**`NameError: name 'spark' is not defined` / `'sc' is not defined`**
Sel inisialisasi belum dijalankan. Jalankan ulang dari atas, atau gunakan *Restart & Run All*.
