# Klasifikasi Tumor Otak Menggunakan MRI (ResNet50)

Proyek ini bertujuan untuk mengklasifikasikan jenis tumor otak berdasarkan citra MRI menggunakan model *deep learning* **ResNet50**. Proyek ini diimplementasikan menggunakan PyTorch dan membandingkan performa model pada dua skenario dataset: **Imbalanced** (tidak seimbang) dan **Balanced** (seimbang).

## Struktur Repositori

- `train.py` : *Script* utama untuk melatih (*training*) dan mengevaluasi model ResNet50. *Script* ini akan secara otomatis melatih model untuk skenario data imbalanced dan balanced, serta menyimpan model dan visualisasi evaluasi.
- `augmentasi.py` : *Script* untuk melakukan augmentasi gambar untuk menyeimbangkan kelas pada dataset (menghasilkan dataset balanced).
- `distribusi.py` : *Script* untuk memvisualisasikan dan mengecek distribusi jumlah sampel dari setiap kelas pada dataset.
- `dataset/` : Direktori yang berisi data MRI untuk pelatihan dan pengujian. (Struktur standar berisi sub-folder seperti `Testing`, `Imbalanced`, dan `Balanced`).
- `results/` : Direktori yang dihasilkan otomatis setelah menjalankan `train.py`, berisi grafik loss, akurasi, metrik evaluasi, serta visualisasi model.
- `laporan_brain_tumor.docx` : Laporan lengkap dari hasil proyek ini.

## Fitur dan Evaluasi

Proyek ini dilengkapi dengan berbagai metrik dan visualisasi untuk mengevaluasi performa klasifikasi tumor secara komprehensif, di antaranya:
- **Grafik Accuracy & Loss**: Menampilkan tren akurasi dan kerugian selama proses training (per epoch).
- **Confusion Matrix**: Matriks kebingungan untuk mengevaluasi klasifikasi per kelas.
- **Evaluation Metrics per Class**: Bar chart untuk Accuracy, Precision, Recall, Specificity, dan F1-Score per kelas.
- **ROC / AUC Curve**: Kurva *Receiver Operating Characteristic* dan nilai AUC.
- **Grad-CAM & Score-CAM**: Visualisasi *Class Activation Mapping* untuk melihat bagian otak mana yang menjadi fokus model dalam mengambil keputusan prediksi.

## Persyaratan (Requirements)

Pastikan Anda telah menginstal pustaka-pustaka Python berikut (disarankan menggunakan environment seperti `venv` atau `conda`):

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
pip install grad-cam # Opsional, diperlukan untuk visualisasi Grad-CAM
```

## Cara Menjalankan

1. **Persiapan Data**: Unduh dataset dari [Kaggle - Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri?resource=download) dan pastikan Anda mengatur folder `dataset/` dengan pembagian dataset yang benar.
   - Folder `dataset/Testing`
   - Folder `dataset/Imbalanced`
   - Folder `dataset/Balanced` (Dapat di-generate menggunakan `augmentasi.py` jika belum ada)
2. **Cek Distribusi Data (Opsional)**:
   ```bash
   python distribusi.py
   ```
3. **Training Model**:
   Jalankan file `train.py` untuk mulai melatih model pada kedua skenario.
   ```bash
   python train.py
   ```
   *Script* akan menampilkan proses training di terminal dan menyimpan semua visualisasi grafik ke dalam folder `results/imbalanced/` dan `results/balanced/`.

## Hasil
Hasil prediksi, grafik evaluasi, dan file model berformat `.pth` akan secara otomatis disimpan di dalam direktori `results/` setelah proses pelatihan selesai.
