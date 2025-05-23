# Prediksi Harga Pangan Indonesia

Aplikasi web ini dirancang untuk memprediksi harga pangan di Indonesia dengan memanfaatkan model machine learning Decision Tree Regressor. Tujuan utama pengembangan aplikasi ini adalah untuk memberikan informasi prediktif yang dapat membantu pengguna dalam mengambil keputusan berdasarkan data harga pangan. Model ini memiliki performa sangat baik dengan RMSE sebesar 1.649,44, MAE 579,07, nilai R² 0,9976, dan MAPE 1,38%, yang menunjukkan akurasi tinggi dan kesalahan prediksi rendah. Hasil prediksi ditampilkan dalam bentuk grafik dan peta interaktif berbasis web menggunakan Streamlit. Kombinasi akurasi dan kecepatan eksekusi membuat Decision Tree menjadi pilihan optimal untuk analisis harga pangan.

## Fitur

- Prediksi harga per hari untuk tiap provinsi
- Visualisasi geografis menggunakan peta interaktif
- Grafik timeline tren harga pangan
- Tampilan data hasil prediksi yang dapat diunduh (downloadable)
- Riwayat prediksi yang tersimpan di website

## Tools

- Python  
- Streamlit  
- GeoPandas  
- Plotly  
- Pandas  
- Scikit-learn  
- GitHub

## Cara Kerja Website

1. Buka aplikasi melalui link [Aplikasi Prediksi Harga Komoditas](https://app-commodity-price-prediction.streamlit.app/)
2. User memilih **Provinsi**, **Komoditas**, dan **Tanggal, Bulan, dan Tahun** prediksi di sidebar.  
3. Saat tombol "Prediksi & Visualisasi" ditekan:  
   - Sistem melakukan prediksi harga menggunakan model Decision Tree yang sudah dilatih.  
   - Menampilkan hasil prediksi harga untuk tanggal dan provinsi terpilih.  
   - Membuat peta interaktif harga pangan di seluruh provinsi untuk tanggal dan komoditas tersebut.  
   - Menampilkan grafik timeline prediksi harga selama satu bulan untuk provinsi dan komoditas terpilih.  
   - Menyimpan prediksi terbaru ke dalam riwayat di memori sesi.  
4. User dapat melihat **data ringkasan** dan **riwayat prediksi** melalui panel yang dapat diperluas (expander).  
5. Semua tabel data prediksi dan ringkasan bisa diunduh sebagai file CSV oleh user.  

## Fungsi-Fungsi Utama

```python
def load_model():
    """
    Memuat model Decision Tree serta file encoding provinsi dan komoditas.
    Mengembalikan tuple: (model, prov_map, com_map)
    """

def visualisasi_peta(df_prediksi, commodity, tanggal, bulan, tahun):
    """
    Menampilkan peta interaktif harga pangan berdasarkan dataframe prediksi setiap provinsi.
    """

def grafik_timeline(model, prov_map, com_map, selected_prov, commodity, tahun, bulan):
    """
    Menampilkan grafik garis harga selama satu bulan penuh untuk provinsi dan komoditas yang dipilih.
    """

def simpan_prediksi_ke_history(tanggal, bulan, tahun, selected_prov, commodity, harga_prediksi):
    """
    Menyimpan hasil prediksi ke session_state sebagai riwayat.
    """

def run(self):
    """
    Fungsi utama aplikasi untuk menampilkan UI, input, dan hasil prediksi.
    Menyusun seluruh komponen antarmuka Streamlit.
    """
