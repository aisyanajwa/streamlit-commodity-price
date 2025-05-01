import streamlit as st
import pandas as pd
import geopandas as gpd
import joblib
import calendar
from datetime import date
import matplotlib.pyplot as plt

# ===============================
# Fungsi load model dan data
# ===============================
@st.cache_data
def load_model():
    model = joblib.load("trained_decision_tree_model.pkl")
    df = pd.read_csv("harga_pangan_encoded.csv")

    prov_map = dict(df[['Provinsi', 'Provinsi_encoded']].drop_duplicates().values)
    com_map = dict(df[['Commodity', 'Commodity_encoded']].drop_duplicates().values)

    return model, prov_map, com_map

# ===============================
# Fungsi visualisasi peta dengan GeoPandas
# ===============================
def visualisasi_peta(df_prediksi, commodity, tanggal, bulan, tahun):
    provinsi_mapping = {
        'DI Yogyakarta': 'Yogyakarta',
        'DKI Jakarta': 'JakartaRaya',
        'Jawa Barat': 'JawaBarat',
        'Jawa Tengah': 'JawaTengah',
        'Jawa Timur': 'JawaTimur',
        'Kalimantan Barat': 'KalimantanBarat',
        'Kalimantan Selatan': 'KalimantanSelatan',
        'Kalimantan Tengah': 'KalimantanTengah',
        'Kalimantan Timur': 'KalimantanTimur',
        'Kalimantan Utara': 'KalimantanUtara',
        'Kepulauan Bangka Belitung': 'BangkaBelitung',
        'Kepulauan Riau': 'KepulauanRiau',
        'Maluku Utara': 'MalukuUtara',
        'Nusa Tenggara Barat': 'NusaTenggaraBarat',
        'Nusa Tenggara Timur': 'NusaTenggaraTimur',
        'Papua Barat': 'PapuaBarat',
        'Sulawesi Barat': 'SulawesiBarat',
        'Sulawesi Selatan': 'SulawesiSelatan',
        'Sulawesi Tengah': 'SulawesiTengah',
        'Sulawesi Tenggara': 'SulawesiTenggara',
        'Sulawesi Utara': 'SulawesiUtara',
        'Sumatera Barat': 'SumateraBarat',
        'Sumatera Selatan': 'SumateraSelatan',
        'Sumatera Utara': 'SumateraUtara'
    }
    df_prediksi['Provinsi'] = df_prediksi['Provinsi'].map(provinsi_mapping).fillna(df_prediksi['Provinsi'])

    # Membaca file JSON provinsi Indonesia
    gdf = gpd.read_file("gadm41_IDN_1.json")  # Pastikan file JSON ini benar-benar ada di direktori Anda
    gdf = gdf.rename(columns={'NAME_1': 'Provinsi'})  # Pastikan nama kolom yang ada di file JSON sesuai
    gdf_merged = gdf.merge(df_prediksi, on='Provinsi', how='left')

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    gdf_merged.plot(
        column="Harga",
        cmap="viridis_r",
        legend=True,
        edgecolor="black",
        missing_kwds={"color": "lightgrey", "label": "No data"},
        ax=ax
    )
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)

# ===============================
# Fungsi grafik timeline
# ===============================
def grafik_timeline(model, prov_map, com_map, selected_prov, commodity, tahun, bulan):
    hari_terakhir = calendar.monthrange(tahun, bulan)[1]
    tanggal_list = list(range(1, hari_terakhir + 1))

    harga_list = []
    for tgl in tanggal_list:
        input_df = pd.DataFrame({
            'Commodity_encoded': [com_map[commodity]],
            'Provinsi_encoded': [prov_map[selected_prov]],
            'Year': [tahun],
            'Month': [bulan],
            'Day': [tgl]
        })
        harga = model.predict(input_df)[0]
        harga_list.append(harga)

    harga_tertinggi = max(harga_list)
    harga_terendah = min(harga_list)

    # Menggunakan st.line_chart untuk menggantikan plt
    df_timeline = pd.DataFrame({"Tanggal": tanggal_list, "Harga": harga_list})
    st.line_chart(df_timeline.set_index('Tanggal'))

    # Menampilkan harga tertinggi dan terendah
    st.write(f"Harga Tertinggi: Rp {harga_tertinggi:,.2f}")
    st.write(f"Harga Terendah: Rp {harga_terendah:,.2f}")

    df_ringkas = pd.DataFrame({"Tanggal": tanggal_list, "Harga": harga_list})
    with st.expander("📋 Lihat Ringkasan Prediksi Bulanan"):
        st.dataframe(df_ringkas, use_container_width=True)

# ===============================
# Main App
# ===============================
def main():
    st.title("📊 Prediksi Harga Pangan per Provinsi di Indonesia")

    model, prov_map, com_map = load_model()

    provinsi_list = list(prov_map.keys())
    commodity_list = list(com_map.keys())

    selected_prov = st.sidebar.selectbox("Pilih Provinsi", options=provinsi_list)
    commodity = st.sidebar.selectbox("Pilih Komoditas", options=commodity_list)
    tahun = st.sidebar.number_input("Tahun", min_value=2025, max_value=2026, value=2025)
    bulan = st.sidebar.number_input("Bulan", min_value=1, max_value=12, value=1)
    tanggal = st.sidebar.number_input("Tanggal", min_value=1, max_value=31, value=1)

    if st.sidebar.button("Prediksi & Visualisasi"):
        st.subheader(f"📅 Prediksi Harga Komoditas {commodity} pada {tanggal}/{bulan}/{tahun}")

        input_df = pd.DataFrame({
            'Commodity_encoded': [com_map[commodity]],
            'Provinsi_encoded': [prov_map[selected_prov]],
            'Year': [tahun],
            'Month': [bulan],
            'Day': [tanggal]
        })
        harga_prediksi = model.predict(input_df)[0]

        st.success(f"💰 Prediksi harga pangan pada {tanggal}/{bulan}/{tahun} di **{selected_prov}** untuk komoditas **{commodity}** adalah: **Rp {harga_prediksi:,.2f}**")

        data_prediksi = []
        for prov, prov_enc in prov_map.items():
            input_df = pd.DataFrame({
                'Commodity_encoded': [com_map[commodity]],
                'Provinsi_encoded': [prov_enc],
                'Year': [tahun],
                'Month': [bulan],
                'Day': [tanggal]
            })
            harga = model.predict(input_df)[0]
            data_prediksi.append({'Provinsi': prov, 'Harga': harga})

        df_prediksi = pd.DataFrame(data_prediksi)
        st.subheader(f"Peta Perbandingan Prediksi Harga Setiap Provinsi")
        st.markdown(f"<h5>{commodity} - {tanggal}/{bulan}/{tahun}", unsafe_allow_html=True)
        visualisasi_peta(df_prediksi, commodity, tanggal, bulan, tahun)

        with st.expander("📋 Lihat Ringkasan Prediksi Harian per Provinsi"):
            st.dataframe(df_prediksi.sort_values(by="Harga", ascending=False), use_container_width=True)

        st.subheader("Grafik Prediksi Harga (1 Bulan) ")
        st.markdown(f"<h5> {commodity} di {selected_prov}, {calendar.month_name[bulan]} {tahun}", unsafe_allow_html=True)
        grafik_timeline(model, prov_map, com_map, selected_prov, commodity, tahun, bulan)

        st.markdown("""
    **Keterangan:**
    Prediksi harga komoditas ini dibuat berdasarkan data harga pangan yang terkumpul selama tahun 2022 hingga 2024. 
    Model prediksi ini menggunakan model Decision Tree untuk memproyeksikan harga berdasarkan pola dan tren historis yang ada. 
    Hasil prediksi ini dapat digunakan untuk merencanakan kebijakan pangan atau membantu dalam keputusan bisnis.
    """)

# ===============================
# Run app
# ===============================
if __name__ == "__main__":
    main()
