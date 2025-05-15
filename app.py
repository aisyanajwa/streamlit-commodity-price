import streamlit as st
import pandas as pd
import geopandas as gpd
import joblib
import calendar
from datetime import date
import matplotlib.pyplot as plt
import plotly.express as px
import json

class PricePredictionApp:
    def __init__(self):
        self.model, self.prov_map, self.com_map = self.load_model()

    @staticmethod
    @st.cache_data
    def load_model():
        """
        Memuat model Decision Tree serta file encoding provinsi dan komoditas.
            - model: Model DecisionTreeRegressor
            - prov_map: Mapping nama provinsi
            - com_map: Mapping nama komoditas
        """
        model = joblib.load("trained_decision_tree_model.pkl")
        df = pd.read_csv("harga_pangan_encoded.csv")

        prov_map = dict(df[['Provinsi', 'Provinsi_encoded']].drop_duplicates().values)
        com_map = dict(df[['Commodity', 'Commodity_encoded']].drop_duplicates().values)

        return model, prov_map, com_map

    @staticmethod
    def visualisasi_peta(df_prediksi, commodity, tanggal, bulan, tahun):
        """
        Menampilkan peta Plotly harga pangan berdasarkan prediksi.

            df_prediksi (DataFrame): DataFrame hasil prediksi
            commodity (str): Nama komoditas
            tanggal (int): Tanggal prediksi
            bulan (int): Bulan prediksi
            tahun (int): Tahun prediksi
        """
        
        with open("gadm41_IDN_1.json", "r", encoding="utf-8") as f:
            geojson = json.load(f)

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

        df_prediksi["Provinsi"] = df_prediksi["Provinsi"].map(provinsi_mapping).fillna(df_prediksi["Provinsi"])

        fig = px.choropleth(
            df_prediksi,
            geojson=geojson,
            locations="Provinsi",
            featureidkey="properties.NAME_1",
            color="Harga",
            color_continuous_scale="Viridis_r",
            labels={'Harga': 'Harga (Rp)'}
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def grafik_timeline(model, prov_map, com_map, selected_prov, commodity, tahun, bulan):
        """
        Menampilkan grafik garis harga pangan selama satu bulan penuh.

            prov_map: Mapping nama provinsi
            com_map: Mapping nama komoditas
            selected_prov (str): Provinsi yang dipilih
            commodity (str): Komoditas yang dipilih
            tahun (int): Tahun
            bulan (int): Bulan
        """
        
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
            harga_list.append((tgl, harga))

        harga_tertinggi_tanggal = max(harga_list, key=lambda x: x[1])
        harga_terendah_tanggal = min(harga_list, key=lambda x: x[1])

        harga_tertinggi = harga_tertinggi_tanggal[1]
        tanggal_tertinggi = harga_tertinggi_tanggal[0]
        
        harga_terendah = harga_terendah_tanggal[1]
        tanggal_terendah = harga_terendah_tanggal[0]

        df_timeline = pd.DataFrame({"Tanggal": tanggal_list, "Harga": [x[1] for x in harga_list]})
        st.line_chart(df_timeline.set_index('Tanggal'))

        st.write(f"Harga Tertinggi: Rp {harga_tertinggi:,.2f} pada {tanggal_tertinggi}/{bulan}/{tahun}")
        st.write(f"Harga Terendah: Rp {harga_terendah:,.2f} pada {tanggal_terendah}/{bulan}/{tahun}")

        df_ringkas = pd.DataFrame({"Tanggal": tanggal_list, "Harga": [x[1] for x in harga_list]})
        with st.expander(f"📋 Lihat Ringkasan Prediksi Bulanan pada {bulan}/{tahun}"):
            st.dataframe(df_ringkas, use_container_width=True, hide_index=True)

    @staticmethod
    def simpan_prediksi_ke_history(tanggal, bulan, tahun, selected_prov, commodity, harga_prediksi):
        """
        Menyimpan hasil prediksi ke dalam session_state sebagai riwayat.

            tanggal (int): Tanggal prediksi
            bulan (int): Bulan prediksi
            tahun (int): Tahun prediksi
            selected_prov (str): Provinsi yang dipilih
            commodity (str): Komoditas
            harga_prediksi (float): Nilai harga hasil prediksi
        """
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        prediksi_baru = {
            'Tanggal': tanggal,
            'Bulan': bulan,
            'Tahun': tahun,
            'Provinsi': selected_prov,
            'Komoditas': commodity,
            'Harga': harga_prediksi
        }
        st.session_state.history.append(prediksi_baru)

    def run(self):
        """
        Fungsi utama aplikasi untuk menampilkan UI, input, dan hasil prediksi.
        Menyusun seluruh komponen antarmuka Streamlit.
        """
        st.title("Prediksi Harga Pangan per Provinsi di Indonesia")

        provinsi_list = list(self.prov_map.keys())
        commodity_list = list(self.com_map.keys())

        col1, col2 = st.columns(2)

        with col1:
            selected_prov = st.selectbox("Pilih Provinsi", options=provinsi_list)

        with col2:
            commodity = st.selectbox("Pilih Komoditas", options=commodity_list)

        col3, col4, col5 = st.columns(3)

        with col3:
            tahun = st.number_input("Tahun", min_value=2025, max_value=2026, value=2025)

        with col4:
            bulan = st.number_input("Bulan", min_value=1, max_value=12, value=1)

        with col5:
            tanggal = st.number_input("Tanggal", min_value=1, max_value=31, value=1)

        if st.button("Prediksi & Visualisasi"):
            input_df = pd.DataFrame({
                'Commodity_encoded': [self.com_map[commodity]],
                'Provinsi_encoded': [self.prov_map[selected_prov]],
                'Year': [tahun],
                'Month': [bulan],
                'Day': [tanggal]
            })
            harga_prediksi = self.model.predict(input_df)[0]

            st.success(f"💰 Prediksi harga pangan pada {tanggal}/{bulan}/{tahun} di **{selected_prov}** untuk komoditas **{commodity}** adalah: **Rp {harga_prediksi:,.2f}**")

            self.simpan_prediksi_ke_history(tanggal, bulan, tahun, selected_prov, commodity, harga_prediksi)

            data_prediksi = []
            for prov, prov_enc in self.prov_map.items():
                input_df = pd.DataFrame({
                    'Commodity_encoded': [self.com_map[commodity]],
                    'Provinsi_encoded': [prov_enc],
                    'Year': [tahun],
                    'Month': [bulan],
                    'Day': [tanggal]
                })
                harga = self.model.predict(input_df)[0]
                data_prediksi.append({'Provinsi': prov, 'Harga': harga})

            df_prediksi = pd.DataFrame(data_prediksi)
            st.subheader(f"Peta Perbandingan Prediksi Harga Setiap Provinsi")
            st.markdown(f"<h5>Prediksi Harga {commodity} pada {tanggal}/{bulan}/{tahun}", unsafe_allow_html=True)
            self.visualisasi_peta(df_prediksi, commodity, tanggal, bulan, tahun)

            with st.expander("📋 Lihat Ringkasan Prediksi Harian per Provinsi"):
                st.dataframe(df_prediksi.sort_values(by="Harga", ascending=False), use_container_width=True, hide_index=True)

            st.subheader("Grafik Prediksi Harga (1 Bulan) ")
            st.markdown(f"<h5> {commodity} di {selected_prov} pada {calendar.month_name[bulan]} {tahun}", unsafe_allow_html=True)
            self.grafik_timeline(self.model, self.prov_map, self.com_map, selected_prov, commodity, tahun, bulan)

            if 'history' in st.session_state:
                with st.expander("📜 Riwayat Prediksi Harga"):
                    history_df = pd.DataFrame(st.session_state.history)
                    st.dataframe(history_df, use_container_width=True, hide_index=True)

            st.markdown("""
**Keterangan:**
Prediksi harga komoditas ini dibuat berdasarkan data harga pangan yang terkumpul selama tahun 2022 hingga 2024. 
Model prediksi ini menggunakan model Decision Tree untuk memproyeksikan harga berdasarkan pola dan tren historis yang ada. 
Hasil prediksi ini dapat digunakan untuk merencanakan kebijakan pangan atau membantu dalam keputusan bisnis.
""")

if __name__ == "__main__":
    app = PricePredictionApp()
    app.run()
