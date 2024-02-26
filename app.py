import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model dan skaler yang telah disimpan sebelumnya
scaler = pickle.load(open('scaler.pkl', 'rb'))
best_model = pickle.load(open('ANN_best_model.pkl', 'rb'))

# Fungsi untuk melakukan prediksi
def predict_status(data):
    # Standardisasi data
    data_scaled = scaler.transform(data)
    # Prediksi hasil Status
    hasil_prediksi = best_model.predict(data_scaled)
    hasil_prediksi = int(hasil_prediksi)
    # Mapping hasil prediksi ke label yang sesuai
    if hasil_prediksi == 0:
        status = "Dropout"
    elif hasil_prediksi == 1:
        status = "Enrolled"
    else:
        status = "Graduate"
    return status

# Membaca dataset dengan pemisah ;
data_institut = pd.read_csv('data_institut.csv', sep=';')

# Judul dan deskripsi aplikasi
st.title('Prediksi Status Mahasiswa')
st.write('Aplikasi ini memprediksi status seorang mahasiswa berdasarkan fitur-fitur tertentu.')

# Input fitur-fitur dari pengguna
st.sidebar.title('Masukkan Fitur-Fitur Mahasiswa')
Curricular_units_1st_sem_enrolled = st.sidebar.slider("SKS yang Didaftarkan Semester 1", 0, 26, 0)
Curricular_units_1st_sem_approved = st.sidebar.slider("SKS yang Lulus Semester 1", 0, 26, 0)
Curricular_units_1st_sem_grade = st.sidebar.slider("Nilai Semester 1", 0.0, 4.0, 0.0, 0.01)

Curricular_units_2nd_sem_enrolled = st.sidebar.slider("SKS yang Didaftarkan Semester 2", 0, 23, 0)
Curricular_units_2nd_sem_approved = st.sidebar.slider("SKS yang Lulus Semester 2", 0, 20, 0)
Curricular_units_2nd_sem_grade = st.sidebar.slider("Nilai Semester 2", 0.0, 4.0, 0.0, 0.01)

Tuition_fees_up_to_date = st.sidebar.radio("Pelunasan Uang Pendidikan", ("Tidak", "Iya"))
Scholarship_holder = st.sidebar.radio("Penerima Beasiswa", ("Tidak", "Iya"))
Admission_grade = st.sidebar.slider("Nilai Penerimaan", 0.0, 200.0, 0.0)
Displaced = st.sidebar.radio("Mahasiswa Terlantar?", ("Tidak", "Iya"))

# Tombol untuk melakukan prediksi saat diklik
if st.sidebar.button('Prediksi'):
    # Ubah input pengguna menjadi array numpy
    data = np.array([[Curricular_units_1st_sem_enrolled, Curricular_units_1st_sem_approved, Curricular_units_1st_sem_grade,
                      Curricular_units_2nd_sem_enrolled, Curricular_units_2nd_sem_approved, Curricular_units_2nd_sem_grade,
                      1 if Tuition_fees_up_to_date == "Iya" else 0, 1 if Scholarship_holder == "Iya" else 0, Admission_grade,
                      1 if Displaced == "Iya" else 0]])
    # Lakukan prediksi menggunakan fungsi predict_status
    status = predict_status(data)
    # Tampilkan hasil prediksi di sidebar
    st.sidebar.success(f'Hasil Prediksi: {status}')

# Menampilkan dataframe di Streamlit
st.write('## Data Institut')
st.write(data_institut)

# Memilih kolom untuk visualisasi
selected_columns = st.multiselect('Pilih Kolom untuk Visualisasi', data_institut.columns)

# Memilih jenis plot
plot_type = st.selectbox('Pilih Jenis Plot', ['Bar Chart', 'Pie Chart', 'Line Chart'])

# Membuat visualisasi berdasarkan pilihan pengguna
if selected_columns and plot_type:
    data_to_plot = data_institut[selected_columns]

    if plot_type == 'Bar Chart':
        st.write('### Bar Chart')
        for column in data_to_plot.columns:
            st.write(f"#### {column}")
            fig, ax = plt.subplots()
            sns.countplot(data=data_institut, x=column, hue='Status', ax=ax)
            st.pyplot(fig)

    elif plot_type == 'Pie Chart':
        st.write('### Pie Chart')
        for column in data_to_plot.columns:
            st.write(f"#### {column}")
            fig, ax = plt.subplots()
            data_grouped = data_institut.groupby(column)['Status'].value_counts().unstack(fill_value=0)
            data_grouped.plot.pie(ax=ax, subplots=True, figsize=(6, 6), autopct='%1.1f%%')
            st.pyplot(fig)

    elif plot_type == 'Line Chart':
        st.write('### Line Chart')
        for column in selected_columns:
            if column != 'Status':
                st.write(f"#### {column}")
                fig, ax = plt.subplots()
                sns.lineplot(data=data_institut, x=column, y='Status', ax=ax)
                st.pyplot(fig)
