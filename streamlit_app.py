#Import libraries
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
import streamlit_authenticator as stauth
from sklearn.cluster import KMeans
from io import BytesIO
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly as py
import plotly.graph_objs as go
import time
import streamlit as st

# Membuat session state dengan dictionary kosong
state = st.session_state.setdefault('state', {})

# Menambahkan key 'data' pada session state jika belum ada
if 'df1' not in state:
   state['df1'] = pd.DataFrame()

if 'dataset' not in state:
   state['dataset'] = pd.DataFrame()

# Fungsi judul halaman  
def judul_halaman(header, subheader):
    nama_app = f"Aplikasi {header} Pengelompokkan Kemampuan Kompetensi BKKLA"
    st.title(nama_app)
    st.subheader(subheader)

# Fungsi untuk Mengimport Data
def import_data():
    
    # Membuat st.file_uploader untuk file pertama
    uploaded_file = st.file_uploader("Masukkan Data Rekap Kuisioner Alumni", type="xlsx")

    # Menggabungkan file jika keduanya sudah di-upload
    if uploaded_file is not None:
        try:        
            RKA = pd.read_excel(uploaded_file)
            data = RKA
            #data.index = data.index+1
            data = data[data['Jenjang'] == 'S1']
            data.reset_index(inplace=True)
            data.index = data.index + 1
            data = data.replace(0, pd.np.nan)
            data[['nimhsmsmh','tahun_lulus']] = data[['nimhsmsmh','tahun_lulus']].astype(str)
            df1 = data[['nimhsmsmh','nmmhsmsmh','Prodi','tahun_lulus','f1761','f1763','f1765','f1767','f1769','f1771','f1773']]
            df2 = df1.rename(columns = {'nimhsmsmh':'NIM','nmmhsmsmh':'Nama',
                                         'f1761':'Etika','f1763':'Keahlian_Bidang_Ilmu',
                                         'f1765':'Bahasa_Inggris',
                                         'f1767':'Penggunaan_Teknologi_Informasi',
                                         'f1769':'Komunikasi','f1771':'Kerjasama_Tim',
                                         'f1773':'Pengembangan_Diri'})
            kolomdanbaris = df2.shape
        
            # Menampilkan pesan sukses
            st.success("Import Data Berhasil")

            # Menampilkan tabel hasil penggabungan di Streamlit
            st.write('Dari data di dapatkan ', df2.shape[0],' Baris dan ', df2.shape[1],' Kolom')
            state['dataset'] = df2

        except(TypeError, IndexError, KeyError):
            # Menampilkan pesan kesalahan jika terjadi masalah saat mengimpor data
            st.error("Data yang diupload tidak sesuai")

# Fungsi untuk melakukan cleaning data
def preprocessing_data():
    df2 = state['dataset']
    if not df2.empty:
        df2[['tahun_lulus','NIM']] = df2[['tahun_lulus','NIM']].astype(str)
        df2.reset_index(inplace=True, drop=True)
        df2.dropna(how='any', axis=0, inplace=True)
        
        dfbin = df2
        #calculate interquartile range
        q11, q21, q31 = np.percentile(dfbin.iloc[:,[4]], [25 , 50, 75])
        iqr1 = q31 - q11
        bbawah1 = q11 - 1.5 * iqr1
        batas1 = q31 + 1.5 * iqr1

        q12, q22, q32 = np.percentile(dfbin.iloc[:,[5]], [25 , 50, 75])
        iqr2 = q32 - q12
        bbawah2 = q12 - 1.5 * iqr2
        batas2 = q32 + 1.5 * iqr2

        q13, q23, q33 = np.percentile(dfbin.iloc[:,[6]], [25 , 50, 75])
        iqr3 = q33 - q13
        bbawah3 = q13 - 1.5 * iqr3
        batas3 = q33 + 1.5 * iqr3

        q14, q24, q34 = np.percentile(dfbin.iloc[:,[7]], [25 , 50, 75])
        iqr4 = q34 - q14
        bbawah4 = q14 - 1.5 * iqr4
        batas4 = q34 + 1.5 * iqr4

        q15, q25, q35 = np.percentile(dfbin.iloc[:, [8]], [25 , 50, 75])
        iqr5 = q35 - q15
        bbawah5 = q15 - 1.5 * iqr5
        batas5 = q35 + 1.5 * iqr5

        q16, q26, q36 = np.percentile(dfbin.iloc[:,[9]], [25 , 50, 75])
        iqr6 = q36 - q16
        bbawah6 = q16 - 1.5 * iqr6
        batas6 = q36 + 1.5 * iqr6

        q17, q27, q37 = np.percentile(dfbin.iloc[:,[10]], [25 , 50, 75])
        iqr7 = q37 - q17
        bbawah7 = q17 - 1.5 * iqr7
        batas7 = q37 + 1.5 * iqr7

        outlieretika = dfbin.loc[(dfbin['Etika'] > batas1) | (dfbin['Etika'] < bbawah1)]
        sizeoutlieretika = outlieretika.index.size

        outlierbidangilmu = dfbin.loc[(dfbin['Keahlian_Bidang_Ilmu'] > batas2) | (dfbin['Keahlian_Bidang_Ilmu'] < bbawah2)]
        sizeoutlierbidangilmu = outlierbidangilmu.index.size

        outlierbahasainggris = dfbin.loc[(dfbin['Bahasa_Inggris'] > batas3) | (dfbin['Bahasa_Inggris'] < bbawah3)]
        sizeoutlierbahasainggris = outlierbahasainggris.index.size

        outlieroteknologiinformasi = dfbin.loc[(dfbin['Penggunaan_Teknologi_Informasi'] > batas4) | (dfbin['Penggunaan_Teknologi_Informasi'] < bbawah4)]
        sizeoutlierpteknologiinformasi = outlieroteknologiinformasi.index.size

        outlierkomunikasi = dfbin.loc[(dfbin['Komunikasi'] > batas5) | (dfbin['Komunikasi'] < bbawah5)]
        sizeoutlierkomunikasi = outlierkomunikasi.index.size

        outlierkerjasamatim = dfbin.loc[(dfbin['Kerjasama_Tim'] > batas6) | (dfbin['Kerjasama_Tim'] < bbawah6)]
        sizeoutlierkerjasamatim = outlierkerjasamatim.index.size

        outlierpengembangandiri = dfbin.loc[(dfbin['Pengembangan_Diri'] > batas7) | (dfbin['Pengembangan_Diri'] < bbawah7)]
        sizeoutlierpengembangandiri = outlierpengembangandiri.index.size

        binetika = dfbin[['NIM','Etika']].copy()
        binebidangilmu = dfbin[['NIM','Keahlian_Bidang_Ilmu']].copy()
        binbahasainggris = dfbin[['NIM','Bahasa_Inggris']].copy()
        bininformasi = dfbin[['NIM','Penggunaan_Teknologi_Informasi']].copy()
        binkomunikasi = dfbin[['NIM','Komunikasi']].copy()
        binkerjasamatim = dfbin[['NIM','Kerjasama_Tim']].copy()
        binpengembangandiri = dfbin[['NIM','Pengembangan_Diri']].copy()

        dfbin.drop(['Etika','Keahlian_Bidang_Ilmu','Bahasa_Inggris','Penggunaan_Teknologi_Informasi', 'Komunikasi','Kerjasama_Tim','Pengembangan_Diri'], axis=1, inplace=True)
        
        # bin Etika
        row1 = binetika.index.size
        squareroot1 = np.sqrt(row1)
        binsum1 = np.round(squareroot1)
        qlabels1 = []
        for i in range(1,int(binsum1)+1):
            t1 = i
            qlabels1.append(t1)

        binetika = binetika.sort_values(by=['Etika'])
        binetika = binetika.reset_index(drop=True)

        # Membuat bin dengan jumlah anggota bin yang sama
        binsetika = pd.qcut(binetika.index, q=int(binsum1), labels=qlabels1)
        binetika['Bin'] = binsetika
        binetika['Etika'] = binetika.groupby('Bin')['Etika'].transform('mean')
        binetika.drop(['Bin'], axis=1, inplace=True)

        #Binning Bidang Ilmu
        row2 = binebidangilmu.index.size
        squareroot2 = np.sqrt(row2)
        binsum2 = np.round(squareroot2)
        qlabels2 = []
        for i in range(1,int(binsum2)+1):
            t2 = i
            qlabels2.append(t2)
        
        binebidangilmu = binebidangilmu.sort_values(by=['Keahlian_Bidang_Ilmu'])
        binebidangilmu = binebidangilmu.reset_index(drop=True)

        # Membuat bin dengan jumlah anggota bin yang sama
        binsebidangilmu = pd.qcut(binebidangilmu.index, q=int(binsum2), labels=qlabels2)
        binebidangilmu['Bin'] = binsebidangilmu

        binebidangilmu['Keahlian_Bidang_Ilmu'] = binebidangilmu.groupby('Bin')['Keahlian_Bidang_Ilmu'].transform('mean')
        binebidangilmu.drop(['Bin'], axis=1, inplace=True)

        #Binning_Bahasa_Inggris
        row3 = binbahasainggris.index.size
        squareroot3 = np.sqrt(row3)
        binsum3 = np.round(squareroot3)
        qlabels3 = []
        for i in range(1,int(binsum3)+1):
            t3 = i
            qlabels3.append(t3)
        
        binbahasainggris = binbahasainggris.sort_values(by=['Bahasa_Inggris'])
        binbahasainggris = binbahasainggris.reset_index(drop=True)

        # Membuat bin dengan jumlah anggota bin yang sama
        binsbahasainggris = pd.qcut(binbahasainggris.index, q=int(binsum3), labels=qlabels3)
        binbahasainggris['Bin'] = binsbahasainggris

        binbahasainggris['Bahasa_Inggris'] = binbahasainggris.groupby('Bin')['Bahasa_Inggris'].transform('mean')
        binbahasainggris.drop(['Bin'], axis=1, inplace=True)

        #Binning Penggunaan_TI
        row4 = bininformasi.index.size
        squareroot4 = np.sqrt(row4)
        binsum4 = np.round(squareroot4)

        qlabels4 = []
        for i in range(1,int(binsum4)+1):
            t4 = i
            qlabels4.append(t4)
        
        bininformasi = bininformasi.sort_values(by=['Penggunaan_Teknologi_Informasi'])
        bininformasi = bininformasi.reset_index(drop=True)

        # Membuat bin dengan jumlah anggota bin yang sama
        binsinformasi = pd.qcut(bininformasi.index, q=int(binsum4), labels=qlabels4)
        bininformasi['Bin'] = binsinformasi

        bininformasi['Penggunaan_Teknologi_Informasi'] = bininformasi.groupby('Bin')['Penggunaan_Teknologi_Informasi'].transform('mean')
        bininformasi.drop(['Bin'], axis=1, inplace=True)

        #binning komunikasi
        row5 = binkomunikasi.index.size
        squareroot5 = np.sqrt(row5)
        binsum5 = np.round(squareroot5)

        qlabels5 = []
        for i in range(1,int(binsum5)+1):
            t5 = i
            qlabels5.append(t5)


        binkomunikasi = binkomunikasi.sort_values(by=['Komunikasi'])
        binkomunikasi = binkomunikasi.reset_index(drop=True)

        # Membuat bin dengan jumlah anggota bin yang sama
        binskomunikasi = pd.qcut(binkomunikasi.index, q=int(binsum5), labels=qlabels5)
        binkomunikasi['Bin'] = binskomunikasi

        binkomunikasi['Komunikasi'] = binkomunikasi.groupby('Bin')['Komunikasi'].transform('mean')
        binkomunikasi.drop(['Bin'], axis=1, inplace=True)

        #binning Kerjasama Tim
        row6 = binkerjasamatim.index.size
        squareroot6 = np.sqrt(row6)
        binsum6 = np.round(squareroot6)

        qlabels6 = []
        for i in range(1,int(binsum6)+1):
            t6 = i
            qlabels6.append(t6)


        binkerjasamatim = binkerjasamatim.sort_values(by=['Kerjasama_Tim'])
        binkerjasamatim = binkerjasamatim.reset_index(drop=True)

        # Membuat bin dengan jumlah anggota bin yang sama
        binskerjasamatim = pd.qcut(binkerjasamatim.index, q=int(binsum6), labels=qlabels6)
        binkerjasamatim['Bin'] = binskerjasamatim

        binkerjasamatim['Kerjasama_Tim'] = binkerjasamatim.groupby('Bin')['Kerjasama_Tim'].transform('mean')
        binkerjasamatim.drop(['Bin'], axis=1, inplace=True)

        #Binning Pengembangan Diri
        row7 = binpengembangandiri.index.size
        squareroot7 = np.sqrt(row7)
        binsum7 = np.round(squareroot7)

        qlabels7 = []
        for i in range(1,int(binsum7)+1):
            t7 = i
            qlabels7.append(t7)

        binpengembangandiri = binpengembangandiri.sort_values(by=['Pengembangan_Diri'])
        binpengembangandiri = binpengembangandiri.reset_index(drop=True)

        # Membuat bin dengan jumlah anggota bin yang sama
        binspengembangandiri = pd.qcut(binpengembangandiri.index, q=int(binsum7), labels=qlabels7)
        binpengembangandiri['Bin'] = binspengembangandiri

        binpengembangandiri['Pengembangan_Diri'] = binpengembangandiri.groupby('Bin')['Pengembangan_Diri'].transform('mean')
        binpengembangandiri.drop(['Bin'], axis=1, inplace=True)

        #Gabungkan Hasil BINNING
        result = pd.concat([dfbin, binetika, binebidangilmu, binbahasainggris, bininformasi, binkomunikasi, binkerjasamatim, binpengembangandiri], axis =1, join="inner")
        result1 = result.iloc[:,[0,1,2,3,5,7,9,11,13,15,17]].reset_index()
        result1.drop(['index'], axis=1, inplace=True)


        result1['Etika'] = result1['Etika'].round(1)
        result1['Keahlian_Bidang_Ilmu'] = result1['Keahlian_Bidang_Ilmu'].round(1)
        result1['Bahasa_Inggris'] = result1['Bahasa_Inggris'].round(1)
        result1['Informasi'] = result1['Penggunaan_Teknologi_Informasi'].round(1)
        result1['Komunikasi'] = result1['Komunikasi'].round(1)
        result1['Kerjasama_Tim'] = result1['Kerjasama_Tim'].round(1)
        result1['Pengembangan_Diri'] = result1['Pengembangan_Diri'].round(1)
        

        state['dataset'] = result1

        state['dataset'].drop(columns='Informasi', inplace=True)

        # Mereset index
        state['dataset'].reset_index(inplace=True, drop=True)
        st.success("Nilai Null Berhasil Dihapus!")
        st.dataframe(state['dataset'])
    else:
        #st.warning("Tidak ada data yang diupload atau data kosong")
        st.error('Gaada')

        

# Menampilkan jumlah nilai null per atribut
def show_null_count():
    if not state['dataset'].empty:
        st.write("Jumlah nilai null per atribut:")
        st.table(state['dataset'].isnull().sum())
    else:
        st.warning("Tidak ada data yang diupload atau data kosong")

# Fungsi DBI
def DBI():
    if not state['dataset'].empty:
        # Memilih Atribut yang ingin digunakan pada DBI
        state['x'] = state['dataset'].iloc[:,4:13]
        scaler = StandardScaler()
        data1_scaled = scaler.fit_transform(state['x'])
        #hasil = pd.DataFrame(data1_scaled, columns = state['df2'].columns)

        # Perhitungan Evaluasi DBI
        result2 = {}

        for i in range(2, 3):
            kmeans = KMeans(n_clusters=i, random_state=30)
            labels = kmeans.fit_predict(data1_scaled)
            db_index = davies_bouldin_score(data1_scaled, labels)
            result2.update({i: db_index})
        result3 = pd.DataFrame(result2.values(), result2.keys())
        df = result3.idxmin().min()

        st.write("Dari data didapatkan rekomendasi pengelompokkan sebanyak ",df," Kelompok") 
     
        

# Fungsi Clustering Data
def clustering_data(input_c):
        try:
            state['dfi'] = {}
            state['clustering'] = {}

            state['clustering'] = state['dataset'].copy()
            state['x'] = state['clustering'].iloc[:, 4:13]
            algorithm = (KMeans(n_clusters = input_c, init='k-means++', random_state= 111, algorithm='elkan'))
            state['algoritma'] = algorithm.fit_predict(state['x'])

        except (ValueError, IndexError, OverflowError):
            st.error("Nilai Jumlah Cluster Tidak Valid")


def show_cluster(input_c):
        try:

            state['clustering']['Kelompok'] = pd.DataFrame(state['algoritma'])
            state['clustering']['Kelompok'] = state['clustering']['Kelompok']+1
            state['clustering'] = state['clustering'].sort_values(by='Prodi')
            state['clustering'] = state['clustering'].reset_index(drop=True)

            state['dfi'] = {}
            state['ratadata'] = {}
            state['rekomendasi'] = {}
            state['datarekomendasi'] = {}

            for i in range(1,input_c+1):
                state['dfi']["clustering{0}".format(i)] = state['clustering'].loc[state['clustering']['Kelompok'] == i+1-1]

                state['dfi']["clustering"+str(i+1-1)] = state['dfi']["clustering"+str(i+1-1)].reset_index(drop=True)
                state['dfi']["clustering"+str(i+1-1)].index += 1
                
                st.write('**Kelompok** ' + str(i))
                st.dataframe(state['dfi']["clustering"+str(i+1-1)])

            
            for i in range(1,input_c+1):

                ascending_order = True if i == 1 else False   
                grup = state['clustering'][state['clustering']['Kelompok'] == i]
                gk = pd.DataFrame(grup.groupby(['Prodi', 'Etika']).size(), columns=['Jumlah Alumni'])
                gk = gk.sort_values(by='Etika', ascending=ascending_order)
                st.write(f"Pengelompokan Etika berdasarkan prodi untuk Kelompok {i}:")
                st.dataframe(gk)
                
                gk1 = pd.DataFrame(grup.groupby(['Prodi', 'Bahasa_Inggris']).size(), columns=['Jumlah Alumni'])
                gk1 = gk1.sort_values(by='Bahasa_Inggris', ascending=ascending_order)
                st.write(f"Pengelompokan Bahasa Inggris berdasarkan prodi untuk Kelompok {i}:")
                st.dataframe(gk1)
                
                gk2 = pd.DataFrame(grup.groupby(['Prodi', 'Keahlian_Bidang_Ilmu']).size(), columns=['Jumlah Alumni'])
                gk2 = gk2.sort_values(by='Keahlian_Bidang_Ilmu', ascending=ascending_order)
                st.write(f"Pengelompokan Keahlian Bidang Ilmu berdasarkan prodi untuk Kelompok {i}:")
                st.dataframe(gk2)
                
                gk3 = pd.DataFrame(grup.groupby(['Prodi', 'Penggunaan_Teknologi_Informasi']).size(), columns=['Jumlah Alumni'])
                gk3 = gk3.sort_values(by='Penggunaan_Teknologi_Informasi', ascending=ascending_order)
                st.write(f"Pengelompokan Penggunaan Teknologi Informasi berdasarkan prodi untuk Kelompok {i}:")
                st.dataframe(gk3)
                
                gk4 = pd.DataFrame(grup.groupby(['Prodi', 'Komunikasi']).size(), columns=['Jumlah Alumni'])
                gk4 = gk4.sort_values(by='Komunikasi', ascending=ascending_order)
                st.write(f"Pengelompokan Komunikasi berdasarkan prodi untuk Kelompok {i}:")
                st.dataframe(gk4)
                
                gk5 = pd.DataFrame(grup.groupby(['Prodi', 'Kerjasama_Tim']).size(), columns=['Jumlah Alumni'])
                gk5 = gk5.sort_values(by='Kerjasama_Tim', ascending=ascending_order)
                st.write(f"Pengelompokan Kerjasama Tim berdasarkan prodi untuk Kelompok {i}:")
                st.dataframe(gk5)
                
                gk6 = pd.DataFrame(grup.groupby(['Prodi', 'Pengembangan_Diri']).size(), columns=['Jumlah Alumni'])
                gk6 = gk6.sort_values(by='Pengembangan_Diri', ascending=ascending_order)
                st.write(f"Pengelompokan Pengembangan Diri berdasarkan prodi untuk Kelompok {i}:")
                st.dataframe(gk6)

            rekomendasi = []

            for i in range(1,input_c+1):
                state['ratadata']["data{0}".format(i)] = state['dfi']["clustering"+str(i+1-1)][['Prodi','Etika','Bahasa_Inggris','Keahlian_Bidang_Ilmu','Penggunaan_Teknologi_Informasi','Komunikasi','Kerjasama_Tim','Pengembangan_Diri']]
                # st.write(state['ratadata']["data"+str(i+1-1)])

                etika = str(round(np.average(state['ratadata']["data"+str(i+1-1)]['Etika']),2))
                bing = str(round(np.average(state['ratadata']["data"+str(i+1-1)]['Bahasa_Inggris']),2))
                keahlian = str(round(np.average(state['ratadata']["data"+str(i+1-1)]['Keahlian_Bidang_Ilmu']),2))
                teknologi = str(round(np.average(state['ratadata']["data"+str(i+1-1)]['Penggunaan_Teknologi_Informasi']),2))
                komunikasi = str(round(np.average(state['ratadata']["data"+str(i+1-1)]['Komunikasi']),2))
                kerjasama = str(round(np.average(state['ratadata']["data"+str(i+1-1)]['Kerjasama_Tim']),2))
                pengembangan = str(round(np.average(state['ratadata']["data"+str(i+1-1)]['Pengembangan_Diri']),2))


                st.write('Kelompok',i)
                st.write('Rata-rata kemampuan Etika = ', etika)
                st.write('Rata-rata kemampuan Bahasa Inggris = ', bing)
                st.write('Rata-rata kemampuan Keahlian Bidang Ilmu = ', keahlian)
                st.write('Rata-rata kemampuan Penggunaan Teknologi Informasi = ', teknologi)
                st.write('Rata-rata kemampuan Komunikasi = ', komunikasi)
                st.write('Rata-rata kemampuan Kerjasama Tim = ', kerjasama)
                st.write('Rata-rata kemampuan Pengembangan Diri = ', pengembangan)  

                rowrekomendasi = [bing, keahlian, teknologi, pengembangan, etika, komunikasi, kerjasama, i]
                rekomendasi.append(rowrekomendasi)
                state['rekomendasi'] = pd.DataFrame(rekomendasi) 

            state['rekomendasi'].columns = ['a','b','c','d','e','f','g','h']
            state['rekomendasi'] = state['rekomendasi'].sort_values(by = ['a','b','c','d','e','f','g'], ascending = [True,True,True,True,True,True,True])
                       
            bing1 = str(state['rekomendasi']['a'].iloc[0])
            keahlian1 = str(state['rekomendasi']['b'].iloc[0])
            teknologi1 = str(state['rekomendasi']['c'].iloc[0])
            pengembangan1 = str(state['rekomendasi']['d'].iloc[0])
            etika1 = str(state['rekomendasi']['e'].iloc[0])
            komunikasi1 = str(state['rekomendasi']['f'].iloc[0])
            kerjasama1 = str(state['rekomendasi']['g'].iloc[0])
            kelompok = str(state['rekomendasi']['h'].iloc[0])


            #st.table(state['rekomendasi']) 
            st.write('**Kesimpulan Rekomendasi**')
            st.write('Kelompok yang kurang dalam kemampuan kompetensinya berada dalam kelompok ',kelompok,'. ', 
                     'Ini Bisa menjadi perhatian BKKLA untuk bisa membuat pelatihan atau seminar yang sesuai dengan kebutuhan alumni dari masing-masing Prodi yang ada di UNIKOM.')
            

            df4 = state['ratadata']["data"+kelompok]

            
            informatikaS12 = df4[df4['Prodi'] == 'Teknik Informatika S1']
            sisinfS12 = df4[df4['Prodi'] == 'Sistem Informasi S1']
            managS12 = df4[df4['Prodi'] == 'Manajemen S1']
            akunS12 = df4[df4['Prodi'] == 'Akuntansi S1']
            desainkomunikasiS12 = df4[df4['Prodi'] == 'Desain Komunikasi Visual S1']
            ilmukomS12 = df4[df4['Prodi'] == 'Ilmu Komunikasi S1']
            siskomS12 = df4[df4['Prodi'] == 'Sistem Komputer S1']
            teknikelekS12 = df4[df4['Prodi'] == 'Teknik Elektro S1']
            perencanaans12 = df4[df4['Prodi'] == 'Perencanaan Wilayah dan Kota S1']
            sastrainggS12 = df4[df4['Prodi'] == 'Sastra Inggris S1']
            sastrajeS12 = df4[df4['Prodi'] == 'Sastra Jepang S1']
            teknikarsiteS12 = df4[df4['Prodi'] == 'Teknik Arsitektur S1']
            ilmupems12 = df4[df4['Prodi'] == 'Ilmu Pemerintahan S1']
            desainintes12 = df4[df4['Prodi'] == 'Desain Interior S1']
            HIS12 = df4[df4['Prodi'] == 'Hubungan Internasional S1']
            tekniksipS12 = df4[df4['Prodi'] == 'Teknik Sipil S1']
            ilmuhuks12 = df4[df4['Prodi'] == 'Ilmu Hukum S1']
            teknikinduS12 = df4[df4['Prodi'] == 'Teknik Industri S1']

            ouput19 = informatikaS12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput20 = sisinfS12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput21 = managS12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput22 = akunS12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput23 = desainkomunikasiS12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput24 = ilmukomS12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput25 = siskomS12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput26 = teknikelekS12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput27 = perencanaans12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput28 = sastrainggS12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput29 = sastrajeS12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput30 = teknikarsiteS12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput31 = ilmupems12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput32 = desainintes12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput33 = HIS12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput34 = tekniksipS12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput35 = ilmuhuks12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]
            ouput36 = teknikinduS12[['Etika', 'Bahasa_Inggris', 'Keahlian_Bidang_Ilmu', 'Penggunaan_Teknologi_Informasi', 'Komunikasi', 'Kerjasama_Tim', 'Pengembangan_Diri']]


            hasil18 = {}
            hasil19 = {}
            hasil20 = {}
            hasil21 = {}
            hasil22 = {}
            hasil23 = {}
            hasil24 = {}
            hasil25 = {}
            hasil26 = {}
            hasil27 = {}
            hasil28 = {}
            hasil29 = {}
            hasil30 = {}
            hasil31 = {}
            hasil32 = {}
            hasil33 = {}
            hasil34 = {}
            hasil35 = {}


            for x in ouput19:
                hasil18[x] = np.round(np.average(ouput19[x]), 2)
            for x in ouput20:
                hasil19[x] = np.round(np.average(ouput20[x]), 2)
            for x in ouput21:
                hasil20[x] = np.round(np.average(ouput21[x]), 2)
            for x in ouput22:
                hasil21[x] = np.round(np.average(ouput22[x]), 2)
            for x in ouput23:
                hasil22[x] = np.round(np.average(ouput23[x]), 2)
            for x in ouput24:
                hasil23[x] = np.round(np.average(ouput24[x]), 2)
            for x in ouput25:
                hasil24[x] = np.round(np.average(ouput25[x]), 2)
            for x in ouput26:
                hasil25[x] = np.round(np.average(ouput26[x]), 2)
            for x in ouput27:
                hasil26[x] = np.round(np.average(ouput27[x]), 2)
            for x in ouput28:
                hasil27[x] = np.round(np.average(ouput28[x]), 2)
            for x in ouput29:
                hasil28[x] = np.round(np.average(ouput29[x]), 2)
            for x in ouput30:
                hasil29[x] = np.round(np.average(ouput30[x]), 2)
            for x in ouput31:
                hasil30[x] = np.round(np.average(ouput31[x]), 2)
            for x in ouput32:
                hasil31[x] = np.round(np.average(ouput32[x]), 2)
            for x in ouput33:
                hasil32[x] = np.round(np.average(ouput33[x]), 2)
            for x in ouput34:
                hasil33[x] = np.round(np.average(ouput34[x]), 2)
            for x in ouput35:
                hasil34[x] = np.round(np.average(ouput35[x]), 2)
            for x in ouput36:
                hasil35[x] = np.round(np.average(ouput36[x]), 2)

            pengeluaran19 = [
                ['Etika', hasil18['Etika']],
                ['Bahasa Inggris', hasil18['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil18['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil18['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil18['Komunikasi']],
                ['Kerjasama Tim', hasil18['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil18['Pengembangan_Diri']]
            ]
            pengeluaran20 = [
                ['Etika', hasil19['Etika']],
                ['Bahasa Inggris', hasil19['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil19['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil19['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil19['Komunikasi']],
                ['Kerjasama Tim', hasil19['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil19['Pengembangan_Diri']]
            ]
            pengeluaran21 = [
                ['Etika', hasil20['Etika']],
                ['Bahasa Inggris', hasil20['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil20['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil20['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil20['Komunikasi']],
                ['Kerjasama Tim', hasil20['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil20['Pengembangan_Diri']]
            ]
            pengeluaran22 = [
                ['Etika', hasil21['Etika']],
                ['Bahasa Inggris', hasil21['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil21['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil21['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil21['Komunikasi']],
                ['Kerjasama Tim', hasil21['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil21['Pengembangan_Diri']]
            ]
            pengeluaran23 = [
                ['Etika', hasil22['Etika']],
                ['Bahasa Inggris', hasil22['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil22['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil22['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil22['Komunikasi']],
                ['Kerjasama Tim', hasil22['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil22['Pengembangan_Diri']]
            ]
            pengeluaran24 = [
                ['Etika', hasil23['Etika']],
                ['Bahasa Inggris', hasil23['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil23['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil23['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil23['Komunikasi']],
                ['Kerjasama Tim', hasil23['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil23['Pengembangan_Diri']]
            ]
            pengeluaran25 = [
                ['Etika', hasil24['Etika']],
                ['Bahasa Inggris', hasil24['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil24['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil24['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil24['Komunikasi']],
                ['Kerjasama Tim', hasil24['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil24['Pengembangan_Diri']]
            ]
            pengeluaran26 = [
                ['Etika', hasil25['Etika']],
                ['Bahasa Inggris', hasil25['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil25['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil25['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil25['Komunikasi']],
                ['Kerjasama Tim', hasil25['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil25['Pengembangan_Diri']]
            ]
            pengeluaran27 = [
                ['Etika', hasil26['Etika']],
                ['Bahasa Inggris', hasil26['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil26['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil26['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil26['Komunikasi']],
                ['Kerjasama Tim', hasil26['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil26['Pengembangan_Diri']]
            ]
            pengeluaran28 = [
                ['Etika', hasil27['Etika']],
                ['Bahasa Inggris', hasil27['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil27['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil27['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil27['Komunikasi']],
                ['Kerjasama Tim', hasil27['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil27['Pengembangan_Diri']]
            ]
            pengeluaran29 = [
                ['Etika', hasil28['Etika']],
                ['Bahasa Inggris', hasil28['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil28['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil28['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil28['Komunikasi']],
                ['Kerjasama Tim', hasil28['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil28['Pengembangan_Diri']]
            ]
            pengeluaran30 = [
                ['Etika', hasil29['Etika']],
                ['Bahasa Inggris', hasil29['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil29['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil29['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil29['Komunikasi']],
                ['Kerjasama Tim', hasil29['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil29['Pengembangan_Diri']]
            ]
            pengeluaran31= [
                ['Etika', hasil30['Etika']],
                ['Bahasa Inggris', hasil30['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil30['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil30['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil30['Komunikasi']],
                ['Kerjasama Tim', hasil30['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil30['Pengembangan_Diri']]
            ]
            pengeluaran32=[
                ['Etika', hasil31['Etika']],
                ['Bahasa Inggris', hasil31['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil31['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil31['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil31['Komunikasi']],
                ['Kerjasama Tim', hasil31['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil31['Pengembangan_Diri']]
            ]
            pengeluaran33= [
                ['Etika', hasil32['Etika']],
                ['Bahasa Inggris', hasil32['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil32['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil32['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil32['Komunikasi']],
                ['Kerjasama Tim', hasil32['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil32['Pengembangan_Diri']]
            ]
            pengeluaran34= [
                ['Etika', hasil33['Etika']],
                ['Bahasa Inggris', hasil33['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil33['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil33['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil33['Komunikasi']],
                ['Kerjasama Tim', hasil33['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil33['Pengembangan_Diri']]
            ]
            pengeluaran35= [
                ['Etika', hasil34['Etika']],
                ['Bahasa Inggris', hasil34['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil34['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil34['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil34['Komunikasi']],
                ['Kerjasama Tim', hasil34['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil34['Pengembangan_Diri']]
            ]
            pengeluaran36= [
                ['Etika', hasil35['Etika']],
                ['Bahasa Inggris', hasil35['Bahasa_Inggris']],
                ['Keahlian Bidang Ilmu', hasil35['Keahlian_Bidang_Ilmu']],
                ['Penggunaan Teknologi Informasi', hasil35['Penggunaan_Teknologi_Informasi']],
                ['Komunikasi', hasil35['Komunikasi']],
                ['Kerjasama Tim', hasil35['Kerjasama_Tim']],
                ['Pengembangan Diri', hasil35['Pengembangan_Diri']]
            ]

            maka19 = pd.DataFrame(pengeluaran19, columns=['Kemampuan', 'Nilai'])
            maka20 = pd.DataFrame(pengeluaran20, columns=['Kemampuan', 'Nilai'])
            maka21 = pd.DataFrame(pengeluaran21, columns=['Kemampuan', 'Nilai'])
            maka22 = pd.DataFrame(pengeluaran22, columns=['Kemampuan', 'Nilai'])
            maka23 = pd.DataFrame(pengeluaran23, columns=['Kemampuan', 'Nilai'])
            maka24 = pd.DataFrame(pengeluaran24, columns=['Kemampuan', 'Nilai'])
            maka25 = pd.DataFrame(pengeluaran25, columns=['Kemampuan', 'Nilai'])
            maka26 = pd.DataFrame(pengeluaran26, columns=['Kemampuan', 'Nilai'])
            maka27 = pd.DataFrame(pengeluaran27, columns=['Kemampuan', 'Nilai'])
            maka28 = pd.DataFrame(pengeluaran28, columns=['Kemampuan', 'Nilai'])
            maka29 = pd.DataFrame(pengeluaran29, columns=['Kemampuan', 'Nilai'])
            maka30 = pd.DataFrame(pengeluaran30, columns=['Kemampuan', 'Nilai'])
            maka31 = pd.DataFrame(pengeluaran31, columns=['Kemampuan', 'Nilai'])
            maka32 = pd.DataFrame(pengeluaran32, columns=['Kemampuan', 'Nilai'])
            maka33 = pd.DataFrame(pengeluaran33, columns=['Kemampuan', 'Nilai'])
            maka34 = pd.DataFrame(pengeluaran34, columns=['Kemampuan', 'Nilai'])
            maka35 = pd.DataFrame(pengeluaran35, columns=['Kemampuan', 'Nilai'])
            maka36 = pd.DataFrame(pengeluaran36, columns=['Kemampuan', 'Nilai'])


            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=maka28['Nilai'],
                theta=maka28['Kemampuan'],
                fill='toself',
                name='Sastra Inggris'
            ))

            fig.add_trace(go.Scatterpolar(
                r=maka29['Nilai'],
                theta=maka29['Kemampuan'],
                fill='toself',
                name='Sastra Jepang'
            ))

            fig.update_layout(
                title='Fakultas Ilmu Budaya',
                polar=dict(
                    radialaxis=dict(
                        visible=True
                    )
                ),
                showlegend=True
            )


            fig2 = go.Figure(
                    data=[
                        go.Scatterpolar(r=maka34['Nilai'], theta=maka34['Kemampuan'], fill='toself', name='HUKUM')],
                    layout=go.Layout(
                        title=go.layout.Title(text='Fakultas Hukum'),
                        polar=dict(
                        radialaxis=dict(
                            visible=True
                        )
                    ),
                        showlegend=True
                    )
                )

            fig3 = go.Figure()
            fig3.add_trace(go.Scatterpolar(
                r=maka19['Nilai'],
                theta=maka19['Kemampuan'],
                fill='toself',
                name='Teknik Informatika'
            ))

            fig3.add_trace(go.Scatterpolar(
                r=maka20['Nilai'],
                theta=maka20['Kemampuan'],
                fill='toself',
                name='Sistem Informasi'
            ))

            fig3.add_trace(go.Scatterpolar(
                r=maka36['Nilai'],
                theta=maka36['Kemampuan'],
                fill='toself',
                name='Teknik Industri'
            ))

            fig3.add_trace(go.Scatterpolar(
                r=maka34['Nilai'],
                theta=maka34['Kemampuan'],
                fill='toself',
                name='Teknik Sipil'
            ))

            fig3.add_trace(go.Scatterpolar(
                r=maka26['Nilai'],
                theta=maka26['Kemampuan'],
                fill='toself',
                name='Teknik Elektro'
            ))

            fig3.add_trace(go.Scatterpolar(
                r=maka27['Nilai'],
                theta=maka27['Kemampuan'],
                fill='toself',
                name='Perencaan Wilayah Kota'
            ))
            fig3.add_trace(go.Scatterpolar(
                r=maka30['Nilai'],
                theta=maka30['Kemampuan'],
                fill='toself',
                name='Teknik Arsitektur'
            ))
            fig3.add_trace(go.Scatterpolar(
                r=maka25['Nilai'],
                theta=maka25['Kemampuan'],
                fill='toself',
                name='Sistem Komputer'
            ))
            fig3.update_layout(
                title='Fakultas Teknik dan Ilmu Komputer',
                polar=dict(
                    radialaxis=dict(
                        visible=True
                    )
                ),
                showlegend=True
            )
            fig4 = go.Figure()
            fig4.add_trace(go.Scatterpolar(
                r=maka21['Nilai'],
                theta=maka21['Kemampuan'],
                fill='toself',
                name='Manajemen'
            ))

            fig4.add_trace(go.Scatterpolar(
                r=maka22['Nilai'],
                theta=maka22['Kemampuan'],
                fill='toself',
                name='Akuntansi'
            ))

            fig4.update_layout(
                title='Fakultas Ekonomi Dan Bisnis',
                polar=dict(
                    radialaxis=dict(
                        visible=True
                    )
                ),
                showlegend=True
            )

            fig5 = go.Figure()

            fig5.add_trace(go.Scatterpolar(
                r=maka23['Nilai'],
                theta=maka23['Kemampuan'],
                fill='toself',
                name='Desain Komunikasi Visual'
            ))

            fig5.add_trace(go.Scatterpolar(
                r=maka32['Nilai'],
                theta=maka32['Kemampuan'],
                fill='toself',
                name='Desain Interior'
            ))




            fig5.update_layout(
                title='Fakultas Desain',
                polar=dict(
                    radialaxis=dict(
                        visible=True
                    )
                ),
                showlegend=True
            )

            fig6 = go.Figure()

            fig6.add_trace(go.Scatterpolar(
                r=maka31['Nilai'],
                theta=maka31['Kemampuan'],
                fill='toself',
                name='Ilmu Pemerintahan'
            ))
            fig6.add_trace(go.Scatterpolar(
                r=maka24['Nilai'],
                theta=maka24['Kemampuan'],
                fill='toself',
                name='Ilmu Komunikasi'
            ))
            fig6.add_trace(go.Scatterpolar(
                r=maka33['Nilai'],
                theta=maka33['Kemampuan'],
                fill='toself',
                name='Hubungan Internasional'
            ))

            fig6.update_layout(
                title='Fakultas Ilmu Sosial Dan Ilmu Politik',
                polar=dict(
                    radialaxis=dict(
                        visible=True
                    )
                ),
                showlegend=True
            )

            st.plotly_chart(fig3)
            st.plotly_chart(fig4)
            st.plotly_chart(fig2)
            st.plotly_chart(fig6)
            st.plotly_chart(fig5)
            st.plotly_chart(fig)

        except(KeyError):
            st.write('')


# Fungsi export ke excel
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data



def sidebar_menu():
    state = st.session_state.setdefault('state', {})
    with st.sidebar:
        selected = option_menu("Menu",["Import Data", "Preprocessing Data", "Clustering Data"],default_index=0)
            
    if (selected == 'Import Data'):
        menu_data()

    if (selected == 'Preprocessing Data'):
        menu_preprocessing()

    if (selected == 'Clustering Data'):
        menu_clustering()

    #menu = ["Home", "Import Data", "Preprocessing Data", "Davies-Bouldin Index", "Clustering Data"]
    #choice = st.sidebar.selectbox("Pilih Opsi Menu", menu)
    
def menu_data():
    judul_halaman('Import Data', '')
    import_data()
    try:
        if not state['dataset'].empty:
            st.dataframe(state['dataset'])
    except(KeyError):
        st.write('')

    
def menu_preprocessing():
    judul_halaman('Preprocessing', 'Missing Value')
    try:
        if not state['dataset'].empty:
            show_null_count()
            if st.button('Bersihkan Data'):
                preprocessing_data()
        else:
            st.warning('Data Belum di Upload')
    except(KeyError):
        st.write('')


def menu_clustering():
    judul_halaman('Clustering Data', ' ')
    DBI()
    # st.write('Tentukan Jumlah Cluster')
    
    input_c = st.number_input('Tentukan Jumlah Kelompok',value=0)
    try:
        if st.button('Mulai Clustering'):
            clustering_data(input_c)
        if not state['dataset'].empty:
            show_cluster(input_c)
            df_xlsx = to_excel(state['clustering'])
            st.download_button(label='Download Hasil Clustering',
                                            data=df_xlsx ,
                                            file_name= 'Hasil Pengelompokkan.xlsx')
        else:
            st.warning('Data Belum di Upload')

        
    except(KeyError):
        st.write('')
        

sidebar_menu()