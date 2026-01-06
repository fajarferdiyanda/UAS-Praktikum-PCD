import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# --- 1. KONFIGURASI HALAMAN MODERN ---
st.set_page_config(
    page_title="UAS Praktikum PCD",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS (GLASSMORPHISM & MODERN UI) ---
st.markdown("""
    <style>
    /* Sembunyikan elemen default yang tidak perlu */
    [data-testid="stSidebar"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Global Background */
    .stApp {
        background: linear-gradient(to bottom right, #f8f9fa, #e9ecef);
        font-family: 'Inter', sans-serif;
    }

    /* Hero Section (Header) */
    .hero-container {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        padding: 40px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
    }
    .hero-container h1 { font-size: 3rem; font-weight: 800; margin: 0; color: white; }
    .hero-container p { font-size: 1.2rem; opacity: 0.9; margin-top: 10px; }

    /* Modern Cards */
    .modern-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.5);
        margin-bottom: 20px;
    }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #4f46e5;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNGSI UTILITAS ---
def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

def convert_to_bytes(img_num, format='PNG'):
    # Konversi array numpy ke bytes untuk tombol download
    is_success, buffer = cv2.imencode(f".{format.lower()}", img_num)
    io_buf = io.BytesIO(buffer)
    return io_buf

# --- 3. HERO SECTION ---
st.markdown("""
    <div class="hero-container">
        <h1>Segmentasi Citra</h1>
    </div>
""", unsafe_allow_html=True)

# --- 4. CONTAINER UPLOAD (CLEAN UI) ---
with st.container():
    col_up1, col_up2 = st.columns([1, 2])
    
    with col_up1:
        st.markdown("### 📤 Upload Gambar")
        st.caption("Format: JPG, PNG, TIF")
        uploaded_file = st.file_uploader("", type=["jpg", "png", "tif", "jpeg"], label_visibility="collapsed")
    
    with col_up2:
        if uploaded_file:
            image_np = load_image(uploaded_file)
            if len(image_np.shape) == 3:
                gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image_np
            
            # Menampilkan info gambar dengan gaya modern
            st.success("✅ Gambar berhasil dimuat!")
            st.markdown(f"**Dimensi:** `{image_np.shape[1]}x{image_np.shape[0]} px` | **Mode:** `Grayscale`")
        else:
            st.info("ℹ️ Silakan upload gambar untuk membuka panel kontrol.")
            st.stop()

st.markdown("---")

# --- 5. PANEL KONTROL & VISUALISASI ---

# Layout: Kiri (Kontrol), Kanan (Visualisasi)
col_control, col_visual = st.columns([1, 2])

with col_control:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Kontrol")
    
    # Dropdown Pilihan Metode
    metode = st.selectbox(
        "Pilih Metode Segmentasi:",
        ["1. Threshold Manual", "2. Threshold Iteratif (Otsu)", "3. Threshold Multilevel"]
    )
    
    # Widget Tambahan: Invert Colors (Menarik!)
    invert_mode = st.toggle("🔄 Balik Warna (Invert)", value=False)
    
    thresh_result = None
    threshold_value_display = 0
    
    # --- LOGIKA PROSES ---
    
    # 1. Manual
    if "Manual" in metode:
        st.markdown("#### Pengaturan Nilai")
        t_val = st.slider("Geser Nilai Ambang", 0, 255, 127)
        threshold_value_display = t_val
        
        type_cv = cv2.THRESH_BINARY_INV if invert_mode else cv2.THRESH_BINARY
        _, thresh_result = cv2.threshold(gray_image, t_val, 255, type_cv)

    # 2. Iteratif (Otsu)
    elif "Iteratif" in metode:
        st.markdown("#### Info Otomatis")
        st.caption("Algoritma menghitung nilai optimal secara otomatis.")
        
        type_cv = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if invert_mode else cv2.THRESH_BINARY + cv2.THRESH_OTSU
        val, thresh_result = cv2.threshold(gray_image, 0, 255, type_cv)
        threshold_value_display = int(val)

    # 3. Multilevel
    elif "Multilevel" in metode:
        st.markdown("#### Rentang Nilai")
        range_val = st.slider("Pilih Batas Bawah & Atas", 0, 255, (50, 200))
        lower, upper = range_val
        threshold_value_display = f"{lower}-{upper}"
        
        # Invert logic manual untuk inRange (sedikit tricky, kita pakai bitwise not jika invert)
        mask = cv2.inRange(gray_image, lower, upper)
        thresh_result = cv2.bitwise_not(mask) if invert_mode else mask

    st.markdown('</div>', unsafe_allow_html=True)

    # --- WIDGET METRIK (Data at a Glance) ---
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.metric(label="Nilai Threshold Aktif", value=threshold_value_display)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- HISTOGRAM (Disembunyikan dalam Expander agar Rapi) ---
    with st.expander("📊 Lihat Histogram Citra"):
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.hist(gray_image.ravel(), 256, [0, 256], color='#6366f1', alpha=0.7)
        ax.set_title("Distribusi Pixel")
        ax.axis('off')
        st.pyplot(fig)


with col_visual:
    # --- TAMPILAN GAMBAR SIDE-BY-SIDE ---
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🖼️ Preview Split", "🔍 Detail Hasil"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.image(gray_image, caption="1. Original (Grayscale)", use_container_width=True)
        with c2:
            st.image(thresh_result, caption=f"2. Hasil: {metode.split('.')[1]}", use_container_width=True)
    
    with tab2:
        st.image(thresh_result, caption="Tampilan Penuh Hasil Segmentasi", use_container_width=True)
        
        # --- TOMBOL DOWNLOAD (Fitur Modern Penting) ---
        btn_col1, btn_col2 = st.columns([3, 1])
        with btn_col2:
            img_bytes = convert_to_bytes(thresh_result)
            st.download_button(
                label="⬇️ Unduh",
                data=img_bytes,
                file_name="hasil_segmentasi.png",
                mime="image/png",
                use_container_width=True
            )
            
    st.markdown('</div>', unsafe_allow_html=True)

# Footer Minimalis
st.markdown("<div style='text-align: center; color: #aaa; margin-top: 50px;'><small>Ditenagai oleh Streamlit & OpenCV</small></div>", unsafe_allow_html=True)