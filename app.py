import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import tempfile
import os
from recommendation import cnv, dme, drusen, normal

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCT Eye Disease Predictor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #94a3b8 !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        margin: 0 0 0.5rem 0;
        color: white;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
    }

    /* Cards */
    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3b82f6;
    }
    .info-card h4 {
        color: #1e293b;
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0 0 0.4rem 0;
    }
    .info-card p {
        color: #64748b;
        font-size: 0.88rem;
        margin: 0;
        line-height: 1.6;
    }

    /* Prediction result box */
    .result-box {
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-box.cnv    { background: #fef2f2; border: 2px solid #fca5a5; }
    .result-box.dme    { background: #fff7ed; border: 2px solid #fdba74; }
    .result-box.drusen { background: #fefce8; border: 2px solid #fde047; }
    .result-box.normal { background: #f0fdf4; border: 2px solid #86efac; }

    .result-box .disease-name {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        margin: 0.5rem 0;
    }
    .result-box.cnv    .disease-name { color: #dc2626; }
    .result-box.dme    .disease-name { color: #ea580c; }
    .result-box.drusen .disease-name { color: #ca8a04; }
    .result-box.normal .disease-name { color: #16a34a; }

    .result-box .confidence-text {
        font-size: 1rem;
        color: #475569;
        margin: 0;
    }

    /* Confidence bar */
    .conf-bar-container {
        margin: 0.6rem 0;
    }
    .conf-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        color: #475569;
        margin-bottom: 4px;
    }
    .conf-bar-bg {
        background: #e2e8f0;
        border-radius: 99px;
        height: 10px;
        width: 100%;
    }
    .conf-bar-fill {
        height: 10px;
        border-radius: 99px;
        transition: width 0.6s ease;
    }

    /* Stat cards on home */
    .stat-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .stat-card .stat-number {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #1e293b;
    }
    .stat-card .stat-label {
        color: #64748b;
        font-size: 0.85rem;
        margin: 0;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 2rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
    }

    /* Divider */
    hr { border-color: #e2e8f0 !important; }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL (cached so it only loads once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "Best_Model.keras"
    if not os.path.exists(model_path):
        model_path = "Trained_Model.keras"
    return tf.keras.models.load_model(model_path)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

def model_prediction(image_path, model):
    img       = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    x         = tf.keras.utils.img_to_array(img)
    x         = np.expand_dims(x, axis=0)
    x         = preprocess_input(x)
    preds     = model.predict(x, verbose=0)
    idx       = int(np.argmax(preds[0]))
    all_probs = {cls: float(p) for cls, p in zip(CLASS_NAMES, preds[0])}
    return idx, all_probs

# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE BARS HELPER
# ─────────────────────────────────────────────────────────────────────────────
BAR_COLORS = {
    'CNV':    '#ef4444',
    'DME':    '#f97316',
    'DRUSEN': '#eab308',
    'NORMAL': '#22c55e',
}

def render_confidence_bars(all_probs, predicted_class):
    st.markdown("**Confidence scores**")
    for cls, prob in all_probs.items():
        pct   = prob * 100
        color = BAR_COLORS[cls] if cls == predicted_class else '#94a3b8'
        bold  = "font-weight:600;" if cls == predicted_class else ""
        st.markdown(f"""
        <div class="conf-bar-container">
            <div class="conf-label">
                <span style="{bold}">{cls}</span>
                <span style="{bold}">{pct:.1f}%</span>
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill"
                     style="width:{pct}%; background:{color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👁️ OCT Analyser")
    st.markdown("---")
    app_mode = st.selectbox(
        "Navigate",
        ["🏠 Home", "🔬 Disease Identification", "📋 About"]
    )
    st.markdown("---")
    st.markdown("""
    <small style='color:#64748b'>
    <b>Model:</b> MobileNetV3Large<br>
    <b>Classes:</b> CNV · DME · DRUSEN · NORMAL<br>
    <b>Dataset:</b> 84,495 OCT images
    </small>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if app_mode == "🏠 Home":

    st.markdown("""
    <div class="main-header">
        <h1>OCT Retinal Analysis Platform</h1>
        <p>AI-powered detection of retinal diseases from Optical Coherence Tomography scans</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="stat-card"><div class="stat-number">84,495</div><p class="stat-label">OCT Images</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stat-card"><div class="stat-number">4</div><p class="stat-label">Disease Classes</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stat-card"><div class="stat-number">224×224</div><p class="stat-label">Input Resolution</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="stat-card"><div class="stat-number">15</div><p class="stat-label">Training Epochs</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Disease cards
    st.markdown("### Detectable Conditions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card" style="border-left-color:#ef4444">
            <h4>🔴 CNV — Choroidal Neovascularization</h4>
            <p>Abnormal blood vessel growth beneath the retina. Associated with wet AMD.
            Characterized by subretinal fluid and neovascular membranes on OCT.</p>
        </div>
        <div class="info-card" style="border-left-color:#f97316">
            <h4>🟠 DME — Diabetic Macular Edema</h4>
            <p>Fluid accumulation in the macula due to diabetes-related vascular leakage.
            Visible as retinal thickening and intraretinal fluid on OCT.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card" style="border-left-color:#eab308">
            <h4>🟡 DRUSEN — Early AMD</h4>
            <p>Yellowish deposits beneath the retinal pigment epithelium. An early marker
            of age-related macular degeneration. Appear as sub-RPE bumps on OCT.</p>
        </div>
        <div class="info-card" style="border-left-color:#22c55e">
            <h4>🟢 NORMAL — Healthy Retina</h4>
            <p>Preserved foveal contour with no signs of fluid, edema, or abnormal
            deposits. Smooth retinal layers visible on OCT.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 Select **Disease Identification** from the sidebar to upload and analyse an OCT scan.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DISEASE IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────
elif app_mode == "🔬 Disease Identification":

    st.markdown("""
    <div class="main-header">
        <h1>Disease Identification</h1>
        <p>Upload an OCT retinal scan to get an instant AI-powered diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### 1. Upload OCT Image")
        test_image = st.file_uploader(
            "Supported formats: JPG, JPEG, PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if test_image is not None:
            st.image(test_image, caption="Uploaded OCT Scan", use_column_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🔍 Analyse Image")
        else:
            st.markdown("""
            <div style='text-align:center; color:#94a3b8; padding:2rem 0'>
                <div style='font-size:3rem'>🩺</div>
                <p>Upload an OCT image to begin analysis</p>
            </div>
            """, unsafe_allow_html=True)
            predict_btn = False

    with col_result:
        st.markdown("#### 2. Prediction Result")

        if test_image is not None and predict_btn:
            with st.spinner("Analysing OCT scan..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(test_image.name)[1]
                ) as tmp:
                    tmp.write(test_image.read())
                    tmp_path = tmp.name

                # Load model and predict
                model               = load_model()
                result_index, probs = model_prediction(tmp_path, model)
                predicted_class     = CLASS_NAMES[result_index]
                confidence          = probs[predicted_class]

                # Clean up temp file
                os.unlink(tmp_path)

            # Result box
            css_class = predicted_class.lower()
            icons = {'CNV': '🔴', 'DME': '🟠', 'DRUSEN': '🟡', 'NORMAL': '🟢'}
            st.markdown(f"""
            <div class="result-box {css_class}">
                <div style='font-size:2.5rem'>{icons[predicted_class]}</div>
                <div class="disease-name">{predicted_class}</div>
                <p class="confidence-text">Confidence: <b>{confidence*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bars
            render_confidence_bars(probs, predicted_class)

        elif test_image is None:
            st.markdown("""
            <div style='text-align:center; color:#94a3b8; padding:3rem 0;
                        border:2px dashed #e2e8f0; border-radius:12px;'>
                <div style='font-size:2.5rem'>📊</div>
                <p>Results will appear here after analysis</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Recommendation section (shown after prediction) ──────────────────────
    if test_image is not None and predict_btn:
        st.markdown("---")
        st.markdown("### 📋 Medical Recommendation")

        rec_map = {
            'CNV':    cnv,
            'DME':    dme,
            'DRUSEN': drusen,
            'NORMAL': normal,
        }
        desc_map = {
            'CNV':    "OCT scan showing **CNV** with neovascular membrane and subretinal fluid.",
            'DME':    "OCT scan showing **DME** with retinal thickening and intraretinal fluid.",
            'DRUSEN': "OCT scan showing **drusen deposits** indicative of early AMD.",
            'NORMAL': "OCT scan showing a **normal retina** with preserved foveal contour.",
        }

        st.info(desc_map[predicted_class])
        st.markdown(rec_map[predicted_class])

        st.markdown("---")
        st.warning(
            "⚠️ **Disclaimer:** This tool is for educational and research purposes only. "
            "It is not a substitute for professional medical diagnosis. "
            "Please consult a qualified ophthalmologist for clinical decisions."
        )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif app_mode == "📋 About":

    st.markdown("""
    <div class="main-header">
        <h1>About This Project</h1>
        <p>Dataset details, methodology, and technical information</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Dataset")
    st.markdown("""
    Retinal optical coherence tomography (OCT) is an imaging technique used to capture
    high-resolution cross-sections of the retinas of living patients. Approximately
    **30 million OCT scans** are performed each year worldwide.

    The dataset contains **84,495 JPEG images** organized into 3 splits:
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Training images", "76,515")
    with c2:
        st.metric("Validation images", "8,000")
    with c3:
        st.metric("Test images", "968")

    st.markdown("""
    ---
    ### Image Categories

    | Class | Description |
    |-------|-------------|
    | **CNV** | Choroidal neovascularization — neovascular membrane with subretinal fluid |
    | **DME** | Diabetic macular edema — retinal thickening with intraretinal fluid |
    | **DRUSEN** | Multiple drusen deposits — early sign of AMD |
    | **NORMAL** | Preserved foveal contour, absence of fluid or edema |

    ---
    ### Source & Verification

    OCT images were collected from multiple renowned centres including:
    - Shiley Eye Institute, University of California San Diego
    - California Retinal Research Foundation
    - Shanghai First People's Hospital
    - Beijing Tongren Eye Center

    Images were graded through a **three-tier verification system**:
    1. Undergraduate and medical students — initial quality control
    2. Four independent ophthalmologists — disease labelling
    3. Two senior retinal specialists (20+ years experience) — final verification

    ---
    ### Model Architecture

    - **Base Model:** MobileNetV3Large pretrained on ImageNet
    - **Custom Head:** BatchNorm → Dropout(0.3) → Dense(256) → Dropout(0.2) → Dense(4, softmax)
    - **Optimizer:** Adam (lr = 0.0001)
    - **Loss:** Categorical Crossentropy
    - **Metrics:** Accuracy, F1 Score (macro)
    """)
