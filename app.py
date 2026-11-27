
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="InsuraAI | Smart Insurance Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    bundle = joblib.load("model.pkl")
    return (
        bundle["model"],
        bundle["scaler"],
        bundle["columns"]
    )

model, scaler, columns = load_model()

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 10%, #24144c 0%, transparent 35%),
        radial-gradient(ellipse at 90% 80%, #102d4b 0%, transparent 35%),
        #080b14;
    color: #ffffff;
}

/* Main container */

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hide Streamlit header */

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

/* Hero section */

.hero {
    padding: 35px;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    background: linear-gradient(
        120deg,
        rgba(112, 65, 220, 0.20),
        rgba(35, 105, 190, 0.10)
    );
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}

.hero-tag {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 30px;
    color: #c5adff;
    background: rgba(147, 103, 255, 0.15);
    border: 1px solid rgba(147, 103, 255, 0.3);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.hero h1 {
    font-size: clamp(32px, 5vw, 48px);
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 10px;
    background: linear-gradient(90deg, #ffffff, #b69aff, #70c9ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #a7afc4;
    font-size: 15px;
    line-height: 1.8;
}

/* Cards */

.glass-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 18px;
    backdrop-filter: blur(15px);
    transition: all 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(160,120,255,0.4);
    transform: translateY(-3px);
}

.card-title {
    font-size: 19px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 5px;
}

.card-description {
    color: #8993aa;
    font-size: 13px;
    margin-bottom: 18px;
}

/* Input labels */

label {
    color: #dce1ef !important;
    font-weight: 600 !important;
}

/* Input fields */

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.055) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
}

div[data-baseweb="input"] input {
    color: white !important;
}

div[data-baseweb="select"] * {
    color: white !important;
}

div[data-baseweb="popover"] {
    background: #171b2b !important;
}

/* Buttons */

.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(100deg, #7952ee, #347fe8);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 14px 20px;
    font-size: 15px;
    font-weight: 700;
    transition: 0.3s ease;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    box-shadow: 0 0 25px rgba(121,82,238,0.45);
    transform: translateY(-2px);
}

/* Result card */

.result-card {
    padding: 32px;
    border-radius: 22px;
    border: 1px solid rgba(125,100,255,0.35);
    background: linear-gradient(
        135deg,
        rgba(100,65,200,0.22),
        rgba(25,95,160,0.15)
    );
    text-align: center;
    margin-top: 25px;
    animation: appear 0.7s ease;
}

.result-label {
    color: #b7c0d5;
    font-size: 14px;
    font-weight: 600;
}

.result-value {
    font-size: clamp(32px, 6vw, 48px);
    font-weight: 800;
    color: #bba4ff;
    margin: 12px 0;
}

.result-caption {
    color: #929cb2;
    font-size: 13px;
}

@keyframes appear {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Divider */

hr {
    border-color: rgba(255,255,255,0.09) !important;
}

/* Footer */

.footer {
    text-align: center;
    color: #737f96;
    font-size: 12px;
    padding: 25px;
}

.footer span {
    color: #ad91ff;
}

/* Mobile */

@media(max-width: 768px) {
    .hero {
        padding: 23px;
    }

    .glass-card {
        padding: 18px;
    }
}

</style>
""", unsafe_allow_html=True)


# ---------------- HERO ----------------

st.markdown("""
<div class="hero">

    <span class="hero-tag">
        ✦ AI POWERED PREDICTION SYSTEM
    </span>

    <h1>Insurance Cost Predictor</h1>

    <p>
        Discover your estimated insurance expenses with
        intelligent machine learning. Enter your details
        and let our AI model generate your prediction.
    </p>

</div>
""", unsafe_allow_html=True)


# ---------------- TOP INFO CARDS ----------------

info1, info2, info3 = st.columns(3)

with info1:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size:25px">🤖</div>
        <div class="card-title">AI Powered</div>
        <div class="card-description">
            Machine learning based estimation
        </div>
    </div>
    """, unsafe_allow_html=True)

with info2:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size:25px">⚡</div>
        <div class="card-title">Instant Results</div>
        <div class="card-description">
            Generate predictions in seconds
        </div>
    </div>
    """, unsafe_allow_html=True)

with info3:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size:25px">🔒</div>
        <div class="card-title">Simple & Secure</div>
        <div class="card-description">
            No account or registration required
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- FORM ----------------

st.markdown("""
<div class="glass-card">

    <div class="card-title">
        👤 Customer Information
    </div>

    <div class="card-description">
        Provide customer details to calculate the estimated
        insurance charges.
    </div>

</div>
""", unsafe_allow_html=True)


with st.form("insurance_form"):

    left, right = st.columns(2, gap="large")

    with left:

        st.markdown("### Personal Details")

        age = st.slider(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35
        )

        sex = st.selectbox(
            "Gender",
            ["male", "female"]
        )

        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=10.0,
            max_value=60.0,
            value=27.5,
            step=0.1
        )

    with right:

        st.markdown("### Lifestyle Details")

        children = st.select_slider(
            "Number of Children",
            options=list(range(11)),
            value=2
        )

        smoker = st.selectbox(
            "Smoking Status",
            ["no", "yes"]
        )

        region = st.selectbox(
            "Residential Region",
            [
                "southeast",
                "southwest",
                "northeast",
                "northwest"
            ]
        )

    st.markdown("---")

    submit = st.form_submit_button(
        "✦ Generate AI Prediction",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------

if submit:

    with st.spinner("Analyzing customer information..."):

        new_customer = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region]
        })

        customer_encoded = pd.get_dummies(
            new_customer,
            columns=["sex", "smoker", "region"],
            dtype=int
        )

        customer_encoded = customer_encoded.reindex(
            columns=columns,
            fill_value=0
        )

        scale_columns = ["age", "bmi", "children"]

        customer_encoded[scale_columns] = scaler.transform(
            customer_encoded[scale_columns]
        )

        prediction = model.predict(customer_encoded)[0]

    # ---------------- RESULT ----------------

    st.markdown("---")

    st.markdown("""
    <div class="result-card">

        <div class="result-label">
            ✦ YOUR AI GENERATED ESTIMATE
        </div>

        <div style="font-size:32px; margin-top:12px;">
            🏥
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-card" style="margin-top:0">

            <div class="result-label">
                ESTIMATED INSURANCE CHARGES
            </div>

            <div class="result-value">
                ${prediction:,.2f}
            </div>

            <div class="result-caption">
                Prediction generated successfully
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📋 Customer Summary")

    r1, r2, r3 = st.columns(3)

    r1.metric("Age", f"{age} Years")
    r2.metric("BMI", f"{bmi}")
    r3.metric("Children", f"{children}")

    r4, r5, r6 = st.columns(3)

    r4.metric("Gender", sex.capitalize())
    r5.metric("Smoker", smoker.capitalize())
    r6.metric("Region", region.capitalize())

    st.info(
        "This prediction is an estimated value generated "
        "by a machine learning model. It is not an actual "
        "insurance quotation."
    )


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">

    Designed with <span>♥</span> using Python,
    Streamlit, Pandas and Machine Learning.

    <br><br>

    INSURAAI © 2026 | SMART INSURANCE PREDICTION

</div>
""", unsafe_allow_html=True)
