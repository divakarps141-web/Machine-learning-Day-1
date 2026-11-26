
import streamlit as st
import pandas as pd
import joblib

# Load trained model
bundle = joblib.load("model.pkl")

model = bundle["model"]
scaler = bundle["scaler"]
columns = bundle["columns"]

# Page configuration
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)

# Application heading
st.title("🏥 Insurance Cost Predictor")

st.write(
    "Predict insurance charges using "
    "Machine Learning."
)

st.divider()

# Input form
st.subheader("Enter Customer Details")

with st.form("insurance_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        sex = st.selectbox(
            "Gender",
            ["male", "female"]
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=27.5
        )

    with col2:
        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=2
        )

        smoker = st.selectbox(
            "Smoker",
            ["yes", "no"]
        )

        region = st.selectbox(
            "Region",
            [
                "southeast",
                "southwest",
                "northeast",
                "northwest"
            ]
        )

    submit = st.form_submit_button(
        "Predict Insurance Charges",
        use_container_width=True
    )

# Prediction
if submit:

    # Create customer dataframe
    new_customer = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # Encode categorical columns
    customer_encoded = pd.get_dummies(
        new_customer,
        columns=["sex", "smoker", "region"],
        dtype=int
    )

    # Match training columns
    customer_encoded = customer_encoded.reindex(
        columns=columns,
        fill_value=0
    )

    # Standardize numerical columns
    scale_columns = ["age", "bmi", "children"]

    customer_encoded[scale_columns] = scaler.transform(
        customer_encoded[scale_columns]
    )

    # Predict
    prediction = model.predict(customer_encoded)[0]

    st.divider()

    st.success("Prediction Completed!")

    st.subheader("Predicted Insurance Charges")

    st.metric(
        label="Estimated Insurance Charges",
        value=f"{prediction:,.2f}"
    )

    st.info(
        "This is a machine learning estimate, "
        "not an actual insurance quotation."
    )

st.divider()

st.caption(
    "Developed using Python, Pandas, Scikit-learn and Streamlit."
)
