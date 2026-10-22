
import os
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="wide"
)

MODEL_PATH = os.path.join(
    "models", "credit_scoring_model.joblib"
)

# --------------------------------------------------
# LOAD THE TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

# --------------------------------------------------
# APP HEADER
# --------------------------------------------------

st.title("💳 Credit Risk Prediction System")

st.write(
    "An educational machine learning application "
    "that predicts historical credit-risk categories."
)

st.info(
    "This is an academic demonstration using the German "
    "Credit Risk dataset. It is not an official credit score, "
    "a loan approval system, or financial advice."
)

if model is None:
    st.error(
        "Trained model not found. First run train_model.py "
        "and check that the model exists inside the models folder."
    )
    st.stop()

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

st.subheader("Applicant Information")

with st.form("credit_risk_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

        housing = st.selectbox(
            "Housing",
            ["own", "rent", "free"]
        )

        saving_account = st.selectbox(
            "Saving accounts",
            ["little", "moderate", "quite rich", "rich"]
        )

        checking_account = st.selectbox(
            "Checking account",
            ["little", "moderate", "rich"]
        )

    with col2:
        credit_amount = st.number_input(
            "Credit amount (dataset currency)",
            min_value=1,
            max_value=1000000,
            value=2500,
            step=100
        )

        duration = st.number_input(
            "Credit duration (months)",
            min_value=1,
            max_value=120,
            value=24,
            step=1
        )

        purpose = st.selectbox(
            "Purpose of credit",
            [
                "car",
                "furniture/equipment",
                "radio/TV",
                "domestic appliances",
                "repairs",
                "education",
                "business",
                "vacation/others"
            ]
        )

    submitted = st.form_submit_button(
        "Predict Credit Risk",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    monthly_proxy = credit_amount / duration

    applicant = pd.DataFrame([{
        "Age": age,
        "Housing": housing,
        "Saving accounts": saving_account,
        "Checking account": checking_account,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose,
        "Monthly credit amount proxy": monthly_proxy
    }])

    try:
        prediction = int(model.predict(applicant)[0])
        probabilities = model.predict_proba(applicant)[0]

        # The training labels are 0 = good and 1 = bad.
        # Find the probability corresponding to class 1.
        class_list = list(model.classes_)
        bad_index = class_list.index(1)
        bad_probability = float(probabilities[bad_index])

        st.divider()
        st.subheader("Prediction Result")

        left, right = st.columns(2)

        with left:
            if prediction == 1:
                st.error("Predicted category: Bad historical risk")
            else:
                st.success("Predicted category: Good historical risk")

        with right:
            st.metric(
                "Estimated probability of bad-risk class",
                f"{bad_probability:.1%}"
            )

        st.caption(
            "This probability is the model's output, not a "
            "guaranteed probability of default. It has not "
            "been independently calibrated for real-world use."
        )

        st.subheader("Submitted Information")
        st.dataframe(applicant, use_container_width=True)

        st.warning(
            "Do not use this prediction to approve or reject a real "
            "loan. The model is trained on a small historical dataset "
            "and has not been validated for real lending decisions."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

with st.expander("About this project"):
    st.write(
        "Models compared during training: Logistic Regression, "
        "Decision Tree, and Random Forest."
    )
    st.write(
        "The selected model was saved using joblib. "
        "Numerical and categorical preprocessing are included "
        "in the saved pipeline."
    )

    st.write(
        "Historical labels: 0 = good risk, 1 = bad risk."
    )
