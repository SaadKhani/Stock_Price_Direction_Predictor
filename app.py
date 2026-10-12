import pickle
import streamlit as st
import pandas as pd

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="centered"
)

# Frontend
st.title("📈 Stock Price Direction Predictor")
st.write(
    "Enter today's stock price features to predict "
    "whether the next closing price will increase."
)

st.divider()

with st.form("prediction_form"):
    open_close = st.number_input(
        "Open - Close",
        value=0.0,
        format="%.4f"
    )

    high_low = st.number_input(
        "High - Low",
        min_value=0.0,
        value=0.0,
        format="%.4f"
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    # Keep the same feature names and order used during training
    input_data = pd.DataFrame(
        [[open_close, high_low]],
        columns=["Open-Close", "High-Low"]
    )

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "The model predicts that the next closing price "
            "will be higher."
        )
    else:
        st.info(
            "The model predicts that the next closing price "
            "will not be higher."
        )

    st.caption(
        "This is a machine learning prediction, not a guarantee "
        "of future stock performance."
    )
