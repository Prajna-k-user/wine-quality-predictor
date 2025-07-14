import streamlit as st
import pandas as pd
import pickle

st.title("🍷 Wine Quality Predictor")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Input fields
st.markdown("### Enter wine chemical properties below:")

features = ['fixed acidity','volatile acidity','citric acid','residual sugar',
            'chlorides','free sulfur dioxide','total sulfur dioxide','density','pH',
            'sulphates','alcohol']

input_data = {f: st.number_input(f, value=0.0) for f in features}

if st.button("Predict Quality"):
    data = pd.DataFrame([input_data])
    prediction = model.predict(data)[0]
    st.success("✅ Good Quality Wine 🍾" if prediction == 1 else "❌ Not Good Quality Wine")
