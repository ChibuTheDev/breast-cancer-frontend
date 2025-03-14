import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Breast Cancer Prediction System", layout="centered")

st.title("Breast Cancer Prediction System")

with st.expander("About the system"):
    st.write(
        """

         This is a system built by students of Babcock university which uses machine learning
         to predict the occurence of breast cancer in patients.
        
         It does this by analyzing tumor features from values gotten from fine needle aspiration. Each feature is to be computed for three different cases:
         Mean (average over the cell nuclei)
         Standard Error (SE) (variation across nuclei)
         Worst (worst-case or most extreme value observed)
        """
    )

    # Input fields for all features
radius_mean = st.number_input("Radius Mean", min_value=0.0, value=10.0)
texture_mean = st.number_input("Texture Mean", min_value=0.0, value=20.0)
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=100.0)
area_mean = st.number_input("Area Mean", min_value=0.0, value=500.0)
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, value=0.1)
compactness_mean = st.number_input("Compactness Mean", min_value=0.0, value=0.2)
concavity_mean = st.number_input("Concavity Mean", min_value=0.0, value=0.3)
concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, value=0.1)
symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, value=0.2)
fractal_dimension_mean = st.number_input(
    "Fractal Dimension Mean", min_value=0.0, value=0.05
)

radius_se = st.number_input("Radius SE", min_value=0.0, value=0.5)
texture_se = st.number_input("Texture SE", min_value=0.0, value=1.0)
perimeter_se = st.number_input("Perimeter SE", min_value=0.0, value=5.0)
area_se = st.number_input("Area SE", min_value=0.0, value=50.0)
smoothness_se = st.number_input("Smoothness SE", min_value=0.0, value=0.01)
compactness_se = st.number_input("Compactness SE", min_value=0.0, value=0.02)
concavity_se = st.number_input("Concavity SE", min_value=0.0, value=0.03)
concave_points_se = st.number_input("Concave Points SE", min_value=0.0, value=0.01)
symmetry_se = st.number_input("Symmetry SE", min_value=0.0, value=0.02)
fractal_dimension_se = st.number_input(
    "Fractal Dimension SE", min_value=0.0, value=0.005
)

radius_worst = st.number_input("Radius Worst", min_value=0.0, value=15.0)
texture_worst = st.number_input("Texture Worst", min_value=0.0, value=25.0)
perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, value=120.0)
area_worst = st.number_input("Area Worst", min_value=0.0, value=800.0)
smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, value=0.15)
compactness_worst = st.number_input("Compactness Worst", min_value=0.0, value=0.3)
concavity_worst = st.number_input("Concavity Worst", min_value=0.0, value=0.4)
concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, value=0.2)
symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, value=0.3)
fractal_dimension_worst = st.number_input(
    "Fractal Dimension Worst", min_value=0.0, value=0.08
)

# Collect inputs into a dataframe
input_data = pd.DataFrame(
    [
        [
            radius_mean,
            texture_mean,
            perimeter_mean,
            area_mean,
            smoothness_mean,
            compactness_mean,
            concavity_mean,
            concave_points_mean,
            symmetry_mean,
            fractal_dimension_mean,
            radius_se,
            texture_se,
            perimeter_se,
            area_se,
            smoothness_se,
            compactness_se,
            concavity_se,
            concave_points_se,
            symmetry_se,
            fractal_dimension_se,
            radius_worst,
            texture_worst,
            perimeter_worst,
            area_worst,
            smoothness_worst,
            compactness_worst,
            concavity_worst,
            concave_points_worst,
            symmetry_worst,
            fractal_dimension_worst,
        ]
    ],
    columns=[
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ],
)

# Load the trained model
model = joblib.load("Ada_boost_breast_cancer_model.pkl")

if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High risk of malignancy! ⚠️")
        
    else:
        st.success("Low risk of malignancy. ✅")
