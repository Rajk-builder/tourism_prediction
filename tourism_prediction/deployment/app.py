import os
import pandas as pd
import joblib
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_model.joblib")
st.set_page_config(page_title="Tourism Package Predictor", page_icon="🧳")
st.title("🧳 Wellness Tourism Package Predictor")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


model = load_model()
if model is None:
    st.error("Model file not found. Run the GitHub Actions pipeline first.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    Age = st.number_input("Age", 18, 100, 35)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.number_input("Duration of Pitch", 0.0, 200.0, 15.0)
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Persons Visiting", 1, 10, 3)
    NumberOfFollowups = st.number_input("Followups", 0.0, 10.0, 3.0)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
with c2:
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    NumberOfTrips = st.number_input("Trips per year", 0.0, 50.0, 2.0)
    Passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "Yes" if x else "No")
    PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    OwnCar = st.selectbox("Owns Car", [0, 1], format_func=lambda x: "Yes" if x else "No")
    NumberOfChildrenVisiting = st.number_input("Children Visiting", 0.0, 10.0, 0.0)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("Monthly Income", 1000.0, 1000000.0, 20000.0)

input_df = pd.DataFrame([{
    "Age": Age, "TypeofContact": TypeofContact, "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch, "Occupation": Occupation, "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting, "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched, "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus, "NumberOfTrips": NumberOfTrips, "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore, "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting, "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    if pred == 1:
        st.success(f"Likely to PURCHASE  (probability {proba:.1%})")
    else:
        st.info(f"Unlikely to purchase  (probability {proba:.1%})")
