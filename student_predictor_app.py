
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(page_title="Student Success Predictor", layout="wide")
st.title("INF Student Sucvvvnnvvvvvcess Predictor")

@st.cache_data
def load_and_train():
    # Load data
    df = pd.read_excel('INF_CALCULATOR.xlsx', sheet_name='INF ADMISSION CALCULATOR')
    df.columns = df.columns.str.strip()
    
    # Prepare data
    le = LabelEncoder()
    df['SERIE'] = le.fit_transform(df['SERIE'])
    
    features = ['B_AVERAGE', 'B_MATH', 'B_PHYSIC', 'B_FRANCAIS', 'SERIE']
    targets = ['1_G-AVERAGE', 'ALG1', 'ANA1', 'ALGO1', 'STR1', 
               'ALG2', 'ANA2', 'ALGO2', 'STR2', 'PROBA2']
    
    # Train models
    models = {}
    for target in targets:
        X = df[features]
        y = (df[target] >= 10).astype(int)
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        models[target] = model
    
    return models, le, le.classes_

def predict_success(input_data, models):
    results = {}
    for course, model in models.items():
        prob = model.predict_proba(np.array(input_data).reshape(1, -1))[0, 1]
        results[course] = prob * 100
    return results

# Load models
models, le, majors = load_and_train()

# Create input form
st.sidebar.header("Student Information")

b_average = st.sidebar.slider("BAC General Average", 0.0, 20.0, 12.0, 0.1)
b_math = st.sidebar.slider("Mathematics Score", 0.0, 20.0, 12.0, 0.1)
b_physic = st.sidebar.slider("Physics Score", 0.0, 20.0, 12.0, 0.1)
b_francais = st.sidebar.slider("French Score", 0.0, 20.0, 12.0, 0.1)
serie = st.sidebar.selectbox("High School Major", majors)

if st.sidebar.button("Predict Success"):
    # Prepare input data
    serie_encoded = le.transform([serie])[0]
    input_data = [b_average, b_math, b_physic, b_francais, serie_encoded]
    
    # Get predictions
    results = predict_success(input_data, models)
    
    # Display results
    st.header("Prediction Results")
    
    # Create three columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("First Year - Semester 1")
        for course in ['1_G-AVERAGE', 'ALG1', 'ANA1', 'ALGO1', 'STR1']:
            prob = results[course]
            color = 'green' if prob >= 70 else 'orange' if prob >= 50 else 'red'
            st.markdown(f"{course}: <span style='color:{color}'>{prob:.1f}%</span>", unsafe_allow_html=True)
    
    with col2:
        st.subheader("First Year - Semester 2")
        for course in ['ALG2', 'ANA2', 'ALGO2', 'STR2', 'PROBA2']:
            prob = results[course]
            color = 'green' if prob >= 70 else 'orange' if prob >= 50 else 'red'
            st.markdown(f"{course}: <span style='color:{color}'>{prob:.1f}%</span>", unsafe_allow_html=True)
    
    with col3:
        st.subheader("Summary")
        avg_prob = sum(results.values()) / len(results)
        st.markdown(f"Overall Success Probability: **{avg_prob:.1f}%**")
        
        if avg_prob >= 70:
            st.success("High chance of success!")
        elif avg_prob >= 50:
            st.warning("Moderate chance of success")
        else:
            st.error("May need additional preparation")
