import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Student Exam Score Prediction",
    page_icon="🎓",
    layout="wide"
)

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_data():
    """Loads the dataset and model performance results."""
    try:
        df = pd.read_csv('stud.csv')
        results_df = pd.read_csv('saved_models/model_results.csv')
        with open('saved_models/feature_info.json', 'r') as f:
            feature_info = json.load(f)
    except FileNotFoundError as e:
        st.error(f"Error: Missing required file - {e.filename}. Please ensure 'stud.csv', 'model_results.csv', and 'feature_info.json' are present.")
        return None, None, None
    return df, results_df, feature_info

@st.cache_resource
def load_model():
    """Loads the single, best model pipeline."""
    try:
        pipeline = joblib.load('saved_models/best_model_pipeline.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("Error: 'best_model_pipeline.pkl' not found. Please run the training notebook to create it.")
        return None

# --- LOAD DATA AND MODEL ---
df, results_df, feature_info = load_data()
pipeline = load_model()

if df is None or pipeline is None:
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🎓 Model Information")
st.sidebar.markdown("This app uses the best-performing model to predict scores.")

# Display best model's info from the results file
if not results_df.empty:
    best_model_info = results_df.iloc[0]
    st.sidebar.subheader(f"🏆 Best Model: {best_model_info['model']}")
    st.sidebar.metric("R² Score", f"{best_model_info['test_r2']:.4f}")
    st.sidebar.metric("RMSE", f"{best_model_info['test_rmse']:.2f}")
    st.sidebar.metric("MAE", f"{best_model_info['test_mae']:.2f}")
else:
    st.sidebar.warning("Could not load model performance data.")


# --- MAIN CONTENT ---
st.title("Student Exam Score Prediction")
st.markdown("Select student attributes to predict their **Math Score**.")

# Feature input form
with st.container():
    st.subheader("Student Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('Gender', options=feature_info['categorical_values']['gender'])
        race_ethnicity = st.selectbox('Race/Ethnicity', options=feature_info['categorical_values']['race_ethnicity'])

    with col2:
        parental_education = st.selectbox(
            'Parental Level of Education',
            options=feature_info['categorical_values']['parental_level_of_education']
        )
        lunch = st.selectbox('Lunch Type', options=feature_info['categorical_values']['lunch'])

    with col3:
        test_prep = st.selectbox(
            'Test Preparation Course',
            options=feature_info['categorical_values']['test_preparation_course']
        )
        reading_score = st.slider('Reading Score', min_value=0, max_value=100, value=75)
        writing_score = st.slider('Writing Score', min_value=0, max_value=100, value=75)

    st.markdown("---")

    if st.button('🚀 Predict Math Score', use_container_width=True):

        # Create input dataframe
        input_data = pd.DataFrame({
            'gender': [gender],
            'race_ethnicity': [race_ethnicity],
            'parental_level_of_education': [parental_education],
            'lunch': [lunch],
            'test_preparation_course': [test_prep],
            'reading_score': [reading_score],
            'writing_score': [writing_score]
        })

        st.write("### Prediction Results")
        with st.spinner("Calculating..."):
            # Predict using the single loaded pipeline
            prediction = pipeline.predict(input_data)[0]

            st.success(f"**Predicted Math Score: `{prediction:.2f}`**")

            # --- ANALYSIS DASHBOARD ---
            st.markdown("---")
            st.subheader("Prediction Analysis Dashboard")

            fig = px.histogram(
                df, x='math_score',
                title='Prediction vs. Overall Math Score Distribution',
                labels={'math_score': 'Math Score'},
                color_discrete_sequence=['#4A90E2']
            )
            fig.add_vline(x=prediction, line_width=3, line_dash="dash", line_color="red",
                           annotation_text=f"Your Prediction: {prediction:.2f}", annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)