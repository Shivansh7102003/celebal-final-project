import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import os

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
        font-size: 2rem;
        margin: 2rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load models and metadata
@st.cache_resource
def load_models():
    """Load all saved models and metadata"""
    models = {}
    model_files = [f for f in os.listdir('saved_models') if f.endswith('_model.pkl')]
    
    for file in model_files:
        model_name = file.replace('_model.pkl', '').replace('_', ' ').title()
        models[model_name] = joblib.load(f'saved_models/{file}')
    
    # Load metadata
    with open('saved_models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load feature info
    with open('saved_models/feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    return models, metadata, feature_info

# Load data
try:
    models, metadata, feature_info = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Title
st.markdown('<h1 class="main-header">📚 Student Performance Predictor</h1>', unsafe_allow_html=True)
st.markdown("<center>Predict student's writing score based on various factors</center>", unsafe_allow_html=True)

# Sidebar for model selection
st.sidebar.header("🤖 Model Selection")
st.sidebar.markdown("---")

# Model selection
available_models = list(models.keys())
selected_model = st.sidebar.selectbox(
    "Choose a Model",
    available_models,
    index=0  # Default to the first model
)

# Display model info
if selected_model:
    # Find the matching model info
    model_info = None
    for score in metadata['all_scores']:
        if score['model'].replace(' ', ' ').title() == selected_model:
            model_info = score
            break
    
    if model_info:
        st.sidebar.markdown("### 📊 Model Performance")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("R² Score", f"{model_info['test_r2']:.4f}")
            st.metric("RMSE", f"{model_info['test_rmse']:.4f}")
        with col2:
            st.metric("MAE", f"{model_info['test_mae']:.4f}")
            st.metric("Train R²", f"{model_info['train_r2']:.4f}")
        
        if model_info.get('best_params'):
            st.sidebar.markdown("### 🔧 Best Parameters")
            for param, value in model_info['best_params'].items():
                param_name = param.replace('regressor__', '')
                st.sidebar.write(f"**{param_name}:** {value}")

# Main content area
st.header("📝 Enter Student Information")
st.markdown("Please provide the following information to predict the student's writing score:")

# Create the prediction form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 Demographics")
        gender = st.selectbox(
            "Gender",
            options=feature_info['categorical_values']['gender'],
            help="Select the student's gender"
        )
        
        race_ethnicity = st.selectbox(
            "Race/Ethnicity",
            options=feature_info['categorical_values']['race_ethnicity'],
            help="Select the student's race/ethnicity group"
        )
        
    with col2:
        st.markdown("### 🎓 Background")
        parental_level_of_education = st.selectbox(
            "Parental Level of Education",
            options=feature_info['categorical_values']['parental_level_of_education'],
            help="Select the highest education level of the student's parents"
        )
        
        lunch = st.selectbox(
            "Lunch Type",
            options=feature_info['categorical_values']['lunch'],
            help="Type of lunch plan"
        )
        
    with col3:
        st.markdown("### 📚 Academic")
        test_preparation_course = st.selectbox(
            "Test Preparation Course",
            options=feature_info['categorical_values']['test_preparation_course'],
            help="Whether the student completed test preparation course"
        )
        
        math_score = st.number_input(
            "Math Score", 
            min_value=0, 
            max_value=100, 
            value=70,
            help="Student's math score (0-100)"
        )
        
        reading_score = st.number_input(
            "Reading Score", 
            min_value=0, 
            max_value=100, 
            value=70,
            help="Student's reading score (0-100)"
        )
    
    submitted = st.form_submit_button("🔮 Predict Writing Score", use_container_width=True)

# Prediction section
if submitted:
    # Prepare input data - MUST match the exact order and names used during training
    input_data = pd.DataFrame({
        'gender': [gender],
        'race_ethnicity': [race_ethnicity],
        'parental_level_of_education': [parental_level_of_education],
        'lunch': [lunch],
        'test_preparation_course': [test_preparation_course],
        'math_score': [math_score],
        'reading_score': [reading_score]
    })
    
    try:
        # Make prediction using the selected model
        model = models[selected_model]
        prediction = model.predict(input_data)[0]
        
        # Display prediction
        st.markdown("---")
        st.header("🎯 Prediction Result")
        
        # Create columns for centered display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Ensure prediction is within valid range
            prediction = max(0, min(100, prediction))
            
            st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Writing Score</h2>
                    <h1 style="color: {'#28a745' if prediction >= 70 else '#ffc107' if prediction >= 50 else '#dc3545'}; font-size: 4rem;">{prediction:.1f}/100</h1>
                </div>
            """, unsafe_allow_html=True)
        
        # Progress bar with color
        progress_color = "normal" if prediction >= 70 else "warning" if prediction >= 50 else "error"
        st.progress(prediction/100)
        
        # Performance message
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            if prediction >= 90:
                st.success("🌟 **Excellent!** The student is predicted to perform exceptionally well in writing.")
            elif prediction >= 80:
                st.success("👏 **Very Good!** The student is expected to achieve strong results in writing.")
            elif prediction >= 70:
                st.info("👍 **Good!** The student is on track for solid performance in writing.")
            elif prediction >= 60:
                st.warning("⚠️ **Average.** There's room for improvement in writing skills.")
            else:
                st.error("❗ **Below Average.** Additional support in writing is strongly recommended.")
        
        # Insights based on input
        st.markdown("### 💡 Key Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Score Analysis")
            # Compare with average scores
            avg_math = 66.1  # You can calculate these from your dataset
            avg_reading = 69.2
            avg_writing = 68.1
            
            math_diff = math_score - avg_math
            reading_diff = reading_score - avg_reading
            writing_diff = prediction - avg_writing
            
            st.metric("Math Score vs Average", f"{math_score}", f"{math_diff:+.1f}")
            st.metric("Reading Score vs Average", f"{reading_score}", f"{reading_diff:+.1f}")
            st.metric("Predicted Writing vs Average", f"{prediction:.1f}", f"{writing_diff:+.1f}")
        
        with col2:
            st.markdown("#### 🎯 Recommendations")
            
            if test_preparation_course == "none":
                st.warning("💡 **Test Prep:** Consider enrolling in a test preparation course. Students who complete test prep typically score 10-15 points higher!")
            else:
                st.success("✅ **Test Prep:** Great! The student has completed test preparation.")
            
            if lunch == "free/reduced":
                st.info("📚 **Resources:** The student may qualify for additional academic support programs.")
            
            if parental_level_of_education in ["some high school", "high school"]:
                st.info("👨‍👩‍👧‍👦 **Family Support:** Consider family engagement programs to support academic success.")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        with st.expander("🔍 Debug Information"):
            st.write("**Error Details:**", str(e))
            st.write("**Input Data:**")
            st.dataframe(input_data)
            st.write("**Selected Model:**", selected_model)
            st.write("**Model Type:**", type(model))

# Additional Analysis Section
st.markdown("---")
if st.button("📈 Show Detailed Analysis Dashboard", use_container_width=True):
    st.header("📊 Detailed Analysis Dashboard")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Feature Importance", "Score Distribution"])
    
    with tab1:
        st.subheader("🔍 Model Performance Comparison")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(metadata['all_scores'])
        
        # R2 Score comparison
        fig1 = px.bar(comparison_df, x='model', y='test_r2', 
                     title="R² Score Comparison Across Models",
                     labels={'test_r2': 'R² Score', 'model': 'Model'},
                     color='test_r2',
                     color_continuous_scale='viridis',
                     text='test_r2')
        fig1.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig1.update_layout(showlegend=False, xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig1, use_container_width=True)
        
        # RMSE vs MAE scatter plot
        fig2 = px.scatter(comparison_df, x='test_rmse', y='test_mae', 
                         text='model', size='test_r2',
                         title="Error Metrics Comparison (Lower is Better)",
                         labels={'test_rmse': 'RMSE', 'test_mae': 'MAE'},
                         color='test_r2',
                         color_continuous_scale='viridis')
        fig2.update_traces(textposition='top center')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed table
        st.subheader("📋 Detailed Metrics")
        display_df = comparison_df[['model', 'test_r2', 'test_rmse', 'test_mae']].round(4)
        display_df = display_df.sort_values('test_r2', ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("🎯 Feature Importance Analysis")
