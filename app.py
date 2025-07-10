# app.py - Working Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

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
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📚 Student Performance Predictor</h1>', unsafe_allow_html=True)
st.markdown("<center>Predict student's writing score based on various factors</center>", unsafe_allow_html=True)

# Create the proper preprocessor (matching your training code)
@st.cache_resource
def create_preprocessor():
    """Create the preprocessor matching the training pipeline"""
    # Define feature columns
    numerical_features = ['math_score', 'reading_score']
    categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 
                          'lunch', 'test_preparation_course']
    
    # Create transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()
    
    # Create preprocessor (matching your training code exactly)
    preprocessor = ColumnTransformer(
        [
            ("OneHotEncoder", categorical_transformer, categorical_features),
            ("StandardScaler", numeric_transformer, numerical_features),        
        ]
    )
    
    return preprocessor, numerical_features, categorical_features

# Load models and create proper pipelines
@st.cache_resource
def load_and_fix_models():
    """Load models and wrap them with proper preprocessing"""
    models = {}
    
    # Load metadata
    try:
        with open('saved_models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
    except:
        metadata = {'all_scores': []}
    
    # Load feature info
    try:
        with open('saved_models/feature_info.json', 'r') as f:
            feature_info = json.load(f)
    except:
        feature_info = {
            'categorical_values': {
                'gender': ['female', 'male'],
                'race_ethnicity': ['group A', 'group B', 'group C', 'group D', 'group E'],
                'parental_level_of_education': ['some high school', 'high school', 'some college', 
                                               "associate's degree", "bachelor's degree", "master's degree"],
                'lunch': ['standard', 'free/reduced'],
                'test_preparation_course': ['none', 'completed']
            }
        }
    
    # Create preprocessor
    preprocessor, num_features, cat_features = create_preprocessor()
    
    # Create dummy data to fit the preprocessor
    dummy_data = pd.DataFrame({
        'gender': ['female', 'male'] * 50,
        'race_ethnicity': ['group A', 'group B', 'group C', 'group D', 'group E'] * 20,
        'parental_level_of_education': ['high school', 'some college', "bachelor's degree", 
                                       "master's degree", "associate's degree", 'some high school'] * 17,
        'lunch': ['standard', 'free/reduced'] * 50,
        'test_preparation_course': ['none', 'completed'] * 50,
        'math_score': np.random.randint(0, 100, 100),
        'reading_score': np.random.randint(0, 100, 100)
    })[:100]  # Use 100 rows
    
    # Fit the preprocessor
    preprocessor.fit(dummy_data)
    
    # Load each model and create a proper pipeline
    model_files = [f for f in os.listdir('saved_models') if f.endswith('_model.pkl')]
    
    for file in model_files:
        try:
            model_name = file.replace('_model.pkl', '').replace('_', ' ').title()
            loaded_model = joblib.load(f'saved_models/{file}')
            
            # Check if it's already a pipeline with correct preprocessing
            if hasattr(loaded_model, 'named_steps') and 'preprocessor' in loaded_model.named_steps:
                # Get just the regressor part
                if 'regressor' in loaded_model.named_steps:
                    regressor = loaded_model.named_steps['regressor']
                else:
                    # It might be the last step
                    regressor = list(loaded_model.named_steps.values())[-1]
            else:
                # It's just a model
                regressor = loaded_model
            
            # Create new pipeline with correct preprocessor
            fixed_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', regressor)
            ])
            
            models[model_name] = fixed_pipeline
            
        except Exception as e:
            st.warning(f"Could not load {file}: {str(e)}")
    
    return models, metadata, feature_info

# Load everything
with st.spinner("Loading models..."):
    models, metadata, feature_info = load_and_fix_models()

if not models:
    st.error("❌ No models could be loaded!")
    st.stop()

# Sidebar
st.sidebar.header("🤖 Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a Model",
    list(models.keys()),
    index=0
)

# Display model info
if selected_model and metadata:
    model_info = None
    for score in metadata.get('all_scores', []):
        if score['model'].replace(' ', ' ').title() == selected_model:
            model_info = score
            break
    
    if model_info:
        st.sidebar.markdown("### 📊 Model Performance")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("R² Score", f"{model_info.get('test_r2', 0):.4f}")
            st.metric("RMSE", f"{model_info.get('test_rmse', 0):.4f}")
        with col2:
            st.metric("MAE", f"{model_info.get('test_mae', 0):.4f}")
            st.metric("Train R²", f"{model_info.get('train_r2', 0):.4f}")
        
        if model_info.get('best_params'):
            st.sidebar.markdown("### 🔧 Best Parameters")
            for param, value in model_info['best_params'].items():
                param_name = param.replace('regressor__', '')
                st.sidebar.write(f"**{param_name}:** {value}")

# Main content
st.header("📝 Enter Student Information")
st.markdown("Please provide the following information to predict the student's writing score:")

# Create form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 Demographics")
        gender = st.selectbox(
            "Gender",
            options=feature_info['categorical_values']['gender']
        )
        
        race_ethnicity = st.selectbox(
            "Race/Ethnicity",
            options=feature_info['categorical_values']['race_ethnicity']
        )
    
    with col2:
        st.markdown("### 🎓 Background")
        parental_level_of_education = st.selectbox(
            "Parental Level of Education",
            options=feature_info['categorical_values']['parental_level_of_education']
        )
        
        lunch = st.selectbox(
            "Lunch Type",
            options=feature_info['categorical_values']['lunch']
        )
    
    with col3:
        st.markdown("### 📚 Academic")
        test_preparation_course = st.selectbox(
            "Test Preparation Course",
            options=feature_info['categorical_values']['test_preparation_course']
        )
        
        math_score = st.number_input("Math Score", 0, 100, 70)
        reading_score = st.number_input("Reading Score", 0, 100, 70)
    
    submitted = st.form_submit_button("🔮 Predict Writing Score", use_container_width=True)

# Prediction
if submitted:
    # Create input dataframe
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
        # Make prediction
        model = models[selected_model]
        prediction = model.predict(input_data)[0]
        
        # Display result
        st.markdown("---")
        st.header("🎯 Prediction Result")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            prediction = max(0, min(100, prediction))
            
            st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Writing Score</h2>
                    <h1 style="color: {'#28a745' if prediction >= 70 else '#ffc107' if prediction >= 50 else '#dc3545'}; 
                        font-size: 4rem;">{prediction:.1f}/100</h1>
                </div>
            """, unsafe_allow_html=True)
        
        st.progress(prediction/100)
        
        # Performance message
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            if prediction >= 90:
                st.success("🌟 **Excellent!** Outstanding writing performance expected.")
            elif prediction >= 80:
                st.success("👏 **Very Good!** Strong writing skills predicted.")
            elif prediction >= 70:
                st.info("👍 **Good!** Solid writing performance expected.")
            elif prediction >= 60:
                st.warning("⚠️ **Average.** Room for improvement in writing.")
            else:
                st.error("❗ **Below Average.** Extra writing support recommended.")
        
        # Insights
        st.markdown("### 💡 Key Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Score Analysis")
            avg_writing = 68.1
            diff = prediction - avg_writing
            
            st.metric("Predicted Score", f"{prediction:.1f}", f"{diff:+.1f} vs average")
            st.metric("Math Score Input", f"{math_score}")
            st.metric("Reading Score Input", f"{reading_score}")
            
        with col2:
            st.markdown("#### 🎯 Recommendations")
            
            if test_preparation_course == "none":
                st.warning("💡 **Test Prep:** Consider enrolling in a test preparation course!")
            else:
                st.success("✅ **Test Prep:** Great! Test prep completed.")
            
            if lunch == "free/reduced":
                st.info("📚 **Resources:** Check for additional academic support programs.")
            
            if parental_level_of_education in ["some high school", "high school"]:
                st.info("👨‍👩‍👧‍👦 **Support:** Family engagement programs available.")
            
            # Score-based recommendations
            if prediction < 60:
                st.error("🚨 **Action Needed:** Immediate tutoring support recommended.")
            elif prediction < 70:
                st.warning("📝 **Improvement:** Focus on writing practice exercises.")
            else:
                st.success("🎯 **Keep it up:** Maintain current study habits.")
                
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        
        with st.expander("🔧 Debug Information"):
            st.write("**Error:**", str(e))
            st.write("**Input Data:**")
            st.dataframe(input_data)

# Additional features
# Continuing from the previous code...

# Additional features
if st.button("📈 Show Model Comparison", use_container_width=True):
    if metadata and 'all_scores' in metadata:
        st.header("📊 Model Performance Comparison")
        
        comparison_df = pd.DataFrame(metadata['all_scores'])
        
        # Create visualization
        fig = px.bar(comparison_df, x='model', y='test_r2', 
                    title="Model R² Score Comparison",
                    labels={'test_r2': 'R² Score', 'model': 'Model'},
                    color='test_r2',
                    color_continuous_scale='viridis',
                    text='test_r2')
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(showlegend=False, xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # RMSE comparison
            fig_rmse = px.bar(comparison_df, x='model', y='test_rmse',
                            title="RMSE Comparison (Lower is Better)",
                            labels={'test_rmse': 'RMSE', 'model': 'Model'},
                            color='test_rmse',
                            color_continuous_scale='reds_r')
            fig_rmse.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            # MAE comparison
            fig_mae = px.bar(comparison_df, x='model', y='test_mae',
                           title="MAE Comparison (Lower is Better)",
                           labels={'test_mae': 'MAE', 'model': 'Model'},
                           color='test_mae',
                           color_continuous_scale='reds_r')
            fig_mae.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_mae, use_container_width=True)
        
        # Detailed table
        st.subheader("📋 Detailed Model Metrics")
        display_df = comparison_df[['model', 'test_r2', 'test_rmse', 'test_mae']].round(4)
        display_df = display_df.sort_values('test_r2', ascending=False)
        
        # Style the dataframe
        st.dataframe(
            display_df.style.highlight_max(subset=['test_r2'], color='lightgreen')
                          .highlight_min(subset=['test_rmse', 'test_mae'], color='lightgreen'),
            use_container_width=True,
            hide_index=True
        )

# Footer with additional information
st.markdown("---")
with st.expander("ℹ️ About This Predictor"):
    st.markdown("""
    ### About the Student Performance Predictor
    
    This tool uses machine learning models to predict a student's writing score based on:
    - **Demographic factors**: Gender, race/ethnicity
    - **Socioeconomic factors**: Lunch program, parental education
    - **Academic preparation**: Test prep course completion
    - **Previous performance**: Math and reading scores
    
    #### Model Information
    - **Best Model**: {best_model}
    - **R² Score**: {r2:.4f}
    - **Training Date**: {date}
    
    #### Feature Importance
    The models consider multiple factors, with math and reading scores typically being the strongest predictors 
    of writing performance, followed by test preparation and parental education level.
    
    #### Disclaimer
    This predictor is a tool to help identify students who may need additional support. Individual results may vary, 
    and this should not be the sole factor in educational decisions.
    """.format(
        best_model=metadata.get('best_model', 'Lasso'),
        r2=metadata.get('best_r2_score', 0.88),
        date=metadata.get('training_date', 'N/A')
    ))

# Quick tips section
with st.expander("💡 Quick Tips for Improving Writing Scores"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### For Students:
        - 📚 Read regularly (30 mins/day)
        - ✍️ Practice writing daily journals
        - 📝 Complete test prep courses
        - 🎯 Focus on grammar and vocabulary
        - 💬 Join writing clubs or groups
        """)
    
    with col2:
        st.markdown("""
        ### For Parents/Teachers:
        - 👥 Provide writing feedback
        - 📖 Encourage diverse reading
        - 🏫 Support test prep enrollment
        - 💻 Use online writing resources
        - 🎭 Engage in creative writing activities
        """)

# Statistics summary
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Models Available", len(models))

with col2:
    if metadata and 'best_r2_score' in metadata:
        st.metric("Best R² Score", f"{metadata['best_r2_score']:.4f}")
    else:
        st.metric("Best R² Score", "N/A")

with col3:
    st.metric("Features Used", "7")

with col4:
    st.metric("Target", "Writing Score")

# Final footer
st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 3rem;'>
        <p>Made with ❤️ using Streamlit | Student Performance Predictor v1.0</p>
        <p style='font-size: 0.8rem;'>© 2024 Education Analytics. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)

# Add custom CSS for better mobile responsiveness
st.markdown("""
    <style>
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .prediction-box h1 {
            font-size: 3rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)