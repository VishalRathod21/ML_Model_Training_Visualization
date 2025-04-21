import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import (accuracy_score, classification_report, mean_squared_error, 
                            r2_score, confusion_matrix, silhouette_score, mean_absolute_error, roc_curve, auc)
import shap
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from scipy import stats
import base64
import warnings
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.warning("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

# -------------------- App Configuration --------------------
st.set_page_config(page_title="MLFlowX", page_icon="🤖", layout="wide")

# -------------------- Authentication --------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = None

# -------------------- Header --------------------
st.markdown(
    "<h1 style='font-size: 3em; font-family: Arial, sans-serif;'>"
    "<span style='color: yellow;'>MLFlowX: </span>Automating Machine Learning at Your Fingertips🤖📊"
    "</h1>", unsafe_allow_html=True
)
st.markdown("### A User-Friendly Tool for Model Training and Visualization")
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "This portal offers an easy way to upload datasets, preprocess data, visualize trends, and train machine learning models. "
    "Whether you're a beginner or an expert, explore data, select features, train models, evaluate performance, and save your models. "
    "Get started with just a few clicks!"
)

# -------------------- Sidebar Login --------------------
users = {'admin': {'password': 'admin123', 'role': 'admin'}, 
         'user': {'password': 'user123', 'role': 'user'}}

if not st.session_state['logged_in']:
    with st.sidebar:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in users and users[username]['password'] == password:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['user_role'] = users[username]['role']
                st.success(f"Welcome {username}!")
            else:
                st.error("Invalid credentials")
    st.stop()

# -------------------- Main Application --------------------
if st.session_state['logged_in']:
    # -------------------- Sidebar Controls --------------------
    with st.sidebar:
        st.title(f"Welcome {st.session_state['username']}")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['user_role'] = None
            st.experimental_rerun()
        
        st.markdown("---")
        st.header("Navigation")
        app_page = st.radio("Go to", ["Data Upload", "Data Preprocessing", 
                                    "Feature Engineering", "Model Training", 
                                    "Model Evaluation", "Deployment", "Export & Reporting",
                                    "NLP", "Chatbot"])

    # -------------------- Data Upload Page --------------------
    if app_page == "Data Upload":
        st.header("📤 Data Upload & Exploration")
        uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "json", "parquet"])
        
        if uploaded_file is not None:
            @st.cache_data
            def load_data(file):
                ext = file.name.split('.')[-1]
                try:
                    if ext == "csv":
                        return pd.read_csv(file)
                    elif ext == "xlsx":
                        return pd.read_excel(file)
                    elif ext == "json":
                        return pd.read_json(file)
                    elif ext == "parquet":
                        return pd.read_parquet(file)
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    return None
            
            df = load_data(uploaded_file)
            st.session_state['data'] = df
            
            if df is not None:
                st.success("Data loaded successfully!")
                with st.expander("Data Preview"):
                    st.dataframe(df.head())
                
                with st.expander("Data Exploration"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Dataset Summary**")
                        buffer = StringIO()
                        df.info(buf=buffer)
                        st.text(buffer.getvalue())
                    with col2:
                        st.write("**Statistical Summary**")
                        st.write(df.describe())
                    
                    st.write("**Missing Values**")
                    missing = df.isnull().sum().to_frame(name="Missing Values")
                    missing["Percentage"] = (missing["Missing Values"] / len(df)) * 100
                    st.write(missing)

                # -------------------- Data Visualization Enhancements --------------------
                with st.expander("Interactive Data Visualization"):
                    st.subheader("Scatter Plot")
                    x_axis = st.selectbox("X-axis", df.columns)
                    y_axis = st.selectbox("Y-axis", df.columns)
                    color_by = st.selectbox("Color by", [None] + df.columns.tolist())
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by)
                    st.plotly_chart(fig)
                    
                    st.subheader("Correlation Heatmap")
                    numeric_cols = df.select_dtypes(include=np.number).columns
                    corr = df[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis')
                    st.plotly_chart(fig)

    # -------------------- Data Preprocessing Page --------------------
    elif app_page == "Data Preprocessing":
        st.header("🧹 Data Preprocessing")
        if st.session_state['data'] is None:
            st.warning("Upload data first!")
            st.stop()
        
        df = st.session_state['data']
        
        with st.expander("Handle Missing Values"):
            st.subheader("Missing Value Treatment")
            missing_method = st.selectbox("Imputation Method", 
                ["Drop NA", "Mean", "Median", "Mode", "KNN Imputer", "MICE", "Custom Value"])
            
            if missing_method == "Custom Value":
                custom_value = st.text_input("Enter custom value for imputation")
            
            if st.button("Apply Missing Value Treatment"):
                if missing_method == "Drop NA":
                    df = df.dropna()
                elif missing_method in ["Mean", "Median", "Mode"]:
                    imputer = SimpleImputer(strategy=missing_method.lower())
                    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
                elif missing_method == "KNN Imputer":
                    imputer = KNNImputer(n_neighbors=5)
                    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
                elif missing_method == "Custom Value":
                    df = df.fillna(custom_value)
                
                st.session_state['data'] = df
                st.success("Missing values handled!")
        
        with st.expander("Outlier Detection & Handling"):
            st.subheader("Outlier Management")
            outlier_method = st.selectbox("Detection Method", ["IQR", "Z-Score"])
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            selected_cols = st.multiselect("Select Columns", numeric_cols)
            
            if st.button("Handle Outliers"):
                for col in selected_cols:
                    if outlier_method == "IQR":
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
                    elif outlier_method == "Z-Score":
                        df = df[(np.abs(stats.zscore(df[col])) < 3)]
                
                st.session_state['data'] = df
                st.success("Outliers handled!")

        with st.expander("Feature Scaling"):
            st.subheader("Feature Scaling")
            scaling_method = st.selectbox("Scaling Technique", 
                ["Standard Scaler", "MinMax Scaler", "Robust Scaler"])
            
            if st.button("Apply Scaling"):
                numeric_cols = df.select_dtypes(include=np.number).columns
                if scaling_method == "Standard Scaler":
                    scaler = StandardScaler()
                elif scaling_method == "MinMax Scaler":
                    scaler = MinMaxScaler()
                else:
                    scaler = RobustScaler()
                
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                st.session_state['data'] = df
                st.success("Scaling applied!")

    # -------------------- Feature Engineering Page --------------------
    elif app_page == "Feature Engineering":
        st.header("⚙️ Feature Engineering")
        if st.session_state['data'] is None:
            st.warning("Upload data first!")
            st.stop()
        
        df = st.session_state['data']
        
        with st.expander("Feature Transformation"):
            st.subheader("Feature Transformation")
            transform_method = st.selectbox("Transformation Method", 
                ["Log Transform", "Square Root", "Polynomial"])
            
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            selected_col = st.selectbox("Select Column", numeric_cols)
            
            if st.button("Apply Transformation"):
                if transform_method == "Log Transform":
                    df[selected_col] = np.log1p(df[selected_col])
                elif transform_method == "Square Root":
                    df[selected_col] = np.sqrt(df[selected_col])
                elif transform_method == "Polynomial":
                    degree = st.slider("Select Degree", 2, 5)
                    df[f"{selected_col}_poly_{degree}"] = df[selected_col] ** degree
                
                st.session_state['data'] = df
                st.success("Transformation applied!")
        
        with st.expander("Dimensionality Reduction"):
            st.subheader("Dimensionality Reduction")
            dr_method = st.selectbox("Reduction Method", ["PCA", "Feature Importance"])
            
            if dr_method == "PCA":
                numeric_cols = df.select_dtypes(include=np.number).columns
                if len(numeric_cols) < 2:
                    st.warning("PCA requires at least 2 numeric columns")
                else:
                    max_components = min(10, len(numeric_cols))
                    n_components = st.slider(
                        "Number of components", 
                        min_value=1,
                        max_value=max_components,
                        value=min(3, max_components)
                    )
                    if st.button("Apply PCA"):
                        pca = PCA(n_components=n_components)
                        pca_features = pca.fit_transform(df[numeric_cols])
                        pca_df = pd.DataFrame(pca_features, columns=[f"PC{i+1}" for i in range(n_components)])
                        df = pd.concat([df.drop(columns=numeric_cols), pca_df], axis=1)
                        st.session_state['data'] = df
                        st.success(f"PCA applied! {n_components} components created.")
            
            elif dr_method == "Feature Importance":
                target = st.selectbox("Select Target Column", df.columns)
                if st.button("Calculate Feature Importance"):
                    X = df.drop(columns=[target])
                    y = df[target]
                    
                    model = RandomForestRegressor() if y.dtype == 'float' else RandomForestClassifier()
                    model.fit(X, y)
                    importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
                    st.plotly_chart(fig)

    # -------------------- Model Training Page --------------------
    elif app_page == "Model Training":
        st.header("🤖 Model Training")
        if st.session_state['data'] is None:
            st.warning("Upload data first!")
            st.stop()
        
        df = st.session_state['data']
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("Model Configuration")
            problem_type = st.selectbox("Problem Type", ["Classification", "Regression", "Clustering"])
            target_col = st.selectbox("Select Target Column", df.columns) if problem_type != "Clustering" else None
            
            model_options = {
                "Classification": ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting"],
                "Regression": ["Linear Regression", "Random Forest", "SVR", "Gradient Boosting"],
                "Clustering": ["K-Means", "DBSCAN"]
            }
            model_choice = st.selectbox("Select Model", model_options[problem_type])
            
            hyper_params = {}
            if "Random Forest" in model_choice:
                hyper_params['n_estimators'] = st.slider("Number of Trees", 10, 500, 100)
                hyper_params['max_depth'] = st.slider("Max Depth", 2, 50, 10)
            elif "SVM" in model_choice:
                hyper_params['C'] = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
                hyper_params['kernel'] = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            elif model_choice == "K-Means":
                hyper_params['n_clusters'] = st.slider("Number of Clusters", 2, 10, 3)
        
        with col2:
            st.subheader("Training Configuration")
            if problem_type != "Clustering":
                test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
                random_state = st.number_input("Random State", 42)
                cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5)
            
            if st.button("Train Model"):
                with st.spinner("Training in progress..."):
                    try:
                        X = df.drop(columns=[target_col]) if target_col else df
                        y = df[target_col] if target_col else None
                        
                        # Model Initialization
                        if model_choice == "Logistic Regression":
                            model = LogisticRegression()
                        elif model_choice == "Linear Regression":
                            model = LinearRegression()   
                        elif model_choice == "Random Forest":
                            model = RandomForestClassifier(**hyper_params) if problem_type == "Classification" else RandomForestRegressor(**hyper_params)
                        elif model_choice == "Gradient Boosting":
                            model = GradientBoostingClassifier() if problem_type == "Classification" else GradientBoostingRegressor()
                        elif model_choice == "SVM":
                            model = SVC(**hyper_params) if problem_type == "Classification" else SVR(**hyper_params)
                        elif model_choice == "K-Means":
                            model = KMeans(**hyper_params)
                        else:
                            raise ValueError(f"Unsupported Model : {model_choice}")
                        # Model Training
                        if problem_type != "Clustering":
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=random_state)
                            model.fit(X_train, y_train)
                            
                            # Cross-Validation
                            cv_scores = cross_val_score(model, X, y, cv=cv_folds)
                            st.write(f"Cross-Validation Scores: {cv_scores}")
                            st.write(f"Mean CV Score: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
                            
                            # Evaluation
                            y_pred = model.predict(X_test)
                            if problem_type == "Classification":
                                st.write("Accuracy:", accuracy_score(y_test, y_pred))
                                st.write(classification_report(y_test, y_pred))
                                fig = px.imshow(confusion_matrix(y_test, y_pred), 
                                                labels=dict(x="Predicted", y="Actual"),
                                                color_continuous_scale='Blues')
                                st.plotly_chart(fig)
                            else:
                                st.write("MSE:", mean_squared_error(y_test, y_pred))
                                st.write("R2 Score:", r2_score(y_test, y_pred))
                                fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
                                fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                                            x1=y_test.max(), y1=y_test.max(), line=dict(color="red"))
                                st.plotly_chart(fig)
                        else:
                            clusters = model.fit_predict(X)
                            df['Cluster'] = clusters
                            st.session_state['data'] = df
                            score = silhouette_score(X, clusters)
                            st.write(f"Silhouette Score: {score:.2f}")
                            fig = px.scatter_matrix(df, dimensions=df.columns[:3], color="Cluster")
                            st.plotly_chart(fig)
                        
                        st.session_state['model'] = model
                        st.success("Training complete!")
                    
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")

    # -------------------- Model Evaluation Page --------------------
    elif app_page == "Model Evaluation":
        st.header("📊 Model Evaluation")
        if st.session_state['model'] is None:
            st.warning("Train a model first!")
            st.stop()
        
        model = st.session_state['model']
        df = st.session_state['data']
        
        st.subheader("Model Explainability")
        explain_method = st.selectbox("Explanation Method", ["SHAP", "Feature Importance", "Partial Dependence"])
        
        if explain_method == "SHAP":
            with st.spinner("Generating SHAP explanations..."):
                explainer = shap.Explainer(model)
                sample = shap.sample(df.select_dtypes(include=np.number), 100)
                shap_values = explainer(sample)
                fig = shap.summary_plot(shap_values, sample, plot_type="bar")
                st.pyplot(fig)
        
        elif explain_method == "Feature Importance" and hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': df.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig)
        
        elif explain_method == "Partial Dependence":
            feature = st.selectbox("Select Feature", df.columns)
            with st.spinner("Calculating PDP..."):
                shap.partial_dependence_plot(
                    feature, model.predict, df, ice=False,
                    model_expected_value=True, feature_expected_value=True
                )
                st.pyplot(plt.gcf())

    # -------------------- Deployment Page --------------------
    elif app_page == "Deployment":
        st.header("🚀 Model Deployment")
        if st.session_state['model'] is None:
            st.warning("Train a model first!")
            st.stop()
        
        model = st.session_state['model']
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Export")
            model_name = st.text_input("Model Name", "my_model")
            if st.button("Export Model"):
                joblib.dump(model, f"{model_name}.pkl")
                with open(f"{model_name}.pkl", "rb") as f:
                    st.download_button("Download Model", f, file_name=f"{model_name}.pkl")
        
        with col2:
            st.subheader("API Generation")
            if st.button("Generate FastAPI Code"):
                api_code = f"""
                from fastapi import FastAPI
                import joblib
                import pandas as pd

                app = FastAPI()
                model = joblib.load('{model_name}.pkl')

                @app.post("/predict")
                async def predict(input_data: dict):
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)
                    return {{"prediction": prediction.tolist()}}
                """
                st.download_button("Download API Code", api_code, file_name="api.py")

    # -------------------- Export & Reporting Page --------------------
    elif app_page == "Export & Reporting":
        st.header("📤 Export & Reporting")
        if st.session_state['data'] is None:
            st.warning("Upload data first!")
            st.stop()
        
        df = st.session_state['data']
        
        st.subheader("Export Data")
        export_format = st.selectbox("Select Export Format", ["CSV", "Excel", "JSON"])
        export_name = st.text_input("Export File Name", "exported_data")
        
        if st.button("Export Data"):
            if export_format == "CSV":
                df.to_csv(f"{export_name}.csv", index=False)
            elif export_format == "Excel":
                df.to_excel(f"{export_name}.xlsx", index=False)
            elif export_format == "JSON":
                df.to_json(f"{export_name}.json", orient="records")
            st.success(f"Data exported as {export_name}.{export_format.lower()}")

        st.subheader("Generate Report")
        if st.button("Generate Report"):
            report = f"""
            # MLFlowX Report
            ## Dataset Summary
            - Number of Rows: {df.shape[0]}
            - Number of Columns: {df.shape[1]}
            - Missing Values: {df.isnull().sum().sum()}
            ## Statistical Summary
            {df.describe().to_markdown()}
            """
            
            # Add Feature Importance to the Report
            if st.session_state['model'] is not None and hasattr(st.session_state['model'], 'feature_importances_'):
                model = st.session_state['model']
                feature_importance = pd.DataFrame({
                    'Feature': df.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                report += f"""
                ## Feature Importance
                {feature_importance.to_markdown()}
                """
            
            # Add Model Comparison to the Report
            if st.session_state.get('model_comparison_results') is not None:
                model_comparison_results = st.session_state['model_comparison_results']
                report += f"""
                ## Model Comparison Results
                {model_comparison_results.to_markdown()}
                """
            
            st.download_button("Download Report", report, file_name="report.md")

        # -------------------- Feature Importance Visualization --------------------
        if st.session_state['model'] is not None and hasattr(st.session_state['model'], 'feature_importances_'):
            st.subheader("Feature Importance")
            model = st.session_state['model']
            feature_importance = pd.DataFrame({
                'Feature': df.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', title="Feature Importance")
            st.plotly_chart(fig)

        # -------------------- Model Comparison --------------------
        if st.session_state.get('model_comparison_results') is not None:
            st.subheader("Model Comparison")
            model_comparison_results = st.session_state['model_comparison_results']
            st.dataframe(model_comparison_results)
            
            # Visualize Model Comparison
            fig = px.bar(model_comparison_results, x='Model', y='Score', color='Model', title="Model Comparison Scores")
            st.plotly_chart(fig)

    # -------------------- NLP Page --------------------
    elif app_page == "NLP":
        st.header("📝 Natural Language Processing")
        
        st.subheader("Text Analysis Tools")
        nlp_task = st.selectbox("Select NLP Task", 
            ["Text Classification", "Sentiment Analysis", "Text Summarization", 
             "Named Entity Recognition", "Topic Modeling"])
        
        if nlp_task == "Text Classification":
            st.write("Upload your text data for classification")
            text_input = st.text_area("Enter text for classification")
            if st.button("Classify"):
                # Add text classification logic here
                st.write("Classification results will appear here")
        
        elif nlp_task == "Sentiment Analysis":
            st.write("Analyze sentiment of your text")
            text_input = st.text_area("Enter text for sentiment analysis")
            if st.button("Analyze"):
                # Add sentiment analysis logic here
                st.write("Sentiment analysis results will appear here")
        
        elif nlp_task == "Text Summarization":
            st.write("Summarize your text")
            text_input = st.text_area("Enter text to summarize")
            if st.button("Summarize"):
                # Add text summarization logic here
                st.write("Summary will appear here")
        
        elif nlp_task == "Named Entity Recognition":
            st.write("Extract named entities from text")
            text_input = st.text_area("Enter text for NER")
            if st.button("Extract Entities"):
                # Add NER logic here
                st.write("Named entities will appear here")
        
        elif nlp_task == "Topic Modeling":
            st.write("Discover topics in your text corpus")
            text_input = st.text_area("Enter multiple documents (one per line)")
            if st.button("Model Topics"):
                # Add topic modeling logic here
                st.write("Topics will appear here")

    # -------------------- Chatbot Page --------------------
    elif app_page == "Chatbot":
        st.header("🤖 Interactive Chatbot (Powered by Gemini)")
        
        # Initialize chat history and settings
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'chat_settings' not in st.session_state:
            st.session_state.chat_settings = {
                'max_history': 10,
                'temperature': 0.7,
                'show_typing': True
            }
        
        # Chat settings sidebar
        with st.sidebar:
            st.subheader("Chat Settings")
            st.session_state.chat_settings['max_history'] = st.slider(
                "Max Conversation History", 5, 50, 10,
                help="Maximum number of messages to keep in history"
            )
            st.session_state.chat_settings['temperature'] = st.slider(
                "Response Temperature", 0.0, 1.0, 0.7,
                help="Higher values make responses more creative but less focused"
            )
            st.session_state.chat_settings['show_typing'] = st.toggle(
                "Show Typing Indicator", True,
                help="Display typing animation while waiting for response"
            )
            
            # Conversation management
            st.subheader("Conversation Management")
            if st.button("Clear Conversation"):
                st.session_state.chat_history = []
                st.success("Conversation cleared!")
            
            # Export options
            st.subheader("Export Options")
            export_format = st.selectbox("Export Format", ["Text", "Markdown", "JSON"])
            if st.button("Export Conversation"):
                if export_format == "Text":
                    chat_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
                    st.download_button(
                        label="Download as Text",
                        data=chat_text,
                        file_name="chat_history.txt",
                        mime="text/plain"
                    )
                elif export_format == "Markdown":
                    chat_md = "\n".join([f"**{msg['role']}**: {msg['content']}" for msg in st.session_state.chat_history])
                    st.download_button(
                        label="Download as Markdown",
                        data=chat_md,
                        file_name="chat_history.md",
                        mime="text/markdown"
                    )
                else:  # JSON
                    import json
                    chat_json = json.dumps(st.session_state.chat_history, indent=2)
                    st.download_button(
                        label="Download as JSON",
                        data=chat_json,
                        file_name="chat_history.json",
                        mime="application/json"
                    )
        
        # Initialize Gemini model
        if 'gemini_model' not in st.session_state:
            if GEMINI_API_KEY:
                try:
                    # List available models
                    available_models = genai.list_models()
                    
                    # Preferred model versions in order of preference
                    preferred_models = [
                        'gemini-1.5-flash',
                        'gemini-1.5-pro',
                        'gemini-pro',
                        'gemini-pro-vision'
                    ]
                    
                    # Find the first available preferred model
                    model_name = None
                    for preferred_model in preferred_models:
                        for model in available_models:
                            if preferred_model in model.name.lower():
                                model_name = model.name
                                break
                        if model_name:
                            break
                    
                    if model_name:
                        st.session_state.gemini_model = genai.GenerativeModel(model_name)
                        st.session_state.model_name = model_name
                    else:
                        st.error("No suitable Gemini model found. Please check your API access.")
                        st.stop()
                except Exception as e:
                    st.error(f"Error initializing Gemini model: {str(e)}")
                    st.stop()
            else:
                st.error("Gemini API key not configured. Please set GEMINI_API_KEY in your .env file.")
                st.stop()
        
        # Display current model being used
        if hasattr(st.session_state, 'model_name'):
            st.info(f"Using model: {st.session_state.model_name}")
        
        # Display chat history with improved UI
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "timestamp" in message:
                    st.caption(message["timestamp"])
        
        # Chat input with improved UI
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to chat history with timestamp
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                try:
                    # Show typing indicator if enabled
                    if st.session_state.chat_settings['show_typing']:
                        with st.spinner("Thinking..."):
                            # Create chat history context for Gemini
                            chat_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
                            
                            # Generate response using Gemini with temperature setting
                            response = st.session_state.gemini_model.generate_content(
                                chat_context,
                                generation_config=genai.types.GenerationConfig(
                                    temperature=st.session_state.chat_settings['temperature']
                                )
                            )
                    else:
                        # Create chat history context for Gemini
                        chat_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
                        
                        # Generate response using Gemini with temperature setting
                        response = st.session_state.gemini_model.generate_content(
                            chat_context,
                            generation_config=genai.types.GenerationConfig(
                                temperature=st.session_state.chat_settings['temperature']
                            )
                        )
                    
                    # Display the response
                    st.write(response.text)
                    
                    # Add assistant response to chat history with timestamp
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response.text,
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Trim history if it exceeds max_history
                    if len(st.session_state.chat_history) > st.session_state.chat_settings['max_history']:
                        st.session_state.chat_history = st.session_state.chat_history[-st.session_state.chat_settings['max_history']:]
                
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "Sorry, I encountered an error. Please try again.",
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
