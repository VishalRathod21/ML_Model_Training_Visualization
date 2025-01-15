import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix
import pickle
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import io


# App Title
st.markdown(
    "<h1 style='text-align: center; font-size: 3em; font-family: Arial, sans-serif;'>"
    "<span style='color: yellow;'>Automated</span> Machine Learning Training Model Dashboard 🤖📊"
    "</h1>", unsafe_allow_html=True
)
st.markdown("### A User-Friendly Tool for Model Training and Visualization")
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "This portal offers an easy way to upload datasets, preprocess data, visualize trends, and train machine learning models. "
    "Whether you're a beginner or an expert, explore data, select features, train models, evaluate performance, and save your models. "
    "Get started with just a few clicks!"
)

# Authentication State
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Sidebar Login
st.sidebar.title("Login")
users = {'admin': 'admin123', 'user': 'user123'}

if not st.session_state['logged_in']:
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login = st.sidebar.button("Login")

    if login:
        if username in users and users[username] == password:
            st.sidebar.success(f"Welcome {username}!")
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
        else:
            st.sidebar.error("Invalid username or password!")

# Main App Logic After Login
if st.session_state['logged_in']:
    st.sidebar.success(f"Logged in as: {st.session_state['username']}")

    # Logout Button
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.sidebar.warning("Logged out. Please log in again.")

    # Upload Dataset
    st.markdown("<h2 style='background: -webkit-linear-gradient(45deg, #4A90E2, #4A90E2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Upload Your Dataset</h2>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Description**: Upload a CSV, Excel, JSON, or Parquet file that contains your dataset. The dataset will be preprocessed, including handling missing values and scaling numerical features.")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "json", "parquet"])

    if uploaded_file is not None:
        @st.cache_data
        def load_and_preprocess_data(uploaded_file, file_type):
            try:
                # Handle the case when openpyxl is missing for .xlsx files
                if file_type == "xlsx":
                    try:
                        import openpyxl  # Attempt to import openpyxl
                    except ImportError:
                        st.error("Missing optional dependency 'openpyxl'. Please install it using 'pip install openpyxl'.")
                        return None
                
                # Load data based on file type with encoding handling
                if file_type == "csv":
                    data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                elif file_type == "xlsx":
                    data = pd.read_excel(uploaded_file)
                elif file_type == "json":
                    data = pd.read_json(uploaded_file)
                elif file_type == "parquet":
                    data = pd.read_parquet(uploaded_file)

                # Handle missing values
                imputer = SimpleImputer(strategy='mean')
                numeric_data = data.select_dtypes(include=['float64', 'int64'])
                data[numeric_data.columns] = imputer.fit_transform(numeric_data)

                # Encode categorical columns
                label_encoders = {}
                categorical_data = data.select_dtypes(include=['object'])
                for col in categorical_data.columns:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))

                # Scale numeric columns
                scaler = StandardScaler()
                data[numeric_data.columns] = scaler.fit_transform(data[numeric_data.columns])
                return data

            except Exception as e:
                st.error(f"Error loading file: {e}")
                return None

        file_type = uploaded_file.name.split('.')[-1]
        data = load_and_preprocess_data(uploaded_file, file_type)

        if data is not None:
            st.write("Processed Data Preview:", data.head())

            # Exploratory Data Analysis (EDA)
            st.markdown("### Exploratory Data Analysis (EDA)")
            st.markdown("**Description**: Perform basic analysis like statistics, distribution, and pairwise relationships.")
            if st.checkbox("Show Statistics"):
                st.write(data.describe())

            if st.checkbox("Show Correlation Heatmap"):
                corr = data.corr()
                fig, ax = plt.subplots(figsize=(10, 8))  # Create figure and axis objects
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                st.pyplot(fig)  # Pass the figure object to st.pyplot()


            # Handle Missing Values, Duplicates, and Data Scaling
            st.markdown("### Data Preprocessing Options")
            st.markdown("**Description**: Handle missing values, remove duplicates, and more.")
            
            if st.checkbox("Remove Duplicates"):
                before_rows = data.shape[0]
                data = data.drop_duplicates()
                after_rows = data.shape[0]
                st.write(f"Removed {before_rows - after_rows} duplicate rows.")
                
            if st.checkbox("Handle Missing Values"):
                missing_strategy = st.selectbox("Choose Strategy for Missing Values", ["Mean", "Median", "Most Frequent"])
                imputer = SimpleImputer(strategy=missing_strategy.lower())
                numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Define numeric_data here
                data[numeric_data.columns] = imputer.fit_transform(data[numeric_data.columns])
                st.write("Missing values handled using", missing_strategy)

            # Feature Selection/Engineering
            st.markdown("### Feature Engineering and Selection")
            st.markdown("**Description**: Feature selection and importance visualization.")
            if st.checkbox("Show Feature Importance (Random Forest)"):

                target_column = st.selectbox("Select Target Column for Feature Importance", data.columns)
                X = data.drop(columns=[target_column])
                y = data[target_column]
                model = RandomForestClassifier() if y.dtype == 'int64' else RandomForestRegressor()
                model.fit(X, y)
                feature_importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': feature_importance
                }).sort_values(by='Importance', ascending=False)
                st.write(importance_df)

            # Model Selection
            st.markdown("### Model Selection")
            model_type = st.selectbox("Select Model Type", ['Linear Regression', 'Logistic Regression', 'Random Forest'])

            # Hyperparameters for Random Forest
            if model_type == 'Random Forest':
                n_estimators = st.slider("Number of Estimators", 10, 200, 100)
                max_depth = st.slider("Max Depth", 1, 20, 10)

            # Model Training
            st.markdown("### Train Model")
            if st.button("Train Model"):
                target_column = st.selectbox("Select Target Column", data.columns)
                X = data.drop(columns=[target_column])
                y = data[target_column]

                # Determine model type
                if model_type == 'Linear Regression':
                    model = LinearRegression()
                elif model_type == 'Logistic Regression':
                    model = LogisticRegression()
                elif model_type == 'Random Forest':
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth) if y.dtype == 'int64' else RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

                # Train Model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Evaluation
                    if y.dtype in ['float64', 'int64']:  # Regression
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        st.write(f"MSE: {mse:.2f}", f"R2: {r2:.2f}")
                    else:  # Classification
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"Accuracy: {accuracy * 100:.2f}%")
                        st.text(classification_report(y_test, y_pred))

                    # Visualizations
                    # Linear Regression Fit Line
                    if model_type == 'Linear Regression':
                        fig, ax = plt.subplots()
                        ax.scatter(y_test, y_pred)
                        ax.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], color='red', lw=2)
                        st.pyplot(fig)

                    # Confusion Matrix (for Classification)
                    if model_type != 'Linear Regression':
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                        st.pyplot(fig)

                    # Model Saving
                    import joblib
                    model_filename = f"{model_type.replace(' ', '_')}_model.pkl"
                    joblib.dump(model, model_filename)
                    st.success(f"Model trained successfully! You can download it below.")

                    # Model Download Button
                    st.download_button("Download Trained Model", model_filename, file_name=model_filename)
                    
                except Exception as e:
                    st.error(f"Error during model training: {e}")
