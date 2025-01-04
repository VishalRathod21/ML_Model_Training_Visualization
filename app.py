import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix
import pickle
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# App Title
st.markdown("<h1 style='text-align: center; font-size: 3em; background: -webkit-linear-gradient(45deg, #E6194B, #3CB44B, #FFE119, #4363D8, #F58231, #911EB4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Automated Machine Learning Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### A User-Friendly Tool for Model Training and Visualization")

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
    st.markdown("<h2 style='background: -webkit-linear-gradient(45deg, #E6194B, #3CB44B, #FFE119, #4363D8, #F58231, #911EB4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Upload Your Dataset</h2>", unsafe_allow_html=True)
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

            # Model Selection
            st.markdown("Model Selection")
            st.markdown("**Description**: Select the model type you want to train, and choose the target column. The models available include Logistic Regression, Linear Regression, Random Forest, SVM, Decision Tree, KNN, and Gradient Boosting.")
            target_column = st.selectbox("Select Target Column", data.columns)
            model_type = st.selectbox("Select Model Type", ['Logistic Regression', 'Linear Regression', 'Random Forest', 'SVM', 'Decision Tree', 'KNN', 'Gradient Boosting'])

            # Hyperparameters for Random Forest
            if model_type == 'Random Forest':
                n_estimators = st.slider("Number of Estimators", 10, 200, 100)
                max_depth = st.slider("Max Depth", 1, 20, 10)
            elif model_type == 'Gradient Boosting':
                n_estimators = st.slider("Number of Estimators", 10, 200, 100)
                learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)

            # Model Training Function
            def train_model(data, target_column, model_type):
                X = data.drop(columns=[target_column])
                y = data[target_column]

                # Determine model type
                if model_type == 'Linear Regression':
                    model = LinearRegression()
                elif model_type == 'Logistic Regression':
                    model = LogisticRegression()
                elif model_type == 'Random Forest':
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth) if y.dtype == 'int64' else RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                elif model_type == 'SVM':
                    model = SVC() if y.dtype == 'int64' else SVR()
                elif model_type == 'Decision Tree':
                    model = DecisionTreeClassifier() if y.dtype == 'int64' else DecisionTreeRegressor()
                elif model_type == 'KNN':
                    model = KNeighborsClassifier() if y.dtype == 'int64' else KNeighborsRegressor()
                elif model_type == 'Gradient Boosting':
                    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate) if y.dtype == 'int64' else GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)

                # Train Model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

                return model, X_test, y_test, y_pred

            model, X_test, y_test, y_pred = train_model(data, target_column, model_type)

            # Confusion Matrix (for classification)
            if model_type in ['Logistic Regression', 'Random Forest', 'SVM', 'Decision Tree', 'KNN', 'Gradient Boosting']:
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
                plt.xlabel('Predicted')
                plt.ylabel('True')
                st.pyplot(fig_cm)

            # Feature Importance (for Random Forest and Gradient Boosting)
            if model_type in ['Random Forest', 'Gradient Boosting']:
                feature_importances = model.feature_importances_
                features = X_test.columns
                fig_fi = px.bar(x=features, y=feature_importances, labels={'x': 'Features', 'y': 'Importance'}, title="Feature Importance")
                st.plotly_chart(fig_fi)

            # Residual Plot (for regression models)
            if model_type in ['Linear Regression', 'Random Forest', 'SVR', 'Decision Tree', 'KNN', 'Gradient Boosting']:
                residuals = y_test - y_pred
                fig_res = plt.figure(figsize=(8, 6))
                sns.scatterplot(x=y_pred, y=residuals)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel('Predicted Values')
                plt.ylabel('Residuals')
                st.pyplot(fig_res)

            # Save Model
            st.markdown("Save Your Trained Model")
            st.markdown("**Description**: Once you've trained your model, you can save it as a `.pkl` file. This allows you to reuse the model later without retraining it.")
            if st.button("Save Model"):
                with open("trained_model.pkl", "wb") as file:
                    pickle.dump(model, file)
                st.success("Model saved successfully!")

            # Save Processed Data
            st.markdown("Save Processed Data")
            st.markdown("**Description**: Save the preprocessed data after cleaning, encoding, and scaling. This can be useful for future use or further analysis.")
            if st.button("Save Processed Data"):
                data.to_csv("processed_data.csv", index=False)
                st.success("Data saved successfully!")
else:
    st.sidebar.warning("Please log in to proceed.")
