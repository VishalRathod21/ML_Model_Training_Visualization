# Automated Machine Learning Dashboard 🚀🤖📊

## Overview
The **Automated Machine Learning Dashboard** is a user-friendly tool that simplifies the process of exploring datasets, preprocessing data, and training machine learning models. This tool is ideal for beginners and experts alike, offering a guided interface to go from raw data to trained models in just a few clicks.

---

## Features
- **Authentication**: Secure login functionality for personalized access.
- **Data Upload**: Supports CSV, Excel, JSON, and Parquet files.
- **Data Preprocessing**: 
  - Handles missing values
  - Encodes categorical features
  - Scales numerical features
  - Removes duplicate rows
- **Exploratory Data Analysis (EDA)**:
  - Statistical summaries
  - Correlation heatmaps
  - Distribution visualizations
- **Feature Engineering**:
  - Feature importance analysis with Random Forest
- **Model Training**:
  - Supports Linear Regression, Logistic Regression, and Random Forest
  - Hyperparameter tuning for Random Forest
  - Handles both classification and regression tasks
- **Model Evaluation**:
  - Metrics like accuracy, MSE, R², and classification reports
  - Confusion matrix visualizations
  - Regression fit line plots
- **Model Export**: Save trained models as `.pkl` files for future use.

---

## How to Use

### 1. Login
- Enter your username and password in the sidebar to log in.
- Default credentials:
  - Username: `admin`, Password: `admin123`
  - Username: `user`, Password: `user123`

### 2. Upload Dataset
- Upload your dataset in CSV, Excel, JSON, or Parquet format.
- The app will preprocess your data, handling missing values, encoding categorical variables, and scaling numerical features.

### 3. Explore Data
- View statistical summaries and generate visualizations such as correlation heatmaps.

### 4. Preprocess Data
- Remove duplicate rows or handle missing values using Mean, Median, or Most Frequent strategies.

### 5. Train Models
- Select a target column and choose one of the available models:
  - **Linear Regression** for predicting continuous values.
  - **Logistic Regression** for binary classification.
  - **Random Forest** for both regression and classification.
- Customize hyperparameters for Random Forest if selected.

### 6. Evaluate Models
- View performance metrics:
  - **Regression**: Mean Squared Error (MSE), R² Score
  - **Classification**: Accuracy, Confusion Matrix, and Classification Report
- Generate and view relevant visualizations.

### 7. Save and Download Models
- Save the trained model as a `.pkl` file.
- Use the "Download Trained Model" button to download the file for deployment or future use.

---
## Installation
To run this application locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repository-url.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

4. Open your browser and visit `http://localhost:8501` to access the dashboard.

## Technologies Used
- **Streamlit** - For creating the interactive web interface.
- **Pandas** - For data manipulation and analysis.
- **Scikit-learn** - For machine learning model training, preprocessing, and evaluation.
- **Plotly** - For interactive visualizations.
- **Seaborn and Matplotlib** - For static plots and visualizations.
- **Pickle** - For saving and loading models.

## Model Types Supported
- **Logistic Regression**
- **Linear Regression**
- **Random Forest**
- **SVM (Support Vector Machines)**
- **Decision Tree**
- **K-Nearest Neighbors (KNN)**
- **Gradient Boosting**

## How to Use
1. **Login:** Use the provided credentials (`admin`/`admin123` or `user`/`user123`) to log in.
2. **Upload Dataset:** Select your dataset file (CSV, Excel, JSON, or Parquet) and upload it. The system will automatically preprocess the data.
3. **Select Model:** Choose the model you want to train and the target column from your dataset.
4. **Train and Evaluate:** The app will train the model and display performance metrics, such as accuracy, MSE, R2, and more.
5. **Visualize:** You can view visualizations like confusion matrices, feature importances, and residual plots.
6. **Save Model:** After training, save your model as a `.pkl` file for future use.
7. **Save Processed Data:** Save the preprocessed data as a CSV file.



## Contributing
Feel free to fork the repository and submit pull requests. If you have suggestions or improvements, please open an issue or a pull request.


