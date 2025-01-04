# Automated Machine Learning Dashboard

## Overview
This web application is a user-friendly tool that allows users to upload a dataset, preprocess the data, select machine learning models, train the models, evaluate their performance, and visualize results. It supports a variety of algorithms for classification and regression tasks, including Logistic Regression, Linear Regression, Random Forest, Support Vector Machines (SVM), Decision Trees, K-Nearest Neighbors (KNN), and Gradient Boosting.

## Features
- **Login System:** Secure login for users with different access levels.
- **Dataset Upload:** Users can upload datasets in CSV, Excel, JSON, or Parquet formats.
- **Data Preprocessing:** Automatic handling of missing values, categorical encoding, and feature scaling.
- **Model Selection:** Choose from various machine learning algorithms (Logistic Regression, Linear Regression, Random Forest, SVM, Decision Trees, KNN, Gradient Boosting).
- **Model Training:** Train models on the uploaded dataset and evaluate performance with metrics like accuracy, MSE, and R2 score.
- **Evaluation Metrics:** View confusion matrix (for classification models), feature importance (for Random Forest and Gradient Boosting), and residual plots (for regression models).
- **Model Saving:** Save trained models as `.pkl` files for future use.
- **Data Saving:** Save processed data as a CSV file for further use or analysis.

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


