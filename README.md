# Automated Machine Learning Dashboard

## Objectives
The main objective of this project is to provide an intuitive and automated machine learning dashboard that allows users to:
- Upload datasets in multiple formats.
- Preprocess data by handling missing values, encoding categorical features, and scaling numerical features.
- Train various machine learning models such as Logistic Regression, Linear Regression, Random Forest, and SVM.
- Evaluate model performance using key metrics.
- Visualize data and model results through interactive plots.
- Save and download processed data for further use.

## Features
- **User Authentication**: Secure login system with predefined credentials.
- **Dataset Upload**: Support for multiple file formats including CSV, Excel (XLSX), JSON, and Parquet.
- **Data Preprocessing**: Automatically handles missing values, encodes categorical features, and scales numerical features.
- **Model Training**: Choose from a variety of machine learning models (Logistic Regression, Linear Regression, Random Forest, SVM).
- **Model Evaluation**: Displays key performance metrics such as accuracy, MSE, R2, and classification reports.
- **Data Visualization**: Visualize data through scatter plots and histograms with Plotly.
- **Save Processed Data**: Option to download the preprocessed dataset for further analysis or use.

## Tech Stack
- **Frontend**: Streamlit for interactive dashboard development.
- **Backend**: Python 3.7+ for backend logic.
- **Machine Learning**: Scikit-learn for model training and evaluation.
- **Data Visualization**: Plotly for interactive plots.
- **Data Handling**: Pandas for data manipulation and preprocessing.
- **Authentication**: Streamlit session state for managing user login.
