# MLFlowX: Automating Machine Learning at Your Fingertips 🤖📊

MLFlowX is a user-friendly web application built with **Streamlit** that simplifies the process of data preprocessing, model training, evaluation, and deployment. Whether you're a beginner or an expert, MLFlowX provides an intuitive interface to explore datasets, preprocess data, train machine learning models, and deploy them with ease.

---

## Features 🌟

1. **Data Upload & Exploration**:
   - Upload datasets in various formats (CSV, Excel, JSON, Parquet).
   - Preview and explore data with summary statistics and missing value analysis.

2. **Data Preprocessing**:
   - Handle missing values using methods like Mean, Median, Mode, KNN Imputer, and more.
   - Detect and handle outliers using IQR or Z-Score.
   - Apply feature scaling techniques like Standard Scaler, MinMax Scaler, and Robust Scaler.

3. **Feature Engineering**:
   - Perform feature transformations like Log Transform, Square Root, and Polynomial.
   - Reduce dimensionality using PCA or feature importance.

4. **Model Training**:
   - Train models for Classification, Regression, and Clustering tasks.
   - Supports models like Logistic Regression, Random Forest, SVM, Gradient Boosting, K-Means, and more.
   - Tune hyperparameters and evaluate models using cross-validation.

5. **Model Evaluation**:
   - Visualize model performance with metrics like Accuracy, MSE, R2 Score, and Silhouette Score.
   - Explain model predictions using SHAP, Feature Importance, and Partial Dependence Plots.

6. **Model Deployment**:
   - Export trained models as `.pkl` files.
   - Generate FastAPI code for deploying models as REST APIs.

---

## How to Use 🛠️

1. **Login**:
   - Use the default credentials (`admin:admin123` or `user:user123`) to log in.

2. **Data Upload**:
   - Navigate to the **Data Upload** page and upload your dataset.

3. **Data Preprocessing**:
   - Go to the **Data Preprocessing** page to handle missing values, outliers, and scaling.

4. **Feature Engineering**:
   - Visit the **Feature Engineering** page to transform features or reduce dimensionality.

5. **Model Training**:
   - Head to the **Model Training** page, select your target column, and train a model.

6. **Model Evaluation**:
   - Evaluate your model's performance and interpret predictions on the **Model Evaluation** page.

7. **Model Deployment**:
   - Export your trained model or generate FastAPI code for deployment on the **Deployment** page.

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

## Contributing
Feel free to fork the repository and submit pull requests. If you have suggestions or improvements, please open an issue or a pull request.


