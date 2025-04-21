# MLFlowX: Automated Machine Learning Platform 🤖📊

MLFlowX is a comprehensive, user-friendly platform that automates the machine learning workflow from data preprocessing to model deployment. Built with Streamlit and powered by various ML libraries, it provides an intuitive interface for both beginners and experts in data science.

## 🌟 Features

### 1. Data Management
- **Data Upload**: Support for multiple file formats (CSV, Excel, JSON, Parquet)
- **Data Exploration**: Interactive data visualization and statistical analysis
- **Data Export**: Export processed data in various formats

### 2. Data Preprocessing
- **Missing Value Handling**: Multiple imputation strategies
- **Outlier Detection**: IQR and Z-Score based detection
- **Feature Scaling**: Standard, MinMax, and Robust scaling options

### 3. Feature Engineering
- **Feature Transformation**: Log, Square Root, and Polynomial transformations
- **Dimensionality Reduction**: PCA and Feature Importance analysis
- **Feature Selection**: Advanced feature selection techniques

### 4. Model Training
- **Multiple Algorithms**: Support for various ML models
  - Classification: Logistic Regression, Random Forest, SVM, Gradient Boosting
  - Regression: Linear Regression, Random Forest, SVR, Gradient Boosting
  - Clustering: K-Means, DBSCAN
- **Hyperparameter Tuning**: Grid search and cross-validation
- **Model Comparison**: Performance metrics and visualization

### 5. Model Evaluation
- **Performance Metrics**: Accuracy, MSE, R2 Score, etc.
- **Model Explainability**: SHAP values, Feature Importance, Partial Dependence
- **Visualization**: Interactive plots and charts

### 6. Deployment
- **Model Export**: Save trained models in various formats
- **API Generation**: Automatic FastAPI code generation
- **Model Versioning**: Track and manage different model versions

### 7. Natural Language Processing
- **Text Analysis**: Classification, Sentiment Analysis
- **Text Processing**: Summarization, NER, Topic Modeling
- **Interactive Chatbot**: Powered by Gemini AI

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mlflowx.git
cd mlflowx
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

### Running the Application
```bash
streamlit run app.py
```

## 📊 Usage Guide

1. **Data Upload**
   - Navigate to the "Data Upload" section
   - Upload your dataset
   - Explore data statistics and visualizations

2. **Data Preprocessing**
   - Handle missing values
   - Detect and manage outliers
   - Scale features as needed

3. **Feature Engineering**
   - Transform features
   - Reduce dimensionality
   - Select important features

4. **Model Training**
   - Choose problem type (Classification/Regression/Clustering)
   - Select appropriate model
   - Configure hyperparameters
   - Train and evaluate model

5. **Model Deployment**
   - Export trained model
   - Generate API code
   - Deploy model

6. **NLP Features**
   - Use text analysis tools
   - Interact with the chatbot

## 🤖 Interactive Chatbot

The platform includes an intelligent chatbot powered by Gemini AI that can:
- Answer questions about the platform
- Provide guidance on ML concepts
- Help with data analysis
- Generate code snippets

### Chatbot Features
- Configurable response temperature
- Conversation history management
- Multiple export formats
- Typing indicators
- Error handling and recovery

## 📝 Export & Reporting

Generate comprehensive reports including:
- Dataset summary
- Statistical analysis
- Feature importance
- Model performance metrics
- Visualization exports

## 🔧 Configuration

Customize your experience through:
- Chat settings
- Model parameters
- Visualization options
- Export formats

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Streamlit for the web framework
- scikit-learn for ML algorithms
- Gemini AI for the chatbot capabilities
- Plotly for interactive visualizations

## 📞 Support

For support, please open an issue in the GitHub repository or contact the development team.

---

Made with ❤️ by the MLFlowX Team
