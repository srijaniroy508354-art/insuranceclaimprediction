# 🛡️ InsurePredict AI
## Insurance Claim Prediction Using Logistic Regression

### 📌 Project Overview

InsurePredict AI is a machine learning-based web application
that predicts whether a customer is likely to make an
insurance claim based on customer and vehicle information.

The project uses Logistic Regression for binary classification
and Flask to provide an interactive web interface.

### 🎯 Objectives

- Predict insurance claim outcomes using machine learning.
- Analyze customer and vehicle-related information.
- Build an interactive web application.
- Evaluate model performance using classification metrics.

### 🛠️ Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- HTML
- CSS
- Microsoft Excel
- Visual Studio Code

### 🤖 Machine Learning Algorithm

Logistic Regression

The model predicts two possible outcomes:

- 0: No Claim
- 1: Claim

### 📊 Dataset

The project uses an Excel dataset containing 5,000 **simulated** records,
created by `generate_dataset.py`. Claims are generated from documented
rules (mileage, past claims, vehicle age, driver age, region, vehicle type)
plus random noise. The original 500-record file is kept as
`insurance_claim_prediction_dataset_original.xlsx`.

Features include:

- Customer Age
- Vehicle Age
- Annual Mileage
- Previous Claims
- Annual Premium
- Vehicle Type
- Region

Target variable: `is_claim`

### 🌐 Website Features

1. Insurance Claim Prediction Form
2. Claim Probability Display
3. Model Performance Dashboard
4. Accuracy, Precision, Recall, and F1 Score
5. Confusion Matrix

### 📈 Model Evaluation

The model uses balanced class weights and a decision threshold of 0.55
(`THRESHOLD` in `metrics.py`). It is evaluated on a 20% held-out test set
(1,000 rows) using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### ▶️ How to Run the Project

1. Open the project folder in VS Code.

2. Create a virtual environment:

   python -m venv .venv

3. Activate the virtual environment on Windows:

   .venv\Scripts\activate

4. Install the required libraries:

   pip install flask pandas openpyxl scikit-learn

5. Run the application:

   python app.py

6. Open the local website:

   http://127.0.0.1:5000

### 📁 Project Structure

Insurance-claim-project/
│
├── app.py
├── evaluate.py
├── metrics.py
├── generate_dataset.py
├── insurance_claim_prediction_dataset_original.xlsx
├── insurance_claim_prediction_dataset.xlsx
├── README.md
│
└── templates/
    ├── index.html
    └── performance.html

### ⚠️ Limitations

The dataset is simulated, so the scores show how well the pipeline
recovers the rules written in `generate_dataset.py`. They are NOT evidence
of real-world insurance behaviour. Predictions must not be used for real
insurance decisions without validation on real customer data.

### 🎓 Project Type

Machine Learning Mini Project