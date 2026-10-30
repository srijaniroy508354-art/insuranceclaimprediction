
from pathlib import Path

from flask import Flask, render_template, request
from metrics import accuracy, precision, recall, f1, cm, THRESHOLD
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "insurance_claim_prediction_dataset.xlsx"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

# Load the Excel dataset
df = pd.read_excel(DATASET_PATH)

# Separate features and target
X = df.drop(columns=["is_claim", "customer_id"])
y = df["is_claim"]

# Identify column types
num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

# Prepare numerical columns
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Prepare categorical columns
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Build and train the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None

    if request.method == "POST":
        customer = pd.DataFrame([{
            "age": int(request.form["age"]),
            "vehicle_age_years": float(
                request.form["vehicle_age_years"]
            ),
            "annual_mileage_km": int(
                request.form["annual_mileage_km"]
            ),
            "past_claims_count": int(
                request.form["past_claims_count"]
            ),
            "annual_premium_inr": int(
                request.form["annual_premium_inr"]
            ),
            "vehicle_type": request.form["vehicle_type"],
            "region": request.form["region"]
        }])

        proba = model.predict_proba(customer)[0][1]
        prediction = 1 if proba >= THRESHOLD else 0
        probability = round(proba * 100, 2)

        if prediction == 1:
            result = "Claim Predicted"
        else:
            result = "No Claim Predicted"

    return render_template(
        "index.html",
        result=result,
        probability=probability
    )

@app.route("/performance")
def performance():
    return render_template(
        "performance.html",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        cm=cm
    )
if __name__ == "__main__":
    app.run(debug=True)