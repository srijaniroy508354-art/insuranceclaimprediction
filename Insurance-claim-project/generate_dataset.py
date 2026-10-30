"""
Synthetic insurance-claim dataset generator (SIMULATED DATA - not real customers).

Claims are drawn from a logistic model whose rules are written out below,
so the dataset has a documented, learnable relationship between customer
details and claims. Because the relationship is defined here, model scores
on this data show how well the pipeline recovers these rules - they are NOT
evidence about real-world insurance behaviour.

Run:  python generate_dataset.py
Output: insurance_claim_prediction_dataset.xlsx  (same columns as the original)
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "insurance_claim_prediction_dataset.xlsx"

N = 5000
SEED = 42
rng = np.random.default_rng(SEED)

age = rng.integers(18, 76, N)
vehicle_age = np.round(rng.uniform(0, 18, N), 1)
mileage = rng.integers(2000, 40000, N)
vehicle_type = rng.choice(["Hatchback", "Sedan", "SUV"], N, p=[0.4, 0.35, 0.25])
region = rng.choice(["Urban", "Suburban", "Rural"], N, p=[0.4, 0.35, 0.25])

# past claims: more likely for young drivers and high mileage
lam = 0.35 + 0.6 * (age < 25) + 0.5 * (mileage > 28000)
past_claims = np.minimum(rng.poisson(lam), 4)

# ---- RULES: log-odds of a claim ------------------------------------------
logit = (
    -2.9
    + 1.25 * past_claims                    # each earlier claim adds risk
    + 1.4 * (mileage - 20000) / 10000       # high annual mileage
    + 0.16 * (vehicle_age - 9)              # older vehicles
    + 1.5 * (age < 25)                      # young drivers
    + 0.7 * (age > 65)                      # senior drivers
    + 0.8 * (region == "Urban")             # dense traffic
    - 0.4 * (region == "Rural")
    + 0.4 * (vehicle_type == "Sedan")
)
noisy = logit + rng.normal(0, 1.0, N)       # randomness no model can explain
is_claim = (rng.random(N) < 1 / (1 + np.exp(-noisy))).astype(int)

# premium loosely follows the customer's visible risk profile (not the noise
# or the outcome, so it carries no hidden answer), plus random variation
risk_rank = pd.Series(logit).rank(pct=True).values
premium = (8000 + 40000 * (0.5 * risk_rank + 0.5 * rng.random(N))).round(-2).astype(int)

df = pd.DataFrame({
    "customer_id": [f"C{i:05d}" for i in range(1, N + 1)],
    "age": age,
    "vehicle_age_years": vehicle_age,
    "annual_mileage_km": mileage,
    "past_claims_count": past_claims,
    "annual_premium_inr": premium,
    "vehicle_type": vehicle_type,
    "region": region,
    "is_claim": is_claim,
})
df.to_excel(OUTPUT_PATH, index=False)
print("rows:", len(df), "| claim rate:", round(df.is_claim.mean() * 100, 1), "%")
