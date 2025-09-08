# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import xgboost as xgb
from scipy.stats import boxcox

CSV_PATH = "data/uk_housing_market_data.csv"

FEATURE_ORDER = [
"Square_Footage", "Bedrooms", "Bathrooms", "Year_Built", "Build_Quality_Rating",
"Location_City", "Location_District", "Property_Type",
"Nearby_Amenities_Score", "Market_Trend_Index", "Agent_Commission_Percentage",
"Days_On_Market", "Transaction_Type", "Revenue_Activity", "Revenue_GBP_Monthly",
]

CATEGORICAL_COLS = [
"Location_City", "Location_District", "Property_Type",
"Transaction_Type", "Revenue_Activity",
]
NUMERIC_COLS = [c for c in FEATURE_ORDER if c not in CATEGORICAL_COLS]
TARGET_RAW = "Sale_Price_GBP"
TARGET = "Price_Boxcox"


def build_categorical_encoders(df: pd.DataFrame, categorical_cols: list) -> dict:
    encoders = {}
    for c in categorical_cols:
        cats = sorted(set(str(v) for v in df[c].dropna().unique()))
        encoders[c] = {v: i + 1 for i, v in enumerate(cats)} # 0 reserved for unknown
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df2 = df.copy()
    for c, mapping in encoders.items():
        df2[c] = df2[c].astype(str).map(mapping).fillna(0).astype(int)
    return df2


def main():
    df = pd.read_csv(CSV_PATH)

    # Ensure target > 0 for Box-Cox; shift if needed
    y = df[TARGET_RAW]
    min_y = float(y.min())
    shift = 1.0 - min_y if min_y <= 0 else 0.0
    y_pos = y + shift
    y_bc, lmbda = boxcox(y_pos)
    df[TARGET] = y_bc

    encoders = build_categorical_encoders(df, CATEGORICAL_COLS)
    df_enc = apply_encoders(df, encoders)
    
    # Numeric medians for imputation (kept in original scale)
    stats = {c: float(df[c].median()) for c in NUMERIC_COLS if c in df.columns}

    X = df_enc[FEATURE_ORDER]
    y = df_enc[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    
    model.fit(X_train, y_train)

    print(f"Validation R^2: {model.score(X_test, y_test):.4f}")

    # Save artifacts
    joblib.dump(model, "artifacts/model_xgb.pkl")
    joblib.dump(encoders, "artifacts/encoders.pkl")
    joblib.dump(stats, "artifacts/stats.pkl")
    joblib.dump(FEATURE_ORDER, "artifacts/feature_order.pkl")

    # Save Box-Cox lambda and shift for inverse transform later
    joblib.dump({"lambda": lmbda, "shift": shift}, "artifacts/boxcox_lambda.pkl")
    print("✅ Saved model + encoders + stats + feature order + boxcox params to artifacts/")


if __name__ == "__main__":
    main()