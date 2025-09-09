import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# --------------------------
# Knowledge Base Context
# --------------------------
class KBContext:
    def __init__(self, df: pd.DataFrame, encoders: Dict[str, LabelEncoder], stats: Dict[str, Any], feature_order: List[str]):
        self.df = df
        self.encoders = encoders
        self.stats = stats
        self.feature_order = feature_order

    @classmethod
    def load(cls, artifact_dir: str) -> "KBContext":
        df = pd.read_csv(os.path.join(artifact_dir, "uk_housing_market_data.csv"))

        with open(os.path.join(artifact_dir, "encoders.pkl"), "rb") as f:
            encoders = pickle.load(f)
        with open(os.path.join(artifact_dir, "stats.pkl"), "rb") as f:
            stats = pickle.load(f)
        with open(os.path.join(artifact_dir, "feature_order.pkl"), "rb") as f:
            feature_order = pickle.load(f)

        return cls(df, encoders, stats, feature_order)


# --------------------------
# Predictor Wrapper
# --------------------------
class XGBPredictor:
    def __init__(self, model=None, feature_order: List[str] = None):
        self.model = model
        self.feature_order = feature_order

    # ---------------------------
    # Training
    # ---------------------------
    @classmethod
    def train(
        cls,
        train_df,
        target_col: str,
        feature_order: List[str],
        encoders: Dict[str, Any],
        stats: Dict[str, Any],
        params: Dict[str, Any] = None,
        num_boost_round: int = 200,
    ):
        """
        Train an XGBoost model.
        """
        if params is None:
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "seed": 42,
            }

        # Encode training data into numeric matrix
        X = []
        for _, row in train_df.iterrows():
            vec = []
            for col in feature_order:
                val = row[col]
                if col in encoders:
                    val = encoders[col].transform([val])[0]
                vec.append(val)
            X.append(vec)

        X = np.array(X)
        y = train_df[target_col].values

        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_order)
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

        return cls(model=model, feature_order=feature_order)

    # ---------------------------
    # Prediction
    # ---------------------------
    def predict(self, feats: Dict[str, Any], encoders, stats) -> float:
        """
        Predict target value for raw feature dict.
        """
        vec = []
        for col in self.feature_order:
            val = feats.get(col)

            # Fill missing values from dataset stats
            if val is None:
                val = stats[col]

            # Encode categorical
            if col in encoders:
                val = encoders[col].transform([val])[0]

            vec.append(val)

        arr = np.array([vec])
        dmatrix = xgb.DMatrix(arr, feature_names=self.feature_order)
        return float(self.model.predict(dmatrix)[0])

    # ---------------------------
    # Save + Load
    # ---------------------------
    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_model(os.path.join(out_dir, "xgb_model.json"))

        with open(os.path.join(out_dir, "feature_order.pkl"), "wb") as f:
            pickle.dump(self.feature_order, f)

    @classmethod
    def load(cls, in_dir: str):
        model = xgb.Booster()
        model.load_model(os.path.join(in_dir, "xgb_model.json"))

        with open(os.path.join(in_dir, "feature_order.pkl"), "rb") as f:
            feature_order = pickle.load(f)

        return cls(model=model, feature_order=feature_order)


# --------------------------
# Query Handler
# --------------------------
def handle_query(q: str, kb: KBContext, predictor: XGBPredictor, llm: Optional[object] = None) -> str:
    """
    q: user question in natural language
    kb: knowledge base context (data + encoders + stats)
    predictor: trained model wrapper
    llm: optional LLM for better NLP parsing (not required here)
    """
    # --- very basic feature extraction ---
    feats: Dict[str, Any] = {}

    # You can make this smarter; for now it's just a placeholder
    if "shoreditch" in q.lower():
        feats["Location_City"] = "London"
        feats["Location_District"] = "Shoreditch"
    if "townhouse" in q.lower():
        feats["Property_Type"] = "Townhouse"
    if "4-bed" in q.lower() or "4 bed" in q.lower():
        feats["Bedrooms"] = 4
    if "2008" in q:
        feats["Year_Built"] = 2008

    # Run prediction
    if not isinstance(feats, dict):
        try:
            feats = feats.model_dump()  # Pydantic v2
        except AttributeError:
            feats = feats.__dict__       # fallback
    pred_val = predictor.predict(feats, kb.encoders, kb.stats)

    return f"Predicted (Price_Boxcox scale): {pred_val:.2f}"