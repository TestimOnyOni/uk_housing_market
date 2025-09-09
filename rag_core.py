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
        df = pd.read_csv(os.path.join(artifact_dir, "properties.csv"))

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
    def __init__(self, model: xgb.Booster, feature_order: List[str]):
        self.model = model
        self.feature_order = feature_order

    @classmethod
    def load(cls, artifact_dir: str) -> "XGBPredictor":
        with open(os.path.join(artifact_dir, "xgb_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(artifact_dir, "feature_order.pkl"), "rb") as f:
            feature_order = pickle.load(f)
        return cls(model, feature_order)

    def predict(self, feats: Dict[str, Any], encoders: Dict[str, LabelEncoder], stats: Dict[str, Any]) -> float:
        vec = []
        for col in self.feature_order:
            val = feats.get(col)
            if val is None:
                val = stats[col]
            if col in encoders:
                val = encoders[col].transform([val])[0]
            vec.append(val)

        dmatrix = xgb.DMatrix(np.array([vec]), feature_names=self.feature_order)
        return float(self.model.predict(dmatrix)[0])


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
    pred_val = predictor.predict(feats, kb.encoders, kb.stats)

    return f"Predicted (Price_Boxcox scale): {pred_val:.2f}"