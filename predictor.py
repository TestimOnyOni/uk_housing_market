import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb


# predictor.py
class XGBPredictor:
    def __init__(self, model, feature_order, encoders, stats):
        self.model = model
        self.feature_order = feature_order
        self.encoders = encoders
        self.stats = stats

    def predict(self, feats: dict, encoders=None, stats=None) -> float:
        encoders = encoders or self.encoders
        stats = stats or self.stats

        vec = []
        for col in self.feature_order:
            val = feats.get(col)
            if val is None:
                val = stats[col]
            # if col in encoders:
            #     val = encoders[col].transform([val])[0]
            # vec.append(val)
            if col in encoders:
                try:
                    val = encoders[col].transform([val])[0]
                except Exception:
                    val = stats[col] # fallback: use stats if encoding fails
            vec.append(float(val))


        # arr = np.array([vec])
        arr = np.array([vec], dtype=float)   # force numeric array
        dmatrix = xgb.DMatrix(arr, feature_names=self.feature_order)
        return float(self.model.predict(dmatrix)[0])


    @classmethod
    def train(
        cls,
        df,
        target_col,
        feature_order,
        encoders,
        stats,
        params=None,
        num_boost_round=300,
    ):
        X, y = [], []
        for _, row in df.iterrows():
            vec = []
            for col in feature_order:
                val = row[col]
                if pd.isna(val):
                    val = stats[col]
                if col in encoders:
                    try:
                        val = encoders[col].transform([val])[0]
                    except Exception:
                        val = stats[col]
                vec.append(float(val))   # ✅ enforce float
            X.append(vec)
            y.append(float(row[target_col]))   # ✅ enforce float target

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_order)
        model = xgb.train(
            params or {"objective": "reg:squarederror"},
            dtrain,
            num_boost_round=num_boost_round,
        )

        return cls(model, feature_order, encoders, stats)



    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_model(os.path.join(path, "xgb_model.json"))
        with open(os.path.join(path, "feature_order.pkl"), "wb") as f:
            pickle.dump(self.feature_order, f)

    @classmethod
    def load(cls, path: str):
        model = xgb.Booster()
        model.load_model(os.path.join(path, "xgb_model.json"))

        with open(os.path.join(path, "feature_order.pkl"), "rb") as f:
            feature_order = pickle.load(f)
        with open(os.path.join(path, "encoders.pkl"), "rb") as f:
            encoders = pickle.load(f)
        with open(os.path.join(path, "stats.pkl"), "rb") as f:
            stats = pickle.load(f)

        return cls(model, feature_order, encoders, stats)
