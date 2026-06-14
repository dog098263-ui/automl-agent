import os
import json
import pickle
import numpy as np
import pandas as pd
import httpx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"XGBoost library check failed: {e}. Using Scikit-Learn GradientBoosting as fallback.")
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/generate"

def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Detects if the target column is regression, classification, or forecasting.
    """
    y = df[target_col].dropna()
    unique_count = y.nunique()
    dtype = str(y.dtype)
    
    is_numeric = "int" in dtype or "float" in dtype
    
    # Check if there is a datetime column for possible forecasting
    datetime_cols = []
    for col in df.columns:
        if col != target_col:
            try:
                parsed = pd.to_datetime(df[col].dropna().head(10), errors='raise')
                if not parsed.isnull().all():
                    datetime_cols.append(col)
                    continue
            except:
                pass
            if "date" in col.lower() or "time" in col.lower():
                datetime_cols.append(col)
                
    if datetime_cols and is_numeric:
        return "forecasting"
        
    if not is_numeric or unique_count < 15:
        return "classification"
        
    return "regression"

def _find_datetime_column(df: pd.DataFrame, target_col: str) -> Optional[str]:
    for col in df.columns:
        if col != target_col:
            try:
                parsed = pd.to_datetime(df[col].dropna().head(10), errors='raise')
                if not parsed.isnull().all():
                    return col
            except:
                pass
            if "date" in col.lower() or "time" in col.lower():
                return col
    return None

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import TargetEncoder

class TextTfidfTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name="text", max_features=50):
        self.column_name = column_name
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            texts = X.iloc[:, 0].fillna("").astype(str).tolist()
        else:
            texts = pd.Series(X[:, 0]).fillna("").astype(str).tolist()
        self.vectorizer.fit(texts)
        return self
        
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = X.iloc[:, 0].fillna("").astype(str).tolist()
        else:
            texts = pd.Series(X[:, 0]).fillna("").astype(str).tolist()
        return self.vectorizer.transform(texts).toarray()

    def get_feature_names_out(self, input_features=None):
        return [f"{self.column_name}_{f}" for f in self.vectorizer.get_feature_names_out()]

def preprocess_and_split(df: pd.DataFrame, target_col: str, task_type: str):
    """
    Splits into features X and target y, builds preprocessing pipeline.
    """
    df_clean = df.dropna(subset=[target_col]).copy()
    
    # Exclude ID or high cardinality columns that shouldn't be features
    cols_to_drop = [target_col]
    for col in df_clean.columns:
        if col != target_col:
            if df_clean[col].nunique() == len(df_clean) and df_clean[col].dtype == "object":
                cols_to_drop.append(col)
            elif "id" in col.lower() and df_clean[col].nunique() > len(df_clean) * 0.8:
                cols_to_drop.append(col)
                
    X = df_clean.drop(columns=cols_to_drop)
    y = df_clean[target_col]
    
    # Label encode y for classification if object
    if task_type == "classification" and y.dtype == "object":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        label_mapping = {int(i): str(c) for i, c in enumerate(le.classes_)}
    else:
        label_mapping = None

    # Identify numeric vs categorical columns (use np.number to capture all types)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    # Preprocessors: Upgraded to KNNImputer for numerical features
    num_transformer = Pipeline(steps=[
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler())
    ])
    
    # Categorize object features for target encoding, tf-idf, or one-hot encoding
    te_cols = [c for c in cat_cols if "company" in c.lower()]
    tfidf_cols = [c for c in cat_cols if "description" in c.lower() or "text" in c.lower()]
    other_cat_cols = [c for c in cat_cols if c not in te_cols and c not in tfidf_cols]
    
    transformers = [
        ("num", num_transformer, num_cols)
    ]
    
    if te_cols:
        from sklearn.model_selection import KFold
        train_size_estimate = max(1, int(len(X) * 0.8))
        cv_val = min(5, max(2, train_size_estimate))
        cv_obj = KFold(n_splits=cv_val, shuffle=True, random_state=42)
        te_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("te", TargetEncoder(smooth="auto", cv=cv_obj))
        ])
        transformers.append(("te", te_transformer, te_cols))
        
    for idx, col in enumerate(tfidf_cols):
        transformers.append((f"tfidf_{idx}", TextTfidfTransformer(column_name=col, max_features=50), [col]))
        
    if other_cat_cols:
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_transformer, other_cat_cols))
        
    preprocessor = ColumnTransformer(transformers=transformers)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols, label_mapping

async def train_best_model(df: pd.DataFrame, target_col: str, project_dir: str) -> dict:
    """
    Trains classification/regression/forecasting models, selects the best, saves it,
    and returns metrics and feature importances.
    """
    task_type = detect_task_type(df, target_col)
    
    date_col = None
    max_date = None
    df_processed = df.copy()
    
    # Time-Series Forecasting Preparation
    if task_type == "forecasting":
        date_col = _find_datetime_column(df_processed, target_col)
        if date_col:
            df_processed[date_col] = pd.to_datetime(df_processed[date_col], errors='coerce')
            df_processed = df_processed.sort_values(by=date_col).reset_index(drop=True)
            max_date_val = df_processed[date_col].max()
            if not pd.isnull(max_date_val):
                max_date = str(max_date_val.date())
            
            # Linear Interpolation for sequence targets & numerical variables
            df_processed[target_col] = df_processed[target_col].interpolate(method='linear').ffill().bfill()
            
            # Extract temporal features
            dates = df_processed[date_col].ffill().bfill()
            df_processed["year"] = dates.dt.year
            df_processed["month"] = dates.dt.month
            df_processed["day"] = dates.dt.day
            df_processed["dayofweek"] = dates.dt.dayofweek
            df_processed["dayofyear"] = dates.dt.dayofyear
            df_processed["quarter"] = dates.dt.quarter
            
            df_processed = df_processed.drop(columns=[date_col])
            
    X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols, label_mapping = preprocess_and_split(df_processed, target_col, task_type)
    
    X_train_trans = preprocessor.fit_transform(X_train, y_train)
    X_test_trans = preprocessor.transform(X_test)
    
    try:
        raw_feature_names = preprocessor.get_feature_names_out().tolist()
        feature_names = []
        for name in raw_feature_names:
            if "__" in name:
                feature_names.append(name.split("__", 1)[1])
            else:
                feature_names.append(name)
    except Exception as e:
        print(f"Failed to get feature names dynamically: {e}")
        encoded_cat_cols = []
        if cat_cols:
            try:
                ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
                encoded_cat_cols = ohe.get_feature_names_out(cat_cols).tolist()
            except:
                encoded_cat_cols = [f"cat_{i}" for i in range(X_train_trans.shape[1] - len(num_cols))]
        feature_names = num_cols + encoded_cat_cols
    
    best_model = None
    best_score = -1.0
    best_algo = ""
    metrics = {}
    
    if task_type == "classification":
        # Candidate 1: Random Forest
        rf = RandomForestClassifier(random_state=42, n_estimators=100)
        rf.fit(X_train_trans, y_train)
        rf_preds = rf.predict(X_test_trans)
        rf_acc = accuracy_score(y_test, rf_preds)
        
        # Candidate 2: XGBoost
        xgb_trained = False
        xgb_acc = -1.0
        if XGBOOST_AVAILABLE:
            try:
                xgb = XGBClassifier(random_state=42, n_estimators=100, eval_metric="logloss")
                xgb.fit(X_train_trans, y_train)
                xgb_preds = xgb.predict(X_test_trans)
                xgb_acc = accuracy_score(y_test, xgb_preds)
                xgb_trained = True
            except Exception as e:
                print(f"XGBoost training failed, using GradientBoosting fallback: {e}")
        
        if not xgb_trained:
            from sklearn.ensemble import GradientBoostingClassifier
            xgb = GradientBoostingClassifier(random_state=42, n_estimators=100)
            xgb.fit(X_train_trans, y_train)
            xgb_preds = xgb.predict(X_test_trans)
            xgb_acc = accuracy_score(y_test, xgb_preds)
        
        if rf_acc >= xgb_acc:
            best_model = rf
            best_score = rf_acc
            best_algo = "Random Forest Classifier"
        else:
            best_model = xgb
            best_score = xgb_acc
            best_algo = "XGBoost Classifier" if xgb_trained else "Gradient Boosting Classifier"
            
        metrics = {
            "accuracy": float(best_score),
            "f1_score": float(f1_score(y_test, best_model.predict(X_test_trans), average="weighted"))
        }
        
    else: # regression or forecasting
        # Candidate 1: Random Forest
        rf = RandomForestRegressor(random_state=42, n_estimators=100)
        rf.fit(X_train_trans, y_train)
        rf_preds = rf.predict(X_test_trans)
        rf_r2 = r2_score(y_test, rf_preds)
        
        # Candidate 2: XGBoost
        xgb_trained = False
        xgb_r2 = -1.0
        if XGBOOST_AVAILABLE:
            try:
                xgb = XGBRegressor(random_state=42, n_estimators=100)
                xgb.fit(X_train_trans, y_train)
                xgb_preds = xgb.predict(X_test_trans)
                xgb_r2 = r2_score(y_test, xgb_preds)
                xgb_trained = True
            except Exception as e:
                print(f"XGBoost training failed, using GradientBoosting fallback: {e}")
                
        if not xgb_trained:
            from sklearn.ensemble import GradientBoostingRegressor
            xgb = GradientBoostingRegressor(random_state=42, n_estimators=100)
            xgb.fit(X_train_trans, y_train)
            xgb_preds = xgb.predict(X_test_trans)
            xgb_r2 = r2_score(y_test, xgb_preds)
            
        if rf_r2 >= xgb_r2:
            best_model = rf
            best_score = rf_r2
            best_algo = "Random Forest Regressor"
        else:
            best_model = xgb
            best_score = xgb_r2
            best_algo = "XGBoost Regressor" if xgb_trained else "Gradient Boosting Regressor"
            
        test_preds = best_model.predict(X_test_trans)
        metrics = {
            "r2_score": float(best_score),
            "rmse": float(np.sqrt(mean_squared_error(y_test, test_preds)))
        }

    # Save complete Pipeline
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", best_model)
    ])
    
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "model.pkl")
    
    # Save original num_cols and cat_cols along with forecasting details for re-alignment in predictions
    with open(model_path, "wb") as f:
        pickle.dump({
            "pipeline": full_pipeline,
            "task_type": task_type,
            "algorithm": best_algo,
            "label_mapping": label_mapping,
            "feature_names": feature_names,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "date_col": date_col,
            "max_date": max_date
        }, f)

    # Feature Importance mapping
    importances = []
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_.tolist()
    
    importance_map = {}
    for name, imp in zip(feature_names, importances):
        original_name = name
        for orig_cat in cat_cols:
            if name.startswith(orig_cat + "_"):
                original_name = orig_cat
                break
        importance_map[original_name] = importance_map.get(original_name, 0.0) + imp
        
    sorted_importance = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)
    sorted_importance_dict = {k: float(v) for k, v in sorted_importance[:10]}

    return {
        "task_type": task_type,
        "algorithm": best_algo,
        "metrics": metrics,
        "feature_importance": sorted_importance_dict,
        "model_path": model_path,
        "label_mapping": label_mapping,
        "target_column": target_col,
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }

async def generate_explanation(task_type: str, algorithm: str, metrics: dict, feature_importance: dict) -> str:
    """
    Calls Ollama to explain the model results in plain English.
    """
    from cleaner import get_available_ollama_model
    
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    top_features = ", ".join([f"{k} ({v*100:.1f}%)" for k, v in list(feature_importance.items())[:3]])
    
    prompt = f"""
You are an expert AI Communicator and Data Scientist. Explain the results of a machine learning model to a completely non-technical business owner.
Provide a highly detailed, professional, and easily digestible breakdown in Markdown format.

Model Details:
- Task Type: {task_type} (e.g. classification means predicting categories/classes, regression/forecasting means predicting numerical trends/values)
- Algorithm Used: {algorithm}
- Model Evaluation Metrics: {metrics_str}
- Key Decision Drivers (Feature Importance): {top_features}

Please write a comprehensive explanation structured into the following sections:
1. **Model Performance & Confidence**: Explain the evaluation scores in detail, what they signify, how reliable the model's predictions are, and what the confidence level is.
2. **Key Decision Drivers**: For each of the top features, provide a real-world, intuitive explanation of why the feature affects predictions and its importance.
3. **Internal Decision Logic**: Detail how the model (Random Forest or XGBoost/Gradient Boosting) works internally to make decisions based on splits, trees, and feature combinations.
4. **Business Implications & Next Steps**: How the business owner should apply this model to improve operations or make decisions.

Write in a professional, clear, and encouraging tone. Do not limit the explanation length; be thorough and detailed. Avoid dry technical jargon where possible, but explain the mechanics clearly.
"""

    model_name = await get_available_ollama_model()

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                OLLAMA_URL,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=15.0
            )
            if resp.status_code == 200:
                result_json = resp.json()
                return result_json.get("response", "").strip()
    except Exception as e:
        print(f"Ollama explanation request failed: {e}")

    # Detailed Fallback explanation matching the new structured prompt
    explanation = f"## 📊 AutoML Model Evaluation Report\n\n"
    explanation += f"We have successfully trained a **{algorithm}** to perform **{task_type}**.\n\n"
    
    explanation += f"### 1. Model Performance & Confidence\n"
    if task_type == "classification":
        acc = metrics.get("accuracy", 0.0) * 100
        f1 = metrics.get("f1_score", 0.0) * 100
        explanation += f"- **Accuracy Score**: The model is **{acc:.1f}%** accurate at making predictions on unseen test data.\n"
        explanation += f"- **F1-Score**: The weighted F1-Score is **{f1:.1f}%**.\n\n"
        explanation += f"This means that in approximately {acc:.0f} out of 100 cases, the model's classification prediction is correct. The F1-score confirms that the model's performance is balanced and reliable across different class categories, providing high confidence for deployment.\n"
    else:
        r2 = metrics.get("r2_score", 0.0) * 100
        rmse = metrics.get("rmse", 0.0)
        explanation += f"- **R-Squared (R²) Score**: The model explains **{r2:.1f}%** of the variation in the target data.\n"
        explanation += f"- **RMSE (Root Mean Squared Error)**: The standard deviation of residual errors is **{rmse:.4f}**.\n\n"
        explanation += f"The R² score indicates a strong capability to predict trends by accounting for {r2:.1f}% of the variations. The RMSE tells us the average deviation of predictions from actual target values, helping us understand the standard margins of error.\n"
        
    explanation += f"\n### 2. Key Decision Drivers (Feature Importance)\n"
    explanation += "The model weights various features to determine the final decision. The top three factors driving predictions are:\n"
    for rank, (feat, val) in enumerate(list(feature_importance.items())[:3], 1):
        explanation += f"{rank}. **{feat}** (accounts for **{val*100:.1f}%** of decision weight). This feature has a major relative impact on predictions, playing a critical role in the model's tree splits.\n"
        
    explanation += f"\n### 3. Internal Decision Logic\n"
    if "Random Forest" in algorithm:
        explanation += "The **Random Forest** algorithm works by training an ensemble of independent decision trees. Each tree analyzes random subsets of data and features to make a prediction. The final outcome is decided by a majority vote of all trees. This collaborative voting reduces individual errors, prevents overfitting, and handles complex correlations.\n"
    else:
        explanation += "The **Gradient Boosting / XGBoost** algorithm works sequentially. It trains an initial weak decision tree and then builds subsequent trees to focus on and correct the specific errors made by previous models. This continuous boosting cycle optimizes the overall mathematical error, yielding a highly accurate, finely tuned predictive system.\n"
        
    explanation += f"\n### 4. Business Implications & Next Steps\n"
    explanation += "You can now use this trained model to generate predictions on new inputs. Focus on monitoring the key decision drivers identified above, as adjusting these variables will yield the highest impact on your target outcomes.\n"
    
    return explanation

