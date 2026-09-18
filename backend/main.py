import os
import json
import sqlite3
import pandas as pd
import numpy as np
import pickle
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional

# Load .env file if present (local dev convenience)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from database import get_db_connection, DB_PATH
from scraper import scrape_url
from cleaner import generate_cleaning_strategy, execute_cleaning
from ml_engine import train_best_model, generate_explanation

app = FastAPI(title="AutoML Agent API", version="1.0")

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths relative to THIS file, not CWD
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_THIS_DIR, "data", "users", "user_1")
os.makedirs(DATA_DIR, exist_ok=True)

class ProjectCreate(BaseModel):
    name: str

class ScrapeRequest(BaseModel):
    url: str

class CleanRequest(BaseModel):
    pass

class TrainRequest(BaseModel):
    target_column: Optional[str] = None  # If None, auto-infer from last column

class PredictRequest(BaseModel):
    input_data: Optional[dict] = None  # key-value pair of feature columns
    prompt_text: Optional[str] = None  # natural language description
    target_column: Optional[str] = None  # targeted variable to predict

@app.get("/api/ollama/status")
async def check_ollama_status():
    import httpx
    ollama_base = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    if not ollama_base:
        return {"status": "disabled", "models": []}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ollama_base}/api/tags", timeout=1.5)
            if resp.status_code == 200:
                models_info = resp.json()
                models = [m["name"] for m in models_info.get("models", [])]
                return {"status": "connected", "models": models}
    except Exception:
        pass
    return {"status": "disconnected", "models": []}

@app.get("/api/health")
def health():
    return {"status": "ok", "app": "AutoML Agent", "version": "1.0"}

@app.get("/api/projects")
def list_projects():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM projects ORDER BY created_at DESC")
        projects = [dict(row) for row in cursor.fetchall()]
        return projects
    finally:
        conn.close()

@app.post("/api/projects")
def create_project(proj: ProjectCreate):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO projects (name, status) VALUES (?, 'created')", (proj.name,))
        project_id = cursor.lastrowid
        conn.commit()
    finally:
        conn.close()

    # Create user folder
    proj_dir = os.path.join(DATA_DIR, f"project_{project_id}")
    os.makedirs(os.path.join(proj_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(proj_dir, "cleaned"), exist_ok=True)
    os.makedirs(os.path.join(proj_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(proj_dir, "predictions"), exist_ok=True)

    return {"id": project_id, "name": proj.name, "status": "created"}

@app.delete("/api/projects/{project_id}")
def delete_project(project_id: int):
    """Delete a project and all its associated data."""
    import shutil
    proj_dir = os.path.join(DATA_DIR, f"project_{project_id}")
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Project not found")
        cursor.execute("DELETE FROM audit_log WHERE project_id = ?", (project_id,))
        cursor.execute("DELETE FROM predictions WHERE project_id = ?", (project_id,))
        cursor.execute("DELETE FROM models WHERE project_id = ?", (project_id,))
        cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()
    finally:
        conn.close()

    if os.path.exists(proj_dir):
        shutil.rmtree(proj_dir)
    return {"message": f"Project {project_id} deleted successfully"}

@app.get("/api/projects/{project_id}")
def get_project(project_id: int):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")
        proj = dict(row)
        cursor.execute("SELECT * FROM models WHERE project_id = ? ORDER BY timestamp DESC", (project_id,))
        models_list = [dict(r) for r in cursor.fetchall()]
        cursor.execute("SELECT * FROM predictions WHERE project_id = ? ORDER BY timestamp DESC", (project_id,))
        predictions_list = [dict(r) for r in cursor.fetchall()]
        cursor.execute("SELECT * FROM audit_log WHERE project_id = ?", (project_id,))
        audit_logs = [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()

    # Parse feature importance JSON
    for model_info in models_list:
        if model_info.get("feature_importance"):
            model_info["feature_importance"] = json.loads(model_info["feature_importance"])
        target_name = model_info.get("target_column") or "default"
        proj_dir = os.path.join(DATA_DIR, f"project_{project_id}")
        meta_path = os.path.join(proj_dir, "models", f"model_meta_{target_name}.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(proj_dir, "models", "model_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta_data = json.load(f)
                    if not model_info.get("target_column"):
                        model_info["target_column"] = meta_data.get("target_column")
                    model_info["num_cols"] = meta_data.get("num_cols")
                    model_info["cat_cols"] = meta_data.get("cat_cols")
            except Exception:
                pass

    # Load dataset stats (done after DB is closed)
    stats = {}
    if proj.get("raw_file_path") and os.path.exists(proj["raw_file_path"]):
        df_raw = pd.read_csv(proj["raw_file_path"])
        stats["raw"] = {"rows": len(df_raw), "columns": list(df_raw.columns),
                        "null_cells": int(df_raw.isnull().sum().sum()),
                        "duplicates": int(df_raw.duplicated().sum())}
    if proj.get("cleaned_file_path") and os.path.exists(proj["cleaned_file_path"]):
        df_clean = pd.read_csv(proj["cleaned_file_path"])
        stats["cleaned"] = {"rows": len(df_clean), "columns": list(df_clean.columns),
                            "null_cells": int(df_clean.isnull().sum().sum()),
                            "duplicates": int(df_clean.duplicated().sum())}

    pred_info = predictions_list[0] if predictions_list else None
    return {
        "project": proj, "stats": stats,
        "model": models_list[0] if models_list else None,
        "models": models_list,
        "prediction": pred_info,
        "predictions": predictions_list,
        "audit_logs": audit_logs
    }


@app.post("/api/projects/{project_id}/upload")
async def upload_dataset(project_id: int, file: UploadFile = File(...)):
    proj_dir = os.path.join(DATA_DIR, f"project_{project_id}")
    raw_dir = os.path.join(proj_dir, "raw")

    # Validate project exists
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM projects WHERE id = ?", (project_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Project not found")
    finally:
        conn.close()

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".csv", ".tsv", ".xlsx", ".xls", ".json"]:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    dest_path = os.path.join(raw_dir, filename)
    with open(dest_path, "wb") as buffer:
        buffer.write(await file.read())

    raw_csv_path = os.path.join(raw_dir, "dataset.csv")
    try:
        if ext == ".csv":
            os.replace(dest_path, raw_csv_path)
        elif ext == ".tsv":
            df = pd.read_csv(dest_path, sep="\t"); df.to_csv(raw_csv_path, index=False)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(dest_path); df.to_csv(raw_csv_path, index=False)
        elif ext == ".json":
            df = pd.read_json(dest_path); df.to_csv(raw_csv_path, index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing file failed: {str(e)}")

    df = pd.read_csv(raw_csv_path)

    conn = get_db_connection()
    try:
        conn.execute(
            "UPDATE projects SET original_filename=?, raw_file_path=?, status='uploaded' WHERE id=?",
            (filename, raw_csv_path, project_id)
        )
        conn.commit()
    finally:
        conn.close()

    preview_data = df.head(50).replace({np.nan: None}).to_dict(orient="records")
    return {"message": "File uploaded and standardized", "rows": len(df),
            "columns": list(df.columns), "preview": preview_data}


@app.post("/api/projects/{project_id}/scrape")
async def scrape_dataset(project_id: int, req: ScrapeRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Project not found")
        
    proj_dir = os.path.join(DATA_DIR, f"project_{project_id}")
    raw_csv_path = os.path.join(proj_dir, "raw", "dataset.csv")
    
    res = await scrape_url(req.url, raw_csv_path)
    if not res.get("success"):
        conn.close()
        raise HTTPException(status_code=500, detail=res.get("error", "Scraping failed"))
        
    # Save project status
    cursor.execute("""
    UPDATE projects 
    SET original_filename = ?, raw_file_path = ?, status = 'uploaded' 
    WHERE id = ?
    """, (f"Scraped: {req.url[:30]}...", raw_csv_path, project_id))
    
    conn.commit()
    conn.close()
    
    df = pd.read_csv(raw_csv_path)
    preview_data = df.head(50).replace({np.nan: None}).to_dict(orient="records")
    
    return {
        "message": f"Successfully scraped URL via {res.get('method')}",
        "rows": len(df),
        "columns": list(df.columns),
        "preview": preview_data
    }

@app.post("/api/projects/{project_id}/clean")
async def clean_dataset(project_id: int):
    # ── Step 1: Read project info from DB then CLOSE connection ──────────────
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        proj_row = cursor.fetchone()
        if not proj_row:
            raise HTTPException(status_code=404, detail="Project not found")
        proj = dict(proj_row)
    finally:
        conn.close()  # ← close BEFORE any slow work

    if not proj["raw_file_path"] or not os.path.exists(proj["raw_file_path"]):
        raise HTTPException(status_code=400, detail="No dataset uploaded yet")

    df = pd.read_csv(proj["raw_file_path"])

    # ── Step 2: Slow work — Ollama + cleaning (DB is closed during this) ─────
    strategy = await generate_cleaning_strategy(df)
    cleaned_df, audit_logs = execute_cleaning(df, strategy)

    # ── Step 3: Save results to disk ─────────────────────────────────────────
    proj_dir = os.path.join(DATA_DIR, f"project_{project_id}")
    cleaned_csv_path = os.path.join(proj_dir, "cleaned", "dataset_cleaned.csv")
    cleaned_df.to_csv(cleaned_csv_path, index=False)
    audit_json_path = os.path.join(proj_dir, "cleaned", "audit_log.json")
    with open(audit_json_path, "w") as f:
        json.dump(audit_logs, f, indent=2)

    # ── Step 4: Fresh short DB write ─────────────────────────────────────────
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM audit_log WHERE project_id = ?", (project_id,))
        for log in audit_logs:
            cursor.execute(
                "INSERT INTO audit_log (project_id, column_name, operation, details) VALUES (?,?,?,?)",
                (project_id, log["column"], log["operation"], log["details"])
            )
        cursor.execute(
            "UPDATE projects SET cleaned_file_path=?, status='cleaned' WHERE id=?",
            (cleaned_csv_path, project_id)
        )
        conn.commit()
    finally:
        conn.close()

    preview_data = cleaned_df.head(50).replace({np.nan: None}).to_dict(orient="records")
    return {"message": "Data cleaned successfully", "strategy": strategy,
            "audit_logs": audit_logs, "preview": preview_data}


@app.post("/api/projects/{project_id}/train")
async def train_model_endpoint(project_id: int, req: TrainRequest):
    # ── Step 1: Read from DB then CLOSE immediately ───────────────────────────
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        proj_row = cursor.fetchone()
        if not proj_row:
            raise HTTPException(status_code=404, detail="Project not found")
        proj = dict(proj_row)
    finally:
        conn.close()  # ← close before any slow work

    if not proj["cleaned_file_path"] or not os.path.exists(proj["cleaned_file_path"]):
        raise HTTPException(status_code=400, detail="Please clean the dataset first")

    df = pd.read_csv(proj["cleaned_file_path"])

    # Determine target columns
    target_column = req.target_column
    if target_column in ["*", "all", "__all__"]:
        targets_to_train = [col for col in df.columns if df[col].dropna().nunique() > 1] or [df.columns[-1]]
    else:
        if not target_column:
            target_column = df.columns[-1]
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        targets_to_train = [target_column]

    proj_dir = os.path.join(DATA_DIR, f"project_{project_id}")
    results = {}
    last_res = None
    last_explanation = ""
    trained_count = 0

    # ── Step 2: Slow work — ML training + Ollama (DB is closed during this) ──
    for target in targets_to_train:
        try:
            train_res = await train_best_model(df, target, proj_dir)
            explanation = await generate_explanation(
                train_res["task_type"], train_res["algorithm"],
                train_res["metrics"], train_res["feature_importance"]
            )

            # Rename model.pkl to target-specific file
            models_dir = os.path.join(proj_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            std_model_path = os.path.join(models_dir, "model.pkl")
            target_model_path = os.path.join(models_dir, f"model_{target}.pkl")
            if os.path.exists(std_model_path):
                try:
                    if os.path.exists(target_model_path):
                        os.remove(target_model_path)
                    os.rename(std_model_path, target_model_path)
                    train_res["model_path"] = target_model_path
                except Exception as ex:
                    print(f"Failed to rename model: {ex}")

            # Save explanation + meta to disk
            predictions_dir = os.path.join(proj_dir, "predictions")
            os.makedirs(predictions_dir, exist_ok=True)
            with open(os.path.join(predictions_dir, f"explanation_{target}.txt"), "w") as f:
                f.write(explanation)
            with open(os.path.join(models_dir, f"model_meta_{target}.json"), "w") as f:
                json.dump(train_res, f, indent=2)

            results[target] = {
                "task_type": train_res["task_type"], "algorithm": train_res["algorithm"],
                "metrics": train_res["metrics"], "feature_importance": train_res["feature_importance"],
                "explanation": explanation
            }
            last_res = train_res
            last_explanation = explanation
            trained_count += 1
        except Exception as e:
            print(f"Failed to train target '{target}': {e}")
            if len(targets_to_train) == 1:
                raise HTTPException(status_code=500, detail=f"Training failed for '{target}': {str(e)}")

    if trained_count == 0:
        raise HTTPException(status_code=500, detail="Failed to train any target columns.")

    # ── Step 3: Fresh short DB write ─────────────────────────────────────────
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        for target, res_data in results.items():
            cursor.execute("DELETE FROM models WHERE project_id=? AND target_column=?", (project_id, target))
            primary_metric = list(res_data["metrics"].values())[0]
            target_model_path = os.path.join(proj_dir, "models", f"model_{target}.pkl")
            cursor.execute(
                "INSERT INTO models (project_id, target_column, task_type, algorithm, accuracy, model_path, feature_importance) VALUES (?,?,?,?,?,?,?)",
                (project_id, target, res_data["task_type"], res_data["algorithm"],
                 primary_metric, target_model_path, json.dumps(res_data["feature_importance"]))
            )
            cursor.execute("DELETE FROM predictions WHERE project_id=? AND target_column=?", (project_id, target))
            cursor.execute(
                "INSERT INTO predictions (project_id, target_column, explanation) VALUES (?,?,?)",
                (project_id, target, res_data["explanation"])
            )
        cursor.execute("UPDATE projects SET status='trained' WHERE id=?", (project_id,))
        conn.commit()
    finally:
        conn.close()

    if len(targets_to_train) > 1:
        return {
            "message": f"Successfully trained models for {trained_count} columns: {list(results.keys())}",
            "trained_targets": list(results.keys()),
            "task_type": last_res["task_type"] if last_res else "multi-target",
            "algorithm": last_res["algorithm"] if last_res else "Various",
            "metrics": last_res["metrics"] if last_res else {},
            "feature_importance": last_res["feature_importance"] if last_res else {},
            "explanation": last_explanation or "Multi-target models trained.", "results": results
        }
    else:
        target = targets_to_train[0]
        res = results[target]
        return {"message": "Model trained successfully", "task_type": res["task_type"],
                "algorithm": res["algorithm"], "metrics": res["metrics"],
                "feature_importance": res["feature_importance"], "explanation": res["explanation"]}

@app.post("/api/projects/{project_id}/predict")
def predict_endpoint(project_id: int, req: PredictRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get project and dataset to search for matches
    cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    proj_row = cursor.fetchone()
    if not proj_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Project not found")
    proj = dict(proj_row)
    
    df = None
    if proj.get("cleaned_file_path") and os.path.exists(proj["cleaned_file_path"]):
        df = pd.read_csv(proj["cleaned_file_path"])
    elif proj.get("raw_file_path") and os.path.exists(proj["raw_file_path"]):
        df = pd.read_csv(proj["raw_file_path"])
        
    target_column = req.target_column
    is_wildcard = target_column in ["*", "all", "__all__"]
    
    if target_column and not is_wildcard:
        cursor.execute("SELECT * FROM models WHERE project_id = ? AND target_column = ?", (project_id, target_column))
    else:
        cursor.execute("SELECT * FROM models WHERE project_id = ? ORDER BY timestamp DESC", (project_id,))
        
    models_rows = cursor.fetchall()
    if not models_rows:
        conn.close()
        msg = f"No trained model found targeting '{target_column}'" if target_column else "No trained model found for this project"
        raise HTTPException(status_code=400, detail=msg)
        
    if not is_wildcard:
        models_rows = [models_rows[0]]
        
    # Build input_data: prefer explicit dict, else parse prompt_text
    input_data = req.input_data or {}
    if not input_data and req.prompt_text:
        input_data = _parse_prompt_to_features(req.prompt_text)
        
    # Find all matching rows from the dataset
    matched_rows_data = []
    if req.prompt_text and df is not None:
        tokens = [t.strip().lower() for t in req.prompt_text.split(",") if t.strip()]
        if len(tokens) <= 1:
            words = [w.strip() for w in req.prompt_text.split() if w.strip()]
            tokens = [w.lower() for w in words if w.lower() not in ["and", "is", "in", "the", "a", "for", "of", "to"]]
            
        if tokens:
            temp_matches = []
            for _, row in df.iterrows():
                match_count = 0
                for token in tokens:
                    for col in df.columns:
                        val_str = str(row[col]).strip().lower()
                        if val_str == token or (len(token) > 2 and token in val_str):
                            match_count += 1
                            break
                if match_count > 0:
                    temp_matches.append((match_count, row))
            
            # Sort matches by match_count descending, limit to 50
            temp_matches.sort(key=lambda x: x[0], reverse=True)
            matched_rows_data = [item[1] for item in temp_matches[:50]]

    # If any matching rows were found, set matched_row to the first one (for backward-compatibility)
    matched_row = None
    if matched_rows_data:
        matched_row = matched_rows_data[0].replace({np.nan: None}).to_dict()
        # Merge first matched row values into input_data for fallback prediction
        for k, v in matched_row.items():
            if k not in input_data or pd.isnull(input_data[k]) or input_data[k] == "":
                input_data[k] = v
                        
    # Try positional alignment as a secondary fallback if input_data is empty and prompt_text has tokens
    if not input_data and req.prompt_text and df is not None:
        tokens = [t.strip() for t in req.prompt_text.split(",") if t.strip()]
        used_cols = set()
        for token in tokens:
            is_num = False
            num_val = None
            try:
                num_val = float(token)
                is_num = True
            except ValueError:
                pass
                
            matched_col = None
            if is_num:
                for col in df.columns:
                    if col not in used_cols and pd.api.types.is_numeric_dtype(df[col]):
                        input_data[col] = num_val
                        used_cols.add(col)
                        break
            else:
                for col in df.columns:
                    if col not in used_cols and not pd.api.types.is_numeric_dtype(df[col]):
                        uniques = [str(x).lower().strip() for x in df[col].dropna().unique()]
                        if token.lower() in uniques:
                            matched_col = col
                            input_data[col] = token
                            used_cols.add(col)
                            break
                if not matched_col:
                    for col in df.columns:
                        if col not in used_cols and not pd.api.types.is_numeric_dtype(df[col]):
                            input_data[col] = token
                            used_cols.add(col)
                            break
                            
    # Load all models
    loaded_models = []
    for model_row in models_rows:
        model_target = model_row["target_column"]
        model_path = model_row["model_path"]
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
                loaded_models.append({
                    "target": model_target,
                    "model_data": model_data
                })
            except Exception as e:
                print(f"Failed to load model for target '{model_target}': {e}")

    def predict_record(target_input_data: dict, model_data: dict) -> dict:
        pipeline = model_data["pipeline"]
        label_mapping = model_data["label_mapping"]
        num_cols = model_data.get("num_cols", [])
        cat_cols = model_data.get("cat_cols", [])
        expected_cols = num_cols + cat_cols
        
        target_input_data = target_input_data.copy()
        
        # Handle Time-Series forecasting temporal generation
        if model_data.get("task_type") == "forecasting":
            date_col = model_data.get("date_col")
            max_date = model_data.get("max_date")
            
            date_val = None
            for k, v in target_input_data.items():
                if k.lower() in ["date", "time", str(date_col).lower()]:
                    date_val = v
                    break
                    
            if not date_val and req.prompt_text:
                import re
                date_match = re.search(r'\b(\d{4}[-/]\d{2}[-/]\d{2})\b', req.prompt_text)
                if date_match:
                    date_val = date_match.group(1)
                    
            if not date_val:
                if max_date:
                    try:
                        base_dt = pd.to_datetime(max_date)
                        target_dt = base_dt + pd.Timedelta(days=1)
                        date_val = str(target_dt.date())
                    except:
                        date_val = max_date
                else:
                    date_val = str(pd.Timestamp.now().date())
                    
            try:
                dt_parsed = pd.to_datetime(date_val)
                target_input_data["year"] = dt_parsed.year
                target_input_data["month"] = dt_parsed.month
                target_input_data["day"] = dt_parsed.day
                target_input_data["dayofweek"] = dt_parsed.dayofweek
                target_input_data["dayofyear"] = dt_parsed.dayofyear
                target_input_data["quarter"] = dt_parsed.quarter
            except:
                pass
          
        # Align features
        aligned_input = {}
        for col in expected_cols:
            normalized_col = col.lower().replace("_", "").replace(" ", "")
            found_val = np.nan
            for k, v in target_input_data.items():
                normalized_k = k.lower().replace("_", "").replace(" ", "")
                if normalized_k == normalized_col:
                    found_val = v
                    break
            aligned_input[col] = found_val
            
        for col in num_cols:
            val = aligned_input[col]
            if not pd.isnull(val):
                try:
                    aligned_input[col] = float(val)
                except ValueError:
                    aligned_input[col] = np.nan
                    
        input_df = pd.DataFrame([aligned_input])[expected_cols]
        pred = pipeline.predict(input_df)[0]
        
        prediction_val = pred
        if label_mapping and str(pred) in label_mapping:
            prediction_val = label_mapping[str(pred)]
        elif label_mapping and int(pred) in label_mapping:
            prediction_val = label_mapping[int(pred)]
            
        return {
            "prediction": prediction_val,
            "raw_prediction": float(pred) if isinstance(pred, (np.floating, float)) else int(pred) if isinstance(pred, (np.integer, int)) else str(pred),
            "parsed_features": {k: v for k, v in aligned_input.items() if not pd.isnull(v)}
        }

    # Predict main input
    predictions_output = {}
    for item in loaded_models:
        model_target = item["target"]
        model_data = item["model_data"]
        try:
            predictions_output[model_target] = predict_record(input_data, model_data)
        except Exception as e:
            print(f"Prediction failed for target '{model_target}': {e}")

    # Predict all matched rows
    matched_rows_output = []
    for row_series in matched_rows_data:
        row_dict = row_series.replace({np.nan: None}).to_dict()
        row_preds = {}
        for item in loaded_models:
            model_target = item["target"]
            model_data = item["model_data"]
            try:
                row_preds[model_target] = predict_record(row_dict, model_data)
            except Exception as e:
                print(f"Row prediction failed for target '{model_target}': {e}")
        matched_rows_output.append({
            "row": row_dict,
            "predictions": row_preds
        })

    conn.close()
    
    if not predictions_output:
        raise HTTPException(status_code=500, detail="Prediction failed for all trained targets.")
        
    first_key = list(predictions_output.keys())[0]
    
    return {
        "prediction": predictions_output[first_key]["prediction"],
        "raw_prediction": predictions_output[first_key].get("raw_prediction"),
        "parsed_features": predictions_output[first_key].get("parsed_features"),
        "predictions": predictions_output,
        "matched_row": matched_row,
        "matched_rows": matched_rows_output
    }

def _parse_prompt_to_features(prompt_text: str) -> dict:
    """Parse natural language like 'age is 34, city is Boston' into feature dict."""
    import re
    result = {}
    # Match: key is/= value, separated by commas or newlines
    pattern = re.compile(r'([\w\s]+?)\s*(?:is|=|:)\s*([^,\n]+)', re.IGNORECASE)
    for match in pattern.finditer(prompt_text):
        key = match.group(1).strip().lower().replace(' ', '_')
        raw_val = match.group(2).strip()
        # Try numeric
        try:
            result[key] = float(raw_val) if '.' in raw_val else int(raw_val)
        except ValueError:
            result[key] = raw_val
    return result

@app.get("/api/projects/{project_id}/download/{file_type}")
def download_file(project_id: int, file_type: str):
    proj_dir = os.path.join(DATA_DIR, f"project_{project_id}")
    
    if file_type == "cleaned":
        path = os.path.join(proj_dir, "cleaned", "dataset_cleaned.csv")
        if os.path.exists(path):
            return FileResponse(path, media_type="text/csv", filename="dataset_cleaned.csv")
    elif file_type == "explanation":
        path = os.path.join(proj_dir, "predictions", "explanation.txt")
        if os.path.exists(path):
            return FileResponse(path, media_type="text/plain", filename="explanation.txt")
    elif file_type == "audit":
        path = os.path.join(proj_dir, "cleaned", "audit_log.json")
        if os.path.exists(path):
            return FileResponse(path, media_type="application/json", filename="audit_log.json")
            
    raise HTTPException(status_code=404, detail="Requested file not found")

@app.get("/api/projects/{project_id}/preview/{data_type}")
def preview_data(project_id: int, data_type: str):
    """Return first 50 rows of raw or cleaned data as JSON."""
    proj_dir = os.path.join(DATA_DIR, f"project_{project_id}")
    if data_type == "cleaned":
        path = os.path.join(proj_dir, "cleaned", "dataset_cleaned.csv")
    elif data_type == "raw":
        path = os.path.join(proj_dir, "raw", "dataset.csv")
    else:
        raise HTTPException(status_code=400, detail="data_type must be 'raw' or 'cleaned'")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No {data_type} dataset found")

    df = pd.read_csv(path)
    preview = df.head(50).replace({np.nan: None}).to_dict(orient="records")
    return {
        "columns": list(df.columns),
        "rows": len(df),
        "preview": preview
    }

# ── Serve frontend static files ──────────────────────
FRONTEND_DIR = os.path.join(_THIS_DIR, "..", "frontend")
FRONTEND_DIR = os.path.abspath(FRONTEND_DIR)

if os.path.exists(FRONTEND_DIR):
    # Serve static assets (css, js, images) at /static
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    # Serve index.html for the root path (SPA fallback)
    @app.get("/")
    def serve_root():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    # Also serve style.css and script.js at root level for the HTML references
    @app.get("/{filename}")
    def serve_frontend_file(filename: str):
        file_path = os.path.join(FRONTEND_DIR, filename)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        # Fallback to index.html for SPA routes
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
else:
    @app.get("/")
    def serve_root_fallback():
        return HTMLResponse("<h1>AutoML Agent API</h1><p>Frontend directory not found. API is running at /api/</p>")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Disable reload in production (cloud sets PORT externally)
    is_dev = os.environ.get("PORT") is None
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=is_dev)
