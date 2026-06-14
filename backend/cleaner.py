import os
import json
import pandas as pd
import numpy as np
import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"

async def get_available_ollama_model() -> str:
    """Probes the local Ollama instance and returns the best available model, defaulting to llama3."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags", timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                if models:
                    names = [m["name"] for m in models]
                    for name in ["llama3:latest", "llama3", "mistral:latest", "mistral", "llama3.1", "kimi-k2.5:cloud"]:
                        if name in names:
                            return name
                        for n in names:
                            if name in n:
                                return n
                    return models[0]["name"]
    except Exception:
        pass
    return "llama3"

async def generate_cleaning_strategy(df: pd.DataFrame) -> dict:
    """
    Sends column descriptions and stats to Ollama and gets a structured JSON cleaning plan.
    Falls back to rule-based cleaning if Ollama is unavailable.
    """
    # 1. Gather stats
    stats = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = float(null_count / len(df))
        unique_count = int(df[col].nunique())
        dtype = str(df[col].dtype)
        sample = df[col].dropna().head(3).tolist()
        sample = [str(x) for x in sample]
        
        stats.append({
            "column": col,
            "dtype": dtype,
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": unique_count,
            "sample": sample
        })
        
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns_stats": stats,
        "duplicate_rows": int(df.duplicated().sum())
    }
    
    prompt = f"""
You are an expert Data Cleaning Agent. Analyze the following dataset summary and decide on the best auto-cleaning strategy.
Dataset Summary:
{json.dumps(summary, indent=2)}

Decide which actions to perform. You must respond ONLY with a valid JSON object matching this schema. Do not include any markdown styling, explanation or other text.

JSON Schema:
{{
  "remove_duplicates": true/false,
  "drop_columns": ["col_name_1", ...],
  "fill_nulls": {{
    "col_name_A": "mean" | "median" | "mode" | "custom_value",
    "col_name_B": "mode"
  }},
  "trim_whitespace": true/false,
  "normalize_case_columns": ["col_name_C", ...],
  "reasoning": "A brief explanation of why these steps were chosen."
}}
"""

    model_name = await get_available_ollama_model()

    # 2. Query Ollama (with short timeout)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                OLLAMA_URL,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=10.0
            )
            if resp.status_code == 200:
                result_json = resp.json()
                raw_text = result_json.get("response", "").strip()
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                strategy = json.loads(raw_text)
                strategy["source"] = f"Ollama LLM Agent ({model_name})"
                return strategy
    except Exception as e:
        print(f"Ollama agent request failed: {e}. Falling back to rule-based strategy.")

    # 3. Rule-based Fallback
    fallback_strategy = {
        "remove_duplicates": True if summary["duplicate_rows"] > 0 else False,
        "drop_columns": [c["column"] for c in stats if c["null_pct"] > 0.9], # drop cols with >90% nulls (preserve features)
        "fill_nulls": {},
        "trim_whitespace": True,
        "normalize_case_columns": [],
        "reasoning": "Fallback rule-based strategy applied because Ollama agent was not responding or not running.",
        "source": "Rule-Based Fallback Engine"
    }
    
    for c in stats:
        if c["null_pct"] > 0 and c["null_pct"] <= 0.9:
            if "int" in c["dtype"] or "float" in c["dtype"]:
                fallback_strategy["fill_nulls"][c["column"]] = "median"
            else:
                fallback_strategy["fill_nulls"][c["column"]] = "mode"
                
        if c["dtype"] == "object" and c["unique_count"] < 15 and c["null_pct"] < 0.5:
            fallback_strategy["normalize_case_columns"].append(c["column"])
            
    return fallback_strategy

def _bin_location_value(val) -> str:
    if pd.isnull(val):
        return "Other"
    loc_lower = str(val).lower().strip()
    if "remote" in loc_lower or "wfh" in loc_lower or "home" in loc_lower:
        return "Remote"
    
    tier1_keywords = ["san francisco", "sf ", "sf,", "new york", "nyc", "boston", "seattle", "austin", "silicon valley", "san jose", "los angeles", "la ", "la,", "chicago", "ny,", "ma,", "tx,", "ca,"]
    for kw in tier1_keywords:
        if kw in loc_lower:
            return "Tier 1 Tech Hub"
            
    intl_keywords = ["london", "tokyo", "berlin", "paris", "toronto", "vancouver", "germany", "uk", "canada", "japan", "india", "bangalore", "singapore", "sydney"]
    for kw in intl_keywords:
        if kw in loc_lower:
            return "International"
            
    return "Other US Hub"

def execute_cleaning(df: pd.DataFrame, strategy: dict) -> tuple[pd.DataFrame, list[dict]]:
    """
    Applies the strategy to the DataFrame and returns the cleaned DataFrame along with an audit log.
    """
    cleaned_df = df.copy()
    audit_log = []
    
    # 0. Selective Row Filtering (Drop rows with >60% missing values)
    before_rows = len(cleaned_df)
    threshold = int(len(cleaned_df.columns) * 0.4)
    cleaned_df.dropna(thresh=max(1, threshold), inplace=True)
    dropped_rows = before_rows - len(cleaned_df)
    if dropped_rows > 0:
        audit_log.append({
            "column": "ALL",
            "operation": "Selective Row Filtering",
            "details": f"Dropped {dropped_rows} rows that had more than 60% missing values."
        })

    # 1. Remove Duplicates
    if strategy.get("remove_duplicates"):
        before = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        removed = before - len(cleaned_df)
        if removed > 0:
            audit_log.append({
                "column": "ALL",
                "operation": "Remove Duplicates",
                "details": f"Removed {removed} duplicate rows."
            })

    # 2. Trim Whitespace & Clean strings (Run early to clean representations like "nan" / "None")
    if strategy.get("trim_whitespace"):
        trimmed_count = 0
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
            cleaned_df[col] = cleaned_df[col].replace('nan', np.nan).replace('None', np.nan).replace('', np.nan)
            trimmed_count += 1
            
        if trimmed_count > 0:
            audit_log.append({
                "column": "ALL_TEXT",
                "operation": "Trim Whitespace",
                "details": f"Trimmed whitespace from all string elements across {trimmed_count} columns."
            })

    # 3. Type Restoration Check (Run early so we compute means/medians on actual numeric columns)
    for col in cleaned_df.columns:
        is_numeric = "int" in str(cleaned_df[col].dtype) or "float" in str(cleaned_df[col].dtype) or "double" in str(cleaned_df[col].dtype)
        if not is_numeric:
            non_nulls = cleaned_df[col].dropna()
            if not non_nulls.empty:
                try:
                    converted = pd.to_numeric(non_nulls, errors='coerce')
                    if converted.notnull().sum() >= 0.8 * len(non_nulls):
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                        audit_log.append({
                            "column": col,
                            "operation": "Type Restoration",
                            "details": f"Restored column '{col}' to numeric type after stripping text anomalies."
                        })
                except Exception:
                    pass
                    
    # Location and Headquarters Binning
    for col in cleaned_df.columns:
        col_lower = col.lower()
        if "location" in col_lower or "headquarter" in col_lower:
            before_uniques = cleaned_df[col].nunique()
            cleaned_df[col] = cleaned_df[col].apply(lambda x: _bin_location_value(x))
            after_uniques = cleaned_df[col].nunique()
            audit_log.append({
                "column": col,
                "operation": "Location Binning",
                "details": f"Binned location values (reduced cardinality from {before_uniques} to {after_uniques} categories)."
            })
            
    # 4. Drop Columns
    for col in strategy.get("drop_columns", []):
        if col in cleaned_df.columns:
            cleaned_df.drop(columns=[col], inplace=True)
            audit_log.append({
                "column": col,
                "operation": "Drop Column",
                "details": f"Dropped column '{col}' due to high null count or request."
            })

    # Find a potential group column (categorical, cardinality 2-15) for group-based imputation
    group_col = None
    for c in cleaned_df.columns:
        is_num = "int" in str(cleaned_df[c].dtype) or "float" in str(cleaned_df[c].dtype) or "double" in str(cleaned_df[c].dtype)
        if not is_num:
            uniq = cleaned_df[c].nunique()
            if 2 <= uniq <= 15:
                group_col = c
                break
            
    # 5. Fill Nulls
    for col, fill_type in strategy.get("fill_nulls", {}).items():
        if col in cleaned_df.columns:
            null_count = int(cleaned_df[col].isnull().sum())
            if null_count > 0:
                fill_value = None
                is_numeric = "int" in str(cleaned_df[col].dtype) or "float" in str(cleaned_df[col].dtype)
                
                imputed = False
                # If numeric column and low-cardinality group column exists, perform group-based mean/median filling
                if is_numeric and group_col and group_col != col:
                    try:
                        if fill_type == "mean":
                            cleaned_df[col] = cleaned_df.groupby(group_col)[col].transform(lambda x: x.fillna(x.mean()))
                            imputed = True
                        elif fill_type == "median":
                            cleaned_df[col] = cleaned_df.groupby(group_col)[col].transform(lambda x: x.fillna(x.median()))
                            imputed = True
                    except Exception as ex:
                        print(f"Group-based imputation failed for {col}: {ex}")
                
                if not imputed:
                    if fill_type == "mean":
                        fill_value = cleaned_df[col].mean()
                    elif fill_type == "median":
                        fill_value = cleaned_df[col].median()
                    elif fill_type == "mode":
                        mode_val = cleaned_df[col].mode()
                        fill_value = mode_val[0] if not mode_val.empty else "N/A"
                    else:
                        fill_value = fill_type
                    cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                    
                details_str = f"Filled {null_count} nulls using {fill_type}"
                if imputed:
                    details_str += f" grouped by '{group_col}'."
                else:
                    details_str += f" ({fill_value})."
                    
                audit_log.append({
                    "column": col,
                    "operation": "Fill Nulls",
                    "details": details_str
                })

    # 6. Normalize Case
    for col in strategy.get("normalize_case_columns", []):
        if col in cleaned_df.columns and cleaned_df[col].dtype == "object":
            cleaned_df[col] = cleaned_df[col].astype(str).str.lower()
            audit_log.append({
                "column": col,
                "operation": "Normalize Case",
                "details": f"Normalized strings to lowercase in column '{col}'."
            })
            
    return cleaned_df, audit_log
