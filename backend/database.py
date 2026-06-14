import sqlite3
import os

DB_PATH = "automl_agent.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Projects Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        original_filename TEXT,
        raw_file_path TEXT,
        cleaned_file_path TEXT,
        status TEXT DEFAULT 'created',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Audit Log Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        column_name TEXT,
        operation TEXT,
        details TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES projects (id)
    );
    """)
    
    # Models Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        task_type TEXT,
        algorithm TEXT,
        accuracy REAL,
        model_path TEXT,
        feature_importance TEXT, -- JSON string
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES projects (id)
    );
    """)
    
    # Predictions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        prediction_file_path TEXT,
        explanation TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES projects (id)
    );
    """)
    
    # Migrations to support multiple target columns per project
    try:
        cursor.execute("ALTER TABLE models ADD COLUMN target_column TEXT;")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN target_column TEXT;")
    except Exception:
        pass
        
    conn.commit()
    conn.close()

# Initialize DB on import
init_db()
