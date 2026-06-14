# AutoML Agent

**A Fully Local, Zero-Cost AI Agentic AutoML Pipeline**

AutoML Agent is a lightweight, self-hosted platform that guides users from raw, messy data to actionable machine learning models with plain-English explanations. The application runs entirely on-device, requiring no cloud access, paid APIs, or deep technical knowledge. 

It features an agentic workflow: **Ingest ➔ Auto-Clean (LLM) ➔ AutoML Train (scikit-learn/XGBoost) ➔ Predict & Explain (LLM)**.

---

## 🌟 Key Features

1. **Flexible Ingestion**: Upload files (CSV, TSV, Excel, JSON), scrape text and tables directly from target URLs using Playwright/BeautifulSoup, or sync tables from a MySQL database.
2. **AI Auto-Clean**: A local LLM (via Ollama) scans your raw dataset, identifies anomalies, duplicates, outliers, and null cells, formulates a strategy, cleans it, and keeps a detailed JSON audit log.
3. **AutoML Engine**: Automatically determines if the target variable requires Classification or Regression, pre-processes the data (imputation, encoding, scaling), runs comparative training across Random Forest and XGBoost, and saves the best model.
4. **Predict & Explain**: Generates single predictions using an interactive form and provides a friendly, plain-English breakdown of decision factors and accuracy metrics written by the LLM.

---

## 📁 Repository Directory Structure

```
automl-agent/
├── backend/                  # Python FastAPI Backend
│   ├── main.py               # API Gateway & static assets server
│   ├── cleaner.py            # Local LLM data cleaning orchestrator
│   ├── ml_engine.py          # Preprocessing, AutoML comparison & Explainer
│   ├── scraper.py            # Playwright + BeautifulSoup scraping engine
│   ├── database.py           # SQLite project metadata manager
│   └── requirements.txt      # Python dependencies
├── servlet/                  # Java Tomcat Integration
│   ├── src/com/automl/       
│   │   ├── DatabaseUtil.java # MySQL Connection helper & test data seeder
│   │   └── AutoMLServlet.java# REST API endpoint for DB load/save sync
│   └── web.xml               # Servlet config mappings
├── frontend/                 # High-Fidelity SPA Web App
│   ├── index.html            # Dashboard HTML layout
│   ├── style.css             # Glassmorphic emerald styling
│   └── script.js             # API bindings and charts rendering
└── .gitignore                # Excludes virtual envs, local dbs, and cache
```

---

## 🚀 Local Setup & Run Guide

### Prerequisite 1: Local LLM (Ollama)
1. Download and install [Ollama](https://ollama.com).
2. Start Ollama and download Llama 3 (or your preferred local LLM):
   ```bash
   ollama run llama3
   ```

### Prerequisite 2: Python Backend (FastAPI)
1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install Playwright browser binaries (for scraping):
   ```bash
   playwright install chromium
   ```
5. Launch the FastAPI server:
   ```bash
   python3 main.py
   ```
   *The server runs at `http://localhost:8000/` and automatically hosts the frontend.*

### Prerequisite 3: Java Servlet (Tomcat Sync)
1. Package the `servlet` directory into a `.war` file, or copy it directly under Tomcat's `webapps/` folder naming it `automl`.
2. Ensure you compile the Java files and copy the compiled `.class` files to `WEB-INF/classes/com/automl/`.
3. Put the MySQL Connector jar file (e.g., `mysql-connector-j-x.x.x.jar`) inside `WEB-INF/lib/` to enable MySQL operations.
4. Set up a local MySQL server on port `3306` with user `root` and an empty password (or modify `DatabaseUtil.java` to match your credentials). The servlet will auto-seed a dummy database `datacleaning` with a dirty table `students_raw` on its first run to help you test the pipeline instantly!

---

## 💾 Local Storage Strategy

- **SQLite database (`automl_agent.db`)**: Houses project metadata, user sessions, audit actions, model accuracies, and file paths.
- **`data/` Folder**: Contains isolated workspaces for each project:
  - `raw/`: Stores the raw uploaded or scraped CSV dataset.
  - `cleaned/`: Houses the processed dataset and `audit_log.json`.
  - `models/`: Stores the trained model (`model.pkl`) and algorithm meta-configurations.
  - `predictions/`: Stores the generated predictions and plain-English `explanation.txt`.
