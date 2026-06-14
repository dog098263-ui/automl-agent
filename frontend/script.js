'use strict';

// ── CONFIG ────────────────────────────────────────
const FASTAPI_API = window.location.origin.startsWith('file://')
    ? 'http://localhost:8000/api'
    : `${window.location.origin}/api`;
const SERVLET_API = window.location.origin.startsWith('file://')
    ? 'http://localhost:8080/automl/api'
    : `${window.location.protocol}//${window.location.hostname}:8080/automl/api`;

// ── STATE ─────────────────────────────────────────
let projects = [];
let activeProjectId = null;
let activeProjectData = null;
let currentColumns = [];

// ── INITIALIZATION ────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkOllama();
    loadProjects();
    setupDropZone();
    setInterval(checkOllama, 10000);

    // Close user dropdown when clicking outside
    document.addEventListener('click', (e) => {
        const wrap = document.getElementById('userMenuWrap');
        if (wrap && !wrap.contains(e.target)) {
            document.getElementById('userDropdown').classList.remove('open');
        }
    });

    // Real-Time Sync event listeners
    const promptInput = document.getElementById('promptInput');
    if (promptInput) {
        promptInput.addEventListener('input', syncNLToForm);
    }
    const formFields = document.getElementById('dynamicFormFields');
    if (formFields) {
        formFields.addEventListener('input', (e) => {
            if (e.target.classList.contains('form-feature-input')) {
                syncFormToNL();
            }
        });
    }
});

// Check Ollama service availability via backend proxy
async function checkOllama() {
    try {
        const r = await fetch(`${FASTAPI_API}/ollama/status`, { signal: AbortSignal.timeout(1500) });
        if (r.ok) {
            const data = await r.json();
            if (data.status === 'connected') {
                document.getElementById('ollama-dot').className = 'api-dot connected';
                document.getElementById('ollama-status').textContent = 'Ollama: Connected ✓';
                return;
            }
        }
        throw new Error();
    } catch (e) {
        document.getElementById('ollama-dot').className = 'api-dot';
        document.getElementById('ollama-status').textContent = 'Ollama: Not Running ✕';
    }
}

// ── USER MENU DROPDOWN ────────────────────────────
function toggleUserMenu() {
    document.getElementById('userDropdown').classList.toggle('open');
}

// ── TABS NAVIGATION ────────────────────────────────
const META = {
    dashboard: { title: 'Dashboard Overview', sub: 'Status, pipeline progress, and dataset previews.' },
    ingest:    { title: 'Data Ingestion', sub: 'Upload files, scrape web URLs, or sync from databases.' },
    clean:     { title: 'AI Data Optimizer', sub: 'Run AI agent to clean and standardize your dataset.' },
    train:     { title: 'AutoML Model Training', sub: 'Auto-detect target and train predictive models locally.' },
    predict:   { title: 'Inference & Explanations', sub: 'Describe your data in plain English to generate a prediction.' }
};

function go(pg) {
    document.querySelectorAll('.page').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    
    document.getElementById('page-' + pg).classList.add('active');
    document.getElementById('nav-' + pg).classList.add('active');
    
    document.getElementById('pg-title').textContent = META[pg].title;
    document.getElementById('pg-sub').textContent = META[pg].sub;
    
    // Refresh page specific views
    if (activeProjectId) {
        if (pg === 'dashboard') refreshDashboard();
        if (pg === 'clean') refreshCleanView();
        if (pg === 'train') refreshTrainView();
        if (pg === 'predict') refreshPredictView();
    }
}

// ── PROJECTS MANAGEMENT ──────────────────────────
async function loadProjects() {
    try {
        const r = await fetch(`${FASTAPI_API}/projects`);
        projects = await r.json();
        
        const sel = document.getElementById('projectSelect');
        sel.innerHTML = '<option value="">-- Select/Switch Project --</option>';
        
        projects.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = `${p.name} (id:${p.id})`;
            if (activeProjectId === p.id) {
                opt.selected = true;
            }
            sel.appendChild(opt);
        });
    } catch (e) {
        toast('Error loading projects: ' + e.message, 'error');
    }
}

function showNewProjectModal() {
    document.getElementById('newProjectName').value = '';
    document.getElementById('newProjectModal').style.display = 'flex';
}

function hideNewProjectModal() {
    document.getElementById('newProjectModal').style.display = 'none';
}

async function createNewProject() {
    const name = document.getElementById('newProjectName').value.trim();
    if (!name) {
        toast('Project name is required', 'warn');
        return;
    }
    
    try {
        const r = await fetch(`${FASTAPI_API}/projects`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name })
        });
        const res = await r.json();
        activeProjectId = res.id;
        toast('Project created successfully');
        hideNewProjectModal();
        await loadProjects();
        await selectProject(activeProjectId);
    } catch (e) {
        toast('Failed to create project: ' + e.message, 'error');
    }
}

// ── PROJECT DELETION ─────────────────────────────
function confirmDeleteProject() {
    if (!activeProjectId || !activeProjectData) {
        toast('No active project to delete', 'warn');
        return;
    }
    document.getElementById('deleteProjectNameDisplay').textContent = `"${activeProjectData.project.name}"`;
    document.getElementById('deleteProjectModal').style.display = 'flex';
}

function hideDeleteProjectModal() {
    document.getElementById('deleteProjectModal').style.display = 'none';
}

async function deleteProject() {
    if (!activeProjectId) return;
    try {
        const r = await fetch(`${FASTAPI_API}/projects/${activeProjectId}`, { method: 'DELETE' });
        if (!r.ok) throw new Error('Delete failed');
        
        hideDeleteProjectModal();
        toast('Project deleted successfully');
        
        // Reset state
        activeProjectId = null;
        activeProjectData = null;
        currentColumns = [];
        
        await loadProjects();
        updateUIForActiveProject();
    } catch (e) {
        toast('Failed to delete project: ' + e.message, 'error');
        hideDeleteProjectModal();
    }
}

async function changeProject() {
    const val = document.getElementById('projectSelect').value;
    if (val) {
        await selectProject(parseInt(val));
    } else {
        activeProjectId = null;
        activeProjectData = null;
        updateUIForActiveProject();
    }
}

async function selectProject(id) {
    activeProjectId = id;
    try {
        const r = await fetch(`${FASTAPI_API}/projects/${id}`);
        activeProjectData = await r.json();
        
        updateUIForActiveProject();
        toast(`Active Project: ${activeProjectData.project.name}`);
        go('dashboard');
    } catch (e) {
        toast('Failed to load project: ' + e.message, 'error');
    }
}

function updateUIForActiveProject() {
    const badge = document.getElementById('activeProjectBadge');
    const dropdownName = document.getElementById('dropdownProjectName');
    const btnDelete = document.getElementById('btnDeleteProject');
    const dropdownDelete = document.getElementById('dropdownDeleteBtn');

    if (activeProjectId && activeProjectData) {
        const name = activeProjectData.project.name;
        badge.textContent = name;
        if (dropdownName) dropdownName.textContent = name;
        if (btnDelete) btnDelete.disabled = false;
        if (dropdownDelete) dropdownDelete.disabled = false;
        
        // Update Pipeline status steps
        updatePipelineSteps(activeProjectData.project.status);
        
        // Update column selectors
        updateColumnsList();
        
        // Refresh previews
        refreshDashboard();
    } else {
        badge.textContent = 'No Project';
        if (dropdownName) dropdownName.textContent = 'No active project';
        if (btnDelete) btnDelete.disabled = true;
        if (dropdownDelete) dropdownDelete.disabled = true;
        resetDashboardStats();
    }
}

function updatePipelineSteps(status) {
    const steps = ['uploaded', 'cleaned', 'trained'];
    
    document.querySelectorAll('.progress-step').forEach(el => el.classList.remove('completed'));
    document.querySelectorAll('.progress-connector').forEach(el => el.classList.remove('active'));
    
    // Step 1: Ingestion always shown as completed when project is selected
    document.getElementById('step-prog-1').classList.add('completed');
    
    if (status === 'uploaded') {
        // Just uploaded
    } else if (status === 'cleaned') {
        document.getElementById('step-prog-2').classList.add('completed');
        document.getElementById('step-conn-1').classList.add('active');
    } else if (status === 'trained') {
        document.getElementById('step-prog-2').classList.add('completed');
        document.getElementById('step-prog-3').classList.add('completed');
        document.getElementById('step-prog-4').classList.add('completed');
        document.getElementById('step-conn-1').classList.add('active');
        document.getElementById('step-conn-2').classList.add('active');
        document.getElementById('step-conn-3').classList.add('active');
    }
}

function updateColumnsList() {
    const stats = activeProjectData.stats;
    let cols = [];
    if (stats.cleaned) {
        cols = stats.cleaned.columns;
    } else if (stats.raw) {
        cols = stats.raw.columns;
    }
    
    currentColumns = cols;
    
    const inferredBox = document.getElementById('inferredTargetBox');
    const targetSelect = document.getElementById('targetColumnSelect');
    if (cols.length > 0 && inferredBox && targetSelect) {
        targetSelect.innerHTML = '';
        
        // Add "All Columns (Target Everything)" option
        const optAll = document.createElement('option');
        optAll.value = '*';
        optAll.textContent = '★ All Columns (Target Everything)';
        targetSelect.appendChild(optAll);
        
        // Add each column
        cols.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            targetSelect.appendChild(opt);
        });
        
        // Set "*" as default selection
        targetSelect.value = '*';
        
        inferredBox.style.display = 'block';
    } else if (inferredBox) {
        inferredBox.style.display = 'none';
    }
}

// ── DATA PREVIEWS & STATS ────────────────────────
function refreshDashboard() {
    if (!activeProjectData) return;
    
    const stats = activeProjectData.stats;
    const proj = activeProjectData.project;
    
    if (stats.cleaned) {
        document.getElementById('d-rows').textContent = stats.cleaned.rows.toLocaleString();
        document.getElementById('d-cols').textContent = stats.cleaned.columns.length;
        document.getElementById('d-nulls').textContent = stats.cleaned.null_cells.toLocaleString();
        document.getElementById('d-dups').textContent = stats.cleaned.duplicates.toLocaleString();
        
        document.getElementById('d-rows-sub').textContent = 'cleaned dataset';
        document.getElementById('d-cols-sub').textContent = 'cleaned dataset';
        document.getElementById('d-nulls-sub').textContent = 'cells fixed';
        document.getElementById('d-dups-sub').textContent = 'duplicates dropped';
        document.getElementById('d-preview-badge').textContent = 'Cleaned Data';
        
        loadPreviewTable('cleaned', document.getElementById('d-preview-container'));
    } else if (stats.raw) {
        document.getElementById('d-rows').textContent = stats.raw.rows.toLocaleString();
        document.getElementById('d-cols').textContent = stats.raw.columns.length;
        document.getElementById('d-nulls').textContent = stats.raw.null_cells.toLocaleString();
        document.getElementById('d-dups').textContent = stats.raw.duplicates.toLocaleString();
        
        document.getElementById('d-rows-sub').textContent = 'raw dataset';
        document.getElementById('d-cols-sub').textContent = 'raw columns';
        document.getElementById('d-nulls-sub').textContent = 'needs cleaning';
        document.getElementById('d-dups-sub').textContent = 'needs dedup';
        document.getElementById('d-preview-badge').textContent = 'Raw Data';
        
        loadPreviewTable('raw', document.getElementById('d-preview-container'));
    } else {
        resetDashboardStats();
    }
}

async function loadPreviewTable(dataType, container) {
    try {
        const r = await fetch(`${FASTAPI_API}/projects/${activeProjectId}/preview/${dataType}`);
        if (!r.ok) {
            container.innerHTML = '<div class="empty"><p>Could not load tabular preview.</p></div>';
            return;
        }
        const res = await r.json();
        const rows = res.preview.slice(0, 15);
        renderTbl(container, rows, res.columns);
    } catch (e) {
        container.innerHTML = `<div class="empty"><p>Error building table: ${e.message}</p></div>`;
    }
}

function parseCSVText(text, maxRows = 10) {
    const lines = text.split('\n').map(l => l.trim()).filter(l => l);
    if (lines.length === 0) return { columns: [], data: [] };
    
    const columns = lines[0].split(',');
    const data = [];
    
    for (let i = 1; i < Math.min(lines.length, maxRows + 1); i++) {
        const values = lines[i].split(',');
        const row = {};
        columns.forEach((col, idx) => {
            row[col] = idx < values.length ? values[idx] : '';
        });
        data.push(row);
    }
    return { columns, data };
}

function resetDashboardStats() {
    ['d-rows', 'd-cols', 'd-nulls', 'd-dups'].forEach(id => {
        document.getElementById(id).textContent = '—';
    });
    document.getElementById('d-preview-badge').textContent = 'No data';
    document.getElementById('d-preview-container').innerHTML = `
        <div class="empty">
            <div class="ico">📂</div>
            <p>No dataset loaded yet.</p>
            <button class="btn btn-primary" style="margin-top:16px;" onclick="go('ingest')">↑ Upload CSV / Start Scraping</button>
        </div>`;
}

function renderTbl(el, rows, cols) {
    if (!rows || !rows.length) {
        el.innerHTML = '<div class="empty"><p>No rows to display.</p></div>';
        return;
    }
    
    const wrap = document.createElement('div');
    wrap.className = 'tbl-wrap';
    
    let h = '<table><thead><tr>' + cols.map(c => `<th>${esc(c)}</th>`).join('') + '</tr></thead><tbody>';
    rows.forEach(row => {
        h += '<tr>' + cols.map(c => {
            const v = String(row[c] === undefined || row[c] === null ? '' : row[c]).trim();
            const isNull = ['null', 'NULL', 'NaN', 'N/A', ''].includes(v);
            return `<td class="${isNull ? 'null-cell' : ''}">${esc(isNull ? 'null' : v)}</td>`;
        }).join('') + '</tr>';
    });
    h += '</tbody></table>';
    
    wrap.innerHTML = h;
    el.innerHTML = '';
    el.appendChild(wrap);
}

function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── STEP 1: INGESTION (UPLOAD / SCRAPE / DB) ────────
function setupDropZone() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    
    dropZone.addEventListener('click', () => {
        if (!requireActiveProject()) return;
        fileInput.click();
    });
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--accent)';
        dropZone.style.background = 'var(--accent-soft)';
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'rgba(16,185,129,0.3)';
        dropZone.style.background = 'var(--surface2)';
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'rgba(16,185,129,0.3)';
        dropZone.style.background = 'var(--surface2)';
        
        if (!requireActiveProject()) return;
        if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) uploadFile(e.target.files[0]);
    });
}

async function uploadFile(file) {
    toast(`Uploading "${file.name}"...`, 'info');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const r = await fetch(`${FASTAPI_API}/projects/${activeProjectId}/upload`, {
            method: 'POST',
            body: formData
        });
        if (!r.ok) throw new Error('Upload failed');
        const res = await r.json();
        
        // Update state without navigating away
        activeProjectData.preview_cache = res.preview;
        activeProjectData.project.status = 'uploaded';
        if (!activeProjectData.stats) activeProjectData.stats = {};
        activeProjectData.stats.raw = { rows: res.rows, columns: res.columns, null_cells: 0, duplicates: 0 };
        
        toast('Dataset uploaded and parsed ✓');
        
        // Stay on ingest page, just update preview
        renderTbl(document.getElementById('ingest-preview-container'), res.preview, res.columns);
        updateColumnsList();
        updatePipelineSteps('uploaded');
    } catch (e) {
        toast('Upload error: ' + e.message, 'error');
    }
}

async function runScraper() {
    if (!requireActiveProject()) return;
    const url = document.getElementById('scraperUrl').value.trim();
    if (!url) {
        toast('Please enter a URL to scrape', 'warn');
        return;
    }
    
    toast('Agent is launching Playwright scraper...', 'info');
    try {
        const r = await fetch(`${FASTAPI_API}/projects/${activeProjectId}/scrape`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ url })
        });
        if (!r.ok) {
            const err = await r.json();
            throw new Error(err.detail || 'Scraping failed');
        }
        const res = await r.json();
        
        // Update state without navigating away — FIX: stay on Ingest page
        activeProjectData.preview_cache = res.preview;
        activeProjectData.project.status = 'uploaded';
        if (!activeProjectData.stats) activeProjectData.stats = {};
        activeProjectData.stats.raw = { rows: res.rows, columns: res.columns, null_cells: 0, duplicates: 0 };
        
        toast('URL Scraped successfully ✓');
        
        // Stay on Ingest page: only update the inline preview below
        renderTbl(document.getElementById('ingest-preview-container'), res.preview, res.columns);
        updateColumnsList();
        updatePipelineSteps('uploaded');
    } catch (e) {
        toast('Scraper error: ' + e.message, 'error');
    }
}

async function loadFromMySQL() {
    if (!requireActiveProject()) return;
    const table = document.getElementById('dbTableName').value.trim();
    if (!table) {
        toast('Enter a database table name', 'warn');
        return;
    }
    
    toast('Connecting to Tomcat Servlet Database sync...', 'info');
    try {
        const r = await fetch(`${SERVLET_API}/db/load`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ table })
        });
        if (!r.ok) throw new Error('Servlet load failed');
        const res = await r.json();
        
        const csvText = convertJSONtoCSV(res.data, res.columns);
        const blob = new Blob([csvText], { type: 'text/csv' });
        const file = new File([blob], `mysql_${table}.csv`);
        
        await uploadFile(file);
        toast(`Synced ${res.rows} rows from MySQL Database!`);
    } catch (e) {
        toast('Tomcat sync failed: ' + e.message, 'error');
    }
}

function convertJSONtoCSV(data, columns) {
    let csv = columns.join(',') + '\n';
    data.forEach(row => {
        csv += columns.map(c => {
            const val = String(row[c] || '').replace(/"/g, '""');
            return `"${val}"`;
        }).join(',') + '\n';
    });
    return csv;
}

// ── STEP 2: AUTO-CLEAN (LLM AGENT) ─────────────────
function refreshCleanView() {
    if (!activeProjectData) return;
    
    const logs = activeProjectData.audit_logs;
    
    if (logs && logs.length > 0) {
        // Show button to re-run if needed
        document.getElementById('clean-status-area').innerHTML = `
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
                <div>
                    <span class="badge badge-green" style="font-size:12px; padding:6px 14px;">✓ Data Optimized — ${logs.length} operations applied</span>
                </div>
                <button class="btn btn-ghost btn-sm" onclick="runAutoClean()">↺ Re-run Optimizer</button>
            </div>`;
        
        // Load Cleaned preview directly
        loadCleanedPreview();
    } else {
        document.getElementById('clean-status-area').innerHTML = `
            <div class="empty">
                <p>Run the AI agent to analyze your dataset and apply intelligent corrections.</p>
                <button class="btn btn-primary" style="margin-top:14px;" onclick="runAutoClean()">✦ Optimize and Clean Data</button>
            </div>`;
        document.getElementById('clean-preview-container').innerHTML = '<div class="empty"><p>Cleaned data will appear here after running the optimizer.</p></div>';
    }
}

async function runAutoClean() {
    if (!requireActiveProject()) return;
    
    toast('AI Optimizer active. Analyzing dataset and querying Ollama...', 'info');
    document.getElementById('clean-status-area').innerHTML = `
        <div class="empty">
            <div class="ico">⏳</div>
            <p>AI Agent is scanning dataset and optimizing via Ollama...</p>
        </div>`;
        
    try {
        const r = await fetch(`${FASTAPI_API}/projects/${activeProjectId}/clean`, { method: 'POST' });
        if (!r.ok) throw new Error('Auto-cleaning failed');
        const res = await r.json();
        
        // Update local state without navigating away — FIX: stay on Clean page
        activeProjectData.audit_logs = res.audit_logs;
        activeProjectData.project.status = 'cleaned';
        if (res.preview && res.preview.length > 0) {
            const cols = Object.keys(res.preview[0]);
            if (!activeProjectData.stats) activeProjectData.stats = {};
            activeProjectData.stats.cleaned = {
                rows: res.preview.length,
                columns: cols,
                null_cells: 0,
                duplicates: 0
            };
            currentColumns = cols;
            updateColumnsList();
        }
        
        toast('Optimization complete! ✓');
        updatePipelineSteps('cleaned');
        refreshCleanView();
    } catch (e) {
        toast('Optimizer error: ' + e.message, 'error');
        refreshCleanView();
    }
}

async function loadCleanedPreview() {
    const container = document.getElementById('clean-preview-container');
    try {
        const r = await fetch(`${FASTAPI_API}/projects/${activeProjectId}/preview/cleaned`);
        if (r.ok) {
            const res = await r.json();
            renderTbl(container, res.preview.slice(0, 10), res.columns);
        }
    } catch (e) {
        container.innerHTML = `<div class="empty"><p>Preview error: ${e.message}</p></div>`;
    }
}

// ── STEP 3: AUTOML MODEL TRAINING ──────────────
function refreshTrainView() {
    if (!activeProjectData) return;
    
    const targetSelect = document.getElementById('targetColumnSelect');
    let selectedTarget = targetSelect ? targetSelect.value : '*';
    
    const models = activeProjectData.models || [];
    let model = null;
    
    if (selectedTarget === '*') {
        model = activeProjectData.model || (models.length > 0 ? models[0] : null);
    } else {
        model = models.find(m => m.target_column === selectedTarget);
    }
    
    const statsRow = document.getElementById('train-stats-row');
    const container = document.getElementById('feature-importance-container');
    
    if (model) {
        statsRow.style.display = 'grid';
        document.getElementById('t-type').textContent = model.task_type;
        document.getElementById('t-algo').textContent = model.algorithm;
        
        if (model.task_type === 'classification') {
            document.getElementById('t-metric-lbl').textContent = 'Accuracy Score';
            document.getElementById('t-score').textContent = (model.accuracy * 100).toFixed(1) + '%';
        } else {
            document.getElementById('t-metric-lbl').textContent = 'R-Squared Score';
            document.getElementById('t-score').textContent = (model.accuracy * 100).toFixed(1) + '%';
        }
        
        renderFeatureImportance(model.feature_importance, container);
    } else {
        statsRow.style.display = 'none';
        container.innerHTML = '<div class="empty"><p>Run training to display feature importances.</p></div>';
    }
}

function renderFeatureImportance(importances, container) {
    if (!importances || Object.keys(importances).length === 0) {
        container.innerHTML = '<div class="empty"><p>No feature importance data available.</p></div>';
        return;
    }
    
    const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]);
    const maxVal = sorted[0] ? sorted[0][1] : 1;
    
    let html = '<div style="margin-top:10px;">';
    sorted.forEach(([feat, val]) => {
        const pct = (val / maxVal) * 100;
        html += `
        <div class="bar-row">
            <div class="bar-label" title="${esc(feat)}">${esc(feat)}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width: ${pct}%;"></div>
            </div>
            <div class="bar-val">${(val * 100).toFixed(1)}%</div>
        </div>`;
    });
    html += '</div>';
    container.innerHTML = html;
}

async function runModelTraining() {
    if (!requireActiveProject()) return;
    
    const targetSelect = document.getElementById('targetColumnSelect');
    const target = targetSelect ? targetSelect.value : (currentColumns.length > 0 ? currentColumns[currentColumns.length - 1] : null);
    
    if (!target) {
        toast('No dataset columns available. Please ingest and clean data first.', 'warn');
        return;
    }
    
    const targetLabel = target === '*' ? 'all columns (Target Everything)' : `"${target}"`;
    toast(`Training model — targeting ${targetLabel}...`, 'info');
    document.getElementById('btnTrainModel').disabled = true;
    document.getElementById('btnTrainModel').textContent = '⏳ Training Models...';
    
    try {
        const r = await fetch(`${FASTAPI_API}/projects/${activeProjectId}/train`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ target_column: target })
        });
        if (!r.ok) {
            const err = await r.json();
            throw new Error(err.detail || 'Training failed');
        }
        const res = await r.json();
        
        // Refresh project data to retrieve all newly trained models and explanations
        const projectR = await fetch(`${FASTAPI_API}/projects/${activeProjectId}`);
        activeProjectData = await projectR.json();
        
        toast(`Training completed successfully! ✓`);
        updatePipelineSteps('trained');
        
        // Restore target selector state
        if (targetSelect) targetSelect.value = target;
        
        refreshTrainView();
    } catch (e) {
        toast('Training error: ' + e.message, 'error');
    } finally {
        document.getElementById('btnTrainModel').disabled = false;
        document.getElementById('btnTrainModel').textContent = '◎ Start AutoML Training';
    }
}

// ── STEP 4: PREDICT & EXPLAIN ──────────────────────
let currentPredictMode = 'nl'; // default

function setPredictMode(mode) {}

function changePredictTarget() {
    const targetSelect = document.getElementById('predictTargetSelect');
    if (targetSelect) {
        updatePredictionFieldsForTarget(targetSelect.value);
    }
}

function updatePredictionFieldsForTarget(targetColumn) {
    if (!activeProjectData) return;
    
    const models = activeProjectData.models || [];
    const model = models.find(m => m.target_column === targetColumn);
    if (!model) return;
    
    // 1. Update Explanation Box
    const predictions = activeProjectData.predictions || [];
    const pred = predictions.find(p => p.target_column === targetColumn) || activeProjectData.prediction;
    const explainBox = document.getElementById('explanationText');
    if (explainBox) {
        if (pred && pred.explanation) {
            explainBox.innerHTML = formatExplanationText(pred.explanation);
        } else {
            explainBox.innerHTML = '<div class="empty"><p>Model training explanation will appear here.</p></div>';
        }
    }
    
    // 2. Enable predict button
    const btnPredict = document.getElementById('btnPredict');
    if (btnPredict) btnPredict.disabled = false;
    
    // 3. Dynamically generate Form Fields
    const numCols = model.num_cols || [];
    const catCols = model.cat_cols || [];
    const formFields = document.getElementById('dynamicFormFields');
    if (formFields) {
        formFields.innerHTML = '';
        if (numCols.length === 0 && catCols.length === 0) {
            formFields.innerHTML = '<p style="color:var(--muted); font-size:12px; grid-column: 1 / -1;">No input features detected for this target.</p>';
            return;
        }
        
        // Numerical inputs
        numCols.forEach(col => {
            const wrap = document.createElement('div');
            wrap.className = 'form-group';
            wrap.style.display = 'flex';
            wrap.style.flexDirection = 'column';
            wrap.style.gap = '4px';
            
            const label = document.createElement('label');
            label.className = 'section-lbl';
            label.textContent = col;
            
            const input = document.createElement('input');
            input.type = 'number';
            input.step = 'any';
            input.className = 'df-input form-feature-input';
            input.dataset.col = col;
            input.dataset.type = 'numeric';
            input.placeholder = 'e.g. 1.5';
            
            wrap.appendChild(label);
            wrap.appendChild(input);
            formFields.appendChild(wrap);
        });
        
        // Categorical inputs
        catCols.forEach(col => {
            const wrap = document.createElement('div');
            wrap.className = 'form-group';
            wrap.style.display = 'flex';
            wrap.style.flexDirection = 'column';
            wrap.style.gap = '4px';
            
            const label = document.createElement('label');
            label.className = 'section-lbl';
            label.textContent = col;
            
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'df-input form-feature-input';
            input.dataset.col = col;
            input.dataset.type = 'categorical';
            input.placeholder = 'e.g. Value';
            
            wrap.appendChild(label);
            wrap.appendChild(input);
            formFields.appendChild(wrap);
        });
        syncNLToForm();
    }
}

function refreshPredictView() {
    if (!activeProjectData) return;
    
    const models = activeProjectData.models || [];
    const select = document.getElementById('predictTargetSelect');
    const container = document.getElementById('predictTargetBox');
    
    if (models.length > 0) {
        if (container) container.style.display = 'block';
        if (select) {
            const currentSelected = select.value;
            select.innerHTML = '';
            
            models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.target_column;
                opt.textContent = m.target_column;
                select.appendChild(opt);
            });
            
            if (currentSelected && models.some(m => m.target_column === currentSelected)) {
                select.value = currentSelected;
            } else {
                select.value = models[0].target_column;
            }
            updatePredictionFieldsForTarget(select.value);
        }
    } else {
        if (container) container.style.display = 'none';
        const explainBox = document.getElementById('explanationText');
        if (explainBox) explainBox.innerHTML = '<div class="empty"><p>Model training explanation will appear here.</p></div>';
        const btnPredict = document.getElementById('btnPredict');
        if (btnPredict) btnPredict.disabled = true;
        document.getElementById('predictionResultCard').style.display = 'none';
    }
}

function formatExplanationText(text) {
    let formatted = esc(text);
    formatted = formatted.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    formatted = formatted.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    formatted = formatted.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    return formatted;
}

// ── NATURAL LANGUAGE PROMPT PARSER ────────────────
function parsePromptToFeatures(promptText) {
    const input_data = {};
    const patterns = [
        /(\w[\w\s]*?)\s+(?:is|=|:)\s+([^\,]+)/gi,
        /(\w[\w\s]*?)\s+of\s+([^\,]+)/gi,
    ];
    
    for (const pattern of patterns) {
        let match;
        while ((match = pattern.exec(promptText)) !== null) {
            const key = match[1].trim().toLowerCase().replace(/\s+/g, '_');
            const rawVal = match[2].trim();
            const num = parseFloat(rawVal);
            input_data[key] = isNaN(num) ? rawVal : num;
        }
    }
    return input_data;
}

// Sync Natural Language prompt input to Structured Form fields
function syncNLToForm() {
    const promptText = document.getElementById('promptInput').value.trim();
    const parsed = parsePromptToFeatures(promptText);
    
    document.querySelectorAll('.form-feature-input').forEach(input => {
        const col = input.dataset.col;
        const normCol = col.toLowerCase().replace(/_/g, '').replace(/\s/g, '');
        
        let foundVal = '';
        for (const [k, val] of Object.entries(parsed)) {
            const normK = k.toLowerCase().replace(/_/g, '').replace(/\s/g, '');
            if (normK === normCol) {
                foundVal = val;
                break;
            }
        }
        input.value = foundVal;
    });
}

// Sync Structured Form fields back to Natural Language prompt
function syncFormToNL() {
    const parts = [];
    document.querySelectorAll('.form-feature-input').forEach(input => {
        const col = input.dataset.col;
        const val = input.value.trim();
        if (val !== '') {
            parts.push(`${col} is ${val}`);
        }
    });
    if (parts.length > 0) {
        document.getElementById('promptInput').value = parts.join(', ');
    }
}

async function runPrediction(event) {
    event.preventDefault();
    if (!requireActiveProject()) return;
    
    const targetSelect = document.getElementById('predictTargetSelect');
    const targetColumn = targetSelect ? targetSelect.value : null;
    
    const promptText = document.getElementById('promptInput').value.trim();
    
    const inputData = {};
    let hasFormValue = false;
    document.querySelectorAll('.form-feature-input').forEach(input => {
        const col = input.dataset.col;
        const type = input.dataset.type;
        const val = input.value.trim();
        if (val !== '') {
            hasFormValue = true;
            if (type === 'numeric') {
                inputData[col] = parseFloat(val);
            } else {
                inputData[col] = val;
            }
        }
    });
    
    if (!promptText && !hasFormValue) {
        toast('Please enter a description or fill in structured features', 'warn');
        return;
    }
    
    let payload = {
        target_column: '*',
        prompt_text: promptText || null,
        input_data: hasFormValue ? inputData : null
    };
    
    toast('Generating prediction...', 'info');
    try {
        const r = await fetch(`${FASTAPI_API}/projects/${activeProjectId}/predict`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        if (!r.ok) {
            const err = await r.json();
            throw new Error(err.detail || 'Prediction failed');
        }
        const res = await r.json();
        
        const card = document.getElementById('predictionResultCard');
        const listContainer = document.getElementById('predictionsListContainer');
        const matchedContainer = document.getElementById('matchedRowContainer');
        const tableHeaderRow = document.getElementById('matchedRowsHeaderRow');
        const tableBody = document.getElementById('matchedRowsBody');
        
        if (card && listContainer) {
            listContainer.innerHTML = '';
            
            const predictions = res.predictions || {};
            if (Object.keys(predictions).length === 0 && res.prediction !== undefined) {
                const activeTarget = targetColumn || 'Target';
                predictions[activeTarget] = { prediction: res.prediction };
            }
            
            let html = '<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:10px;">';
            Object.entries(predictions).forEach(([target, info]) => {
                html += `
                <div class="row-flex" style="justify-content:space-between; padding: 10px 14px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:8px; border-left:4px solid var(--accent);">
                    <strong style="text-transform:capitalize; font-size:13px; color:var(--muted);">${esc(target)}</strong>
                    <span style="color:var(--accent); font-weight:bold; font-size:15px;">${esc(info.prediction)}</span>
                </div>`;
            });
            html += '</div>';
            listContainer.innerHTML = html;
            
            // Render matched records table if available
            if (matchedContainer && tableHeaderRow && tableBody) {
                const matchedRows = res.matched_rows || [];
                if (matchedRows.length > 0) {
                    tableHeaderRow.innerHTML = '';
                    tableBody.innerHTML = '';
                    
                    // Collect all columns from the first row's dict keys
                    const firstRow = matchedRows[0].row;
                    const cols = Object.keys(firstRow);
                    
                    // Render headers: original columns, then predictions
                    cols.forEach(col => {
                        const th = document.createElement('th');
                        th.textContent = col;
                        tableHeaderRow.appendChild(th);
                    });
                    
                    // Add prediction header columns
                    const firstRowPreds = matchedRows[0].predictions || {};
                    const predTargets = Object.keys(firstRowPreds);
                    predTargets.forEach(target => {
                        const th = document.createElement('th');
                        th.style.color = 'var(--accent)';
                        th.style.fontWeight = 'bold';
                        th.textContent = `${target} (Pred)`;
                        tableHeaderRow.appendChild(th);
                    });
                    
                    // Render body rows
                    matchedRows.forEach(mr => {
                        const tr = document.createElement('tr');
                        
                        // Feature cells
                        cols.forEach(col => {
                            const td = document.createElement('td');
                            const val = mr.row[col];
                            td.textContent = val === null || val === undefined ? '—' : val;
                            tr.appendChild(td);
                        });
                        
                        // Prediction cells
                        predTargets.forEach(target => {
                            const td = document.createElement('td');
                            td.style.color = 'var(--accent)';
                            td.style.fontWeight = 'bold';
                            const info = mr.predictions[target];
                            td.textContent = (info && info.prediction !== undefined) ? info.prediction : '—';
                            tr.appendChild(td);
                        });
                        
                        tableBody.appendChild(tr);
                    });
                    
                    matchedContainer.style.display = 'block';
                } else {
                    matchedContainer.style.display = 'none';
                }
            }
            
            card.style.display = 'block';
        }
        toast('Predictions computed ✓');
    } catch (e) {
        toast('Prediction error: ' + e.message, 'error');
    }
}

async function saveToMySQL() {
    if (!requireActiveProject()) return;
    
    const table = document.getElementById('mysqlSaveTable').value.trim();
    if (!table) {
        toast('Target table name is required', 'warn');
        return;
    }
    
    toast('Exporting dataset to Tomcat MySQL DB...', 'info');
    
    try {
        const previewR = await fetch(`${FASTAPI_API}/projects/${activeProjectId}/download/cleaned`);
        if (!previewR.ok) throw new Error('Could not download cleaned data');
        const csvText = await previewR.text();
        
        const parsed = parseCSVText(csvText, 10000);
        const raw_rows = parsed.data.map(row => parsed.columns.map(c => row[c] || ''));
        
        const r = await fetch(`${SERVLET_API}/db/save`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                table: table,
                columns: parsed.columns,
                rows: raw_rows
            })
        });
        if (!r.ok) throw new Error('Servlet save failed');
        
        toast(`Saved dataset to MySQL table: "${table}" successfully! ✓`);
    } catch (e) {
        toast('Failed to save to MySQL: ' + e.message, 'error');
    }
}

// ── UTILITIES ─────────────────────────────────────
function requireActiveProject() {
    if (!activeProjectId) {
        toast('Please create or select an active project first.', 'warn');
        return false;
    }
    return true;
}

function downloadFile(type) {
    if (!requireActiveProject()) return;
    window.location.href = `${FASTAPI_API}/projects/${activeProjectId}/download/${type}`;
}

function toast(msg, type) {
    const el = document.createElement('div');
    el.className = 'toast' + (type ? ' ' + type : '');
    el.textContent = msg;
    document.getElementById('toasts').appendChild(el);
    setTimeout(() => el.remove(), 3200);
}
