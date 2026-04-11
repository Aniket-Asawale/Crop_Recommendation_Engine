@echo off
setlocal EnableDelayedExpansion
title Crop Recommendation Engine - Start API and Dashboard
color 0A

echo ============================================================
echo   Crop Recommendation Engine - Starting Services
echo ============================================================
echo.
echo   Components:
echo     - FastAPI API Server   ^(port 8001^)
echo     - Streamlit Dashboard  ^(port 8501^)
echo.

cd /d "%~dp0"

REM --- 0. Check if API Gateway is running ---
echo [0/4] Checking API Gateway availability...
netstat -ano | findstr ":8080 " | findstr "LISTENING" >nul
if %ERRORLEVEL% NEQ 0 (
    echo   [WARN] API Gateway not detected on port 8080
    echo         Make sure to run 'start_all.bat' from root or start it separately
)
echo.

REM --- 1. Cleanup stale processes ---
echo [1/4] Cleaning up any stale processes...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8001 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8501 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
timeout /t 1 /nobreak >nul
echo   Done.
echo.

REM --- 2. Start API Server ---
echo [2/4] Starting FastAPI API Server ^(port 8001^)...
if exist "venv\Scripts\python.exe" (
    start "Crop API" cmd /k "venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8001 --reload"
    echo   Started in new window.
) else (
    echo [ERROR] No venv found. Install with: python -m venv venv && venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
timeout /t 3 /nobreak >nul
echo.

REM --- 3. Start Dashboard ---
echo [3/4] Starting Streamlit Dashboard ^(port 8501^)...
if exist "venv\Scripts\python.exe" (
    start "Crop Dashboard" cmd /k "venv\Scripts\streamlit run app.py -- --server.port=8501"
    echo   Started in new window.
) else (
    echo [ERROR] No venv found.
    pause
    exit /b 1
)
timeout /t 3 /nobreak >nul
echo.

REM --- 4. Open browser windows ---
echo [4/4] Opening browser windows...
echo.
echo   Local URLs:
echo     API Docs:   http://127.0.0.1:8001/docs
echo     Dashboard:  http://127.0.0.1:8501
echo.
echo   Public URLs ^(via Cloudflare^):
echo     API:        https://crop-dashboard.agroaiapp.me
echo     Dashboard:  https://crop-dashboard.agroaiapp.me
echo.

timeout /t 2 /nobreak >nul
start http://127.0.0.1:8501
timeout /t 1 /nobreak >nul
start https://crop-dashboard.agroaiapp.me
timeout /t 1 /nobreak >nul

echo.
echo ===========================================================
echo   Crop Services running in separate windows
echo ===========================================================
echo   Press Ctrl+C in either window to stop that service
echo ===========================================================
echo.

endlocal
