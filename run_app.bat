@echo off
title Algo - Login + Movement service
cd /d D:\trade_Prash
call .venv\Scripts\activate.bat
echo === Step 1: Upstox login (token expires daily) ===
python -m src.broker.auth
if errorlevel 1 (
  echo Login failed. Fix the error above, or run: python -m src.movement --mock
  pause
  exit /b 1
)
echo === Step 2: Starting console on http://127.0.0.1:8510 ===
start "Algo - Console" cmd /k "cd /d D:\trade_Prash && call .venv\Scripts\activate.bat && python -m src.console"
echo === Step 3: Movement service on http://127.0.0.1:8503 ===
python -m src.movement
pause
