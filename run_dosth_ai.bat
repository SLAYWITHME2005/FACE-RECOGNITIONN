@echo off
title DOSTH AI - Live Elderly Facial Expression Assistant
cd /d "%~dp0"
echo ==========================================================
echo    DOSTH AI - Live Facial Expression Recognition System
echo ==========================================================
echo  Starting web application on http://localhost:8000 ...
echo  Keep this terminal window open while using the app.
echo ==========================================================
.\venv\Scripts\python scripts\run_fer_web_ui.py
pause
