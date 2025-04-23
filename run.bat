@echo off
echo Starting FastAPI server...
start "FastAPI Server" cmd /k python app.py

echo Starting Signal Generator...
start "Signal Generator" cmd /k utils/signal_generator.py

echo Both scripts started in separate windows.
pause
