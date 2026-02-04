@echo off
REM FinBot Quick Start Script for Windows

setlocal enabledelayedexpansion

echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║         FinBot - Financial Advisor Chatbot Setup             ║
echo ╚═══════════════════════════════════════════════════════════════╝

REM Check Python version
echo.
echo Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python is not installed
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✓ Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo ✓ Virtual environment activated

REM Create .env file
echo.
echo Creating environment configuration...
if not exist ".env" (
    copy .env.example .env
    echo ✓ Created .env file from template
    echo ⚠ Please edit .env file with your configuration
) else (
    echo .env file already exists
)

REM Install dependencies
echo.
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo ✓ Dependencies installed

REM Create necessary directories
echo.
echo Creating project directories...
if not exist "data\documents" mkdir data\documents
if not exist "embeddings" mkdir embeddings
if not exist "logs" mkdir logs
if not exist "checkpoints" mkdir checkpoints
echo ✓ Project directories created

REM Show next steps
echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║                    Setup Complete! 🎉                         ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.
echo Next Steps:
echo.
echo 1. Edit configuration (optional):
echo    notepad .env
echo.
echo 2. Add financial documents:
echo    # Add PDF or TXT files to: data\documents\
echo.
echo 3. Start the backend API (Terminal 1):
echo    venv\Scripts\activate.bat
echo    python -m backend.api
echo.
echo 4. Start the frontend (Terminal 2):
echo    venv\Scripts\activate.bat
echo    streamlit run frontend\app.py
echo.
echo 5. Open in browser:
echo    http://localhost:8501
echo.
echo For Docker deployment:
echo    docker-compose up -d
echo.
echo For detailed documentation, see:
echo    - README.md (Project overview)
echo    - deploy.md (Deployment guide)
echo    - SETUP_GUIDE.md (Configuration options)
echo.
pause
