@echo off
echo 🚀 Installing Social Media Addiction Assessment System...
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ❌ Please don't run this script as administrator. Run as regular user.
    pause
    exit /b 1
)

echo 📋 Detected OS: Windows
echo.

REM Function to check if command exists
:check_command
where %1 >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ %1 is installed
) else (
    echo ❌ %1 is not installed
)
goto :eof

REM Check prerequisites
echo 🔍 Checking prerequisites...
call :check_command python
call :check_command git
call :check_command docker
echo.

REM Main menu
echo 🎯 Choose installation method:
echo 1) Docker (Recommended)
echo 2) Python Direct
echo.
set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" goto docker_install
if "%choice%"=="2" goto python_install
echo ❌ Invalid choice. Please run the script again.
pause
exit /b 1

:docker_install
echo 🐳 Installing with Docker...

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed.
    echo 📥 Please download and install Docker Desktop from:
    echo    https://www.docker.com/products/docker-desktop/
    echo.
    echo After installation, restart this script.
    pause
    exit /b 1
)

echo ✅ Docker is installed
echo.

REM Clone repository if not present
if not exist "dr_social_addiction" (
    echo 📥 Cloning repository...
    git clone https://github.com/BeingDataScientist/dr_social_addiction.git
    if errorlevel 1 (
        echo ❌ Failed to clone repository. Check your internet connection.
        pause
        exit /b 1
    )
)

cd dr_social_addiction

REM Start with Docker
echo 🚀 Starting application with Docker...
docker-compose up --build -d

if errorlevel 1 (
    echo ❌ Failed to start application with Docker.
    echo Check if Docker Desktop is running.
    pause
    exit /b 1
)

echo ⏳ Waiting for application to start...
timeout /t 15 /nobreak >nul

REM Check if application is running
curl -f http://localhost:5000/ >nul 2>&1
if errorlevel 1 (
    echo ❌ Application failed to start. Check logs with: docker-compose logs
    pause
    exit /b 1
) else (
    echo ✅ Application is running successfully!
    echo 🌐 Open your browser to: http://localhost:5000
)
goto :success

:python_install
echo 🐍 Installing with Python...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed.
    echo 📥 Please download and install Python from:
    echo    https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo ✅ Python is installed
echo.

REM Clone repository if not present
if not exist "dr_social_addiction" (
    echo 📥 Cloning repository...
    git clone https://github.com/BeingDataScientist/dr_social_addiction.git
    if errorlevel 1 (
        echo ❌ Failed to clone repository. Check your internet connection.
        pause
        exit /b 1
    )
)

cd dr_social_addiction

REM Install dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies.
    pause
    exit /b 1
)

REM Start application
echo 🚀 Starting application with Python...
start /b python app.py

echo ⏳ Waiting for application to start...
timeout /t 10 /nobreak >nul

REM Check if application is running
curl -f http://localhost:5000/ >nul 2>&1
if errorlevel 1 (
    echo ❌ Application failed to start. Check the output above for errors.
    pause
    exit /b 1
) else (
    echo ✅ Application is running successfully!
    echo 🌐 Open your browser to: http://localhost:5000
)

:success
echo.
echo 🎉 Installation complete!
echo.
echo 📊 Useful commands:
echo   - View logs: docker-compose logs -f (Docker) or check terminal (Python)
echo   - Stop application: docker-compose down (Docker) or Ctrl+C (Python)
echo   - Restart: docker-compose restart (Docker) or run python app.py (Python)
echo.
echo 📚 For more information, see INSTALLATION_GUIDE.md
echo.
pause
