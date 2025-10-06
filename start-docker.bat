@echo off
echo 🐳 Starting Social Media Addiction Assessment System with Docker...
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Build and start the application
echo 🔨 Building Docker image...
docker-compose build

echo 🚀 Starting application...
docker-compose up -d

REM Wait for application to start
echo ⏳ Waiting for application to start...
timeout /t 10 /nobreak >nul

REM Check if application is running
curl -f http://localhost:5000/ >nul 2>&1
if errorlevel 1 (
    echo ❌ Application failed to start. Check logs with: docker-compose logs
    pause
    exit /b 1
) else (
    echo.
    echo ✅ Application is running successfully!
    echo 🌐 Open your browser to: http://localhost:5000
    echo.
    echo 📊 To view logs: docker-compose logs -f
    echo 🛑 To stop: docker-compose down
    echo.
    pause
)
