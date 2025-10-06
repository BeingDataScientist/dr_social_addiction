# 🐳 Docker Setup Guide for Windows

## 📋 Prerequisites Installation

### Step 1: Install Docker Desktop for Windows

1. **Download Docker Desktop:**
   - Go to: https://www.docker.com/products/docker-desktop/
   - Click "Download for Windows"
   - Run the installer

2. **System Requirements:**
   - Windows 10 64-bit: Pro, Enterprise, or Education (Build 15063 or later)
   - WSL 2 feature enabled
   - Virtualization enabled in BIOS

3. **Installation Steps:**
   - Run the installer as Administrator
   - Follow the installation wizard
   - Restart your computer when prompted

4. **Verify Installation:**
   ```bash
   docker --version
   docker-compose --version
   ```

### Step 2: Enable WSL 2 (if not already enabled)

1. **Open PowerShell as Administrator**
2. **Run these commands:**
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```
3. **Restart your computer**
4. **Download and install WSL 2 Linux kernel update:**
   - Go to: https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi
   - Install the update

## 🚀 Quick Start After Installation

Once Docker is installed, you can use these commands:

### Option 1: Using Docker Compose (Recommended)
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# Stop
docker-compose down
```

### Option 2: Using Start Scripts
```bash
# Windows
start-docker.bat

# Linux/Mac
./start-docker.sh
```

## 🔧 Troubleshooting

### Common Issues:

1. **"Docker is not recognized":**
   - Restart your computer after installation
   - Check if Docker Desktop is running
   - Add Docker to PATH environment variable

2. **WSL 2 issues:**
   - Update Windows to latest version
   - Enable Hyper-V feature
   - Check BIOS virtualization settings

3. **Permission issues:**
   - Run PowerShell as Administrator
   - Check Docker Desktop settings

## 📊 What You'll Get

After successful setup:
- ✅ Containerized Flask application
- ✅ All dependencies included
- ✅ Easy deployment to any system
- ✅ Consistent environment across platforms
- ✅ Easy scaling and management

## 🎯 Next Steps

1. Install Docker Desktop
2. Run `docker-compose up --build`
3. Access application at http://localhost:5000
4. Deploy to cloud platforms (AWS, Azure, GCP)
