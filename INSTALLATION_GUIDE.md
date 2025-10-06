# 🚀 Installation Guide for Social Media Addiction Assessment System

This guide provides step-by-step instructions for installing the system on various target systems.

## 📋 Prerequisites

- Target system with internet connection
- Administrative/sudo access
- Basic command line knowledge

## 🎯 Installation Options

### Option 1: Docker Installation (Recommended)
### Option 2: Direct Python Installation
### Option 3: Cloud Platform Deployment

---

## 🐳 Option 1: Docker Installation (Recommended)

### For Windows Systems:

#### Step 1: Install Docker Desktop
1. **Download Docker Desktop:**
   - Go to: https://www.docker.com/products/docker-desktop/
   - Click "Download for Windows"
   - Run the installer as Administrator

2. **System Requirements:**
   - Windows 10/11 64-bit
   - WSL 2 enabled
   - Virtualization enabled in BIOS

3. **Installation:**
   ```powershell
   # Run installer and follow wizard
   # Restart computer when prompted
   ```

#### Step 2: Download and Run Application
1. **Download the project:**
   ```bash
   git clone https://github.com/BeingDataScientist/dr_social_addiction.git
   cd dr_social_addiction
   ```

2. **Start the application:**
   ```bash
   # Option A: Using startup script
   start-docker.bat
   
   # Option B: Using docker-compose
   docker-compose up --build
   ```

3. **Access the application:**
   - Open browser to: http://localhost:5000

### For Linux Systems:

#### Step 1: Install Docker
```bash
# Update package index
sudo apt update

# Install Docker
sudo apt install docker.io docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again
```

#### Step 2: Download and Run Application
```bash
# Download project
git clone https://github.com/BeingDataScientist/dr_social_addiction.git
cd dr_social_addiction

# Start application
./start-docker.sh
# OR
docker-compose up --build
```

### For macOS Systems:

#### Step 1: Install Docker Desktop
1. **Download from:** https://www.docker.com/products/docker-desktop/
2. **Install and start Docker Desktop**
3. **Verify installation:**
   ```bash
   docker --version
   ```

#### Step 2: Download and Run Application
```bash
# Download project
git clone https://github.com/BeingDataScientist/dr_social_addiction.git
cd dr_social_addiction

# Start application
./start-docker.sh
# OR
docker-compose up --build
```

---

## 🐍 Option 2: Direct Python Installation

### For Windows Systems:

#### Step 1: Install Python
1. **Download Python 3.9+ from:** https://www.python.org/downloads/
2. **Install with "Add to PATH" option checked**
3. **Verify installation:**
   ```cmd
   python --version
   pip --version
   ```

#### Step 2: Install Application
```cmd
# Download project
git clone https://github.com/BeingDataScientist/dr_social_addiction.git
cd dr_social_addiction

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

### For Linux Systems:

#### Step 1: Install Python and Dependencies
```bash
# Update package index
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Run Application
```bash
# Run application
python app.py
```

### For macOS Systems:

#### Step 1: Install Python
```bash
# Using Homebrew (recommended)
brew install python3

# Or download from python.org
```

#### Step 2: Install Application
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

---

## ☁️ Option 3: Cloud Platform Deployment

### AWS EC2 Deployment:

#### Step 1: Launch EC2 Instance
1. **Go to AWS Console → EC2**
2. **Launch Instance:**
   - AMI: Ubuntu Server 20.04 LTS
   - Instance Type: t2.micro (free tier)
   - Security Group: Allow HTTP (port 80) and Custom TCP (port 5000)

#### Step 2: Connect and Install
```bash
# Connect via SSH
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install docker.io docker-compose
sudo systemctl start docker
sudo usermod -aG docker ubuntu

# Download and run application
git clone https://github.com/BeingDataScientist/dr_social_addiction.git
cd dr_social_addiction
docker-compose up -d
```

#### Step 3: Access Application
- **Public IP:** http://YOUR-EC2-IP:5000

### Google Cloud Platform:

#### Step 1: Create VM Instance
```bash
# Using gcloud CLI
gcloud compute instances create social-addiction-app \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --machine-type=e2-micro \
    --zone=us-central1-a \
    --tags=http-server,https-server
```

#### Step 2: Deploy Application
```bash
# SSH into instance
gcloud compute ssh social-addiction-app --zone=us-central1-a

# Install Docker and deploy
sudo apt update
sudo apt install docker.io docker-compose
git clone https://github.com/BeingDataScientist/dr_social_addiction.git
cd dr_social_addiction
sudo docker-compose up -d
```

### Azure Deployment:

#### Step 1: Create Virtual Machine
1. **Go to Azure Portal → Virtual Machines**
2. **Create VM:**
   - Image: Ubuntu Server 20.04 LTS
   - Size: B1s (free tier)
   - Allow SSH and HTTP ports

#### Step 2: Deploy Application
```bash
# SSH into VM
ssh azureuser@your-vm-ip

# Install Docker and deploy
sudo apt update
sudo apt install docker.io docker-compose
git clone https://github.com/BeingDataScientist/dr_social_addiction.git
cd dr_social_addiction
sudo docker-compose up -d
```

---

## 🔧 Post-Installation Configuration

### 1. Configure Firewall (Linux/macOS)
```bash
# Allow port 5000
sudo ufw allow 5000
sudo ufw enable
```

### 2. Set Up Auto-Start (Linux)
```bash
# Create systemd service
sudo nano /etc/systemd/system/social-addiction.service

# Add this content:
[Unit]
Description=Social Addiction Assessment System
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/path/to/dr_social_addiction
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down

[Install]
WantedBy=multi-user.target

# Enable and start service
sudo systemctl enable social-addiction.service
sudo systemctl start social-addiction.service
```

### 3. Configure Reverse Proxy (Optional)
```bash
# Install Nginx
sudo apt install nginx

# Configure Nginx
sudo nano /etc/nginx/sites-available/social-addiction

# Add configuration:
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/social-addiction /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 🛠️ Troubleshooting

### Common Issues:

1. **Port 5000 already in use:**
   ```bash
   # Change port in docker-compose.yml
   ports:
     - "8080:5000"
   ```

2. **Permission denied:**
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER .
   ```

3. **Docker not found:**
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   ```

4. **Python dependencies issues:**
   ```bash
   # Use virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

### Health Checks:

```bash
# Check if application is running
curl http://localhost:5000/

# Check Docker containers
docker ps

# Check logs
docker-compose logs -f
```

---

## 📊 System Requirements

### Minimum Requirements:
- **CPU:** 1 core
- **RAM:** 1 GB
- **Storage:** 2 GB free space
- **OS:** Windows 10+, Ubuntu 18.04+, macOS 10.14+

### Recommended Requirements:
- **CPU:** 2 cores
- **RAM:** 2 GB
- **Storage:** 5 GB free space
- **OS:** Latest versions

---

## 🎉 Success!

After successful installation:
- ✅ Application running on http://localhost:5000
- ✅ All ML models loaded
- ✅ Web interface accessible
- ✅ Data persistence configured

### Next Steps:
1. Test the application functionality
2. Configure backup procedures
3. Set up monitoring
4. Plan for scaling if needed

**Your Social Media Addiction Assessment System is now ready for use!** 🚀
