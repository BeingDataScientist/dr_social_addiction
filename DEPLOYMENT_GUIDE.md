# Social Media Addiction Assessment - AWS Deployment Guide

## 🎯 Overview
This guide covers multiple ways to deploy your Social Media Addiction Assessment system on AWS.

## 📋 Prerequisites
- AWS Account
- AWS CLI installed
- Your project files ready
- Domain name (optional)

---

## 🚀 Option 1: AWS Elastic Beanstalk (Recommended for Beginners)

### Step 1: Prepare Your Application
```bash
# Create .ebextensions directory
mkdir .ebextensions

# Create configuration file
cat > .ebextensions/01_packages.config << EOF
packages:
  yum:
    git: []
EOF

# Create Procfile
echo "web: python app.py" > Procfile

# Create .ebignore (optional)
echo "*.pyc
__pycache__/
*.log
mlruns/
TRAIN_Analysis/
*.md" > .ebignore
```

### Step 2: Install EB CLI
```bash
pip install awsebcli
```

### Step 3: Deploy
```bash
# Initialize EB
eb init

# Create environment
eb create production

# Deploy
eb deploy
```

### Step 4: Configure Environment Variables
```bash
eb setenv FLASK_ENV=production
eb setenv PORT=5000
```

---

## 🖥️ Option 2: AWS EC2 (Full Control)

### Step 1: Launch EC2 Instance
- **AMI**: Ubuntu Server 20.04 LTS
- **Instance Type**: t3.medium (2 vCPU, 4 GB RAM)
- **Storage**: 20 GB GP3
- **Security Group**: Allow HTTP (80), HTTPS (443), SSH (22)

### Step 2: Connect and Setup
```bash
# Connect to instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3-pip python3-venv nginx git -y

# Create application directory
sudo mkdir -p /var/www/your-app
sudo chown ubuntu:ubuntu /var/www/your-app
cd /var/www/your-app
```

### Step 3: Deploy Application
```bash
# Clone your repository
git clone https://github.com/your-username/your-repo.git .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install gunicorn

# Test the application
python app.py
```

### Step 4: Configure Gunicorn
```bash
# Create Gunicorn configuration
cat > gunicorn.conf.py << EOF
bind = "127.0.0.1:8000"
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True
EOF

# Create systemd service
sudo nano /etc/systemd/system/your-app.service
```

### Step 5: Create Systemd Service
```ini
[Unit]
Description=Social Media Addiction Assessment App
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/var/www/your-app
Environment="PATH=/var/www/your-app/venv/bin"
ExecStart=/var/www/your-app/venv/bin/gunicorn --config gunicorn.conf.py app:app
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always

[Install]
WantedBy=multi-user.target
```

### Step 6: Configure Nginx
```bash
# Create Nginx configuration
sudo nano /etc/nginx/sites-available/your-app
```

```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /var/www/your-app/static;
    }
}
```

### Step 7: Enable and Start Services
```bash
# Enable Nginx site
sudo ln -s /etc/nginx/sites-available/your-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Enable and start your app
sudo systemctl enable your-app
sudo systemctl start your-app
sudo systemctl status your-app
```

---

## 🐳 Option 3: AWS App Runner (Containerized)

### Step 1: Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
```

### Step 2: Create apprunner.yaml
```yaml
version: 1.0
runtime: python3
build:
  commands:
    build:
      - echo "Build started on `date`"
      - pip install -r requirements.txt
run:
  runtime-version: 3.9.16
  command: gunicorn -w 4 -b 0.0.0.0:8000 app:app
  network:
    port: 8000
    env: PORT
  env:
    - name: PORT
      value: "8000"
```

### Step 3: Deploy to App Runner
1. Go to AWS App Runner console
2. Create service
3. Choose "Container registry" or "Source code repository"
4. Configure service settings
5. Deploy

---

## 🔧 Production Optimizations

### 1. Environment Variables
```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
export SECRET_KEY=your-secret-key
export DATABASE_URL=your-database-url
```

### 2. Security Headers
```python
# Add to app.py
from flask_talisman import Talisman

app = Flask(__name__)
Talisman(app, force_https=True)
```

### 3. Database Integration
```python
# For production, consider using RDS
import psycopg2
from sqlalchemy import create_engine

# Replace CSV storage with database
engine = create_engine('postgresql://user:pass@host:port/db')
```

### 4. Monitoring and Logging
```python
import logging
from flask import request
import time

@app.before_request
def log_request_info():
    app.logger.info(f'Request: {request.method} {request.url}')

@app.after_request
def log_response_info(response):
    app.logger.info(f'Response: {response.status_code}')
    return response
```

---

## 💰 Cost Estimation

### Elastic Beanstalk
- **t3.micro**: ~$8.50/month
- **t3.small**: ~$17/month
- **t3.medium**: ~$34/month

### EC2
- **t3.micro**: ~$8.50/month
- **t3.small**: ~$17/month
- **t3.medium**: ~$34/month
- **Storage**: ~$2/month (20GB)

### App Runner
- **0.25 vCPU, 0.5 GB**: ~$25/month
- **0.5 vCPU, 1 GB**: ~$50/month

---

## 🔒 Security Best Practices

1. **Use HTTPS**: Configure SSL certificate
2. **Environment Variables**: Store secrets securely
3. **Security Groups**: Restrict access
4. **Regular Updates**: Keep system updated
5. **Monitoring**: Set up CloudWatch alarms
6. **Backup**: Regular data backups

---

## 📊 Monitoring Setup

### CloudWatch Metrics
- CPU utilization
- Memory usage
- Request count
- Error rate
- Response time

### Health Checks
```python
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })
```

---

## 🚀 Quick Start Commands

### For EC2 Deployment:
```bash
# 1. Launch EC2 instance
# 2. Connect and run:
sudo apt update && sudo apt install python3-pip nginx git -y
git clone your-repo-url
cd your-project
pip3 install -r requirements.txt gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### For Elastic Beanstalk:
```bash
pip install awsebcli
eb init
eb create production
eb deploy
```

---

## 📞 Support

For deployment issues:
1. Check AWS documentation
2. Review application logs
3. Test locally first
4. Use AWS support if needed

---

*Last updated: October 2025*
