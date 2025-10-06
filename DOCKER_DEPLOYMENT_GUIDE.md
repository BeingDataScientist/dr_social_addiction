# 🐳 Docker Deployment Guide for Social Media Addiction Assessment System

This guide will help you containerize and deploy your Flask application using Docker.

## 📋 Prerequisites

- Docker installed on your system
- Docker Compose installed
- All project files present

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Build and run the application:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - Open your browser to: http://localhost:5000
   - The application will be running in the container

3. **Stop the application:**
   ```bash
   docker-compose down
   ```

### Option 2: Using Docker Commands

1. **Build the Docker image:**
   ```bash
   docker build -t social-addiction-assessment .
   ```

2. **Run the container:**
   ```bash
   docker run -p 5000:5000 --name social-addiction-app social-addiction-assessment
   ```

3. **Access the application:**
   - Open your browser to: http://localhost:5000

4. **Stop the container:**
   ```bash
   docker stop social-addiction-app
   docker rm social-addiction-app
   ```

## 📁 What's Included in the Container

- ✅ Flask web application
- ✅ ML model (Logistic Regression)
- ✅ HTML templates
- ✅ All Python dependencies
- ✅ CSV data files
- ✅ Health checks

## 🔧 Configuration Options

### Environment Variables

You can customize the application using environment variables:

```bash
# Using docker-compose
docker-compose up -e FLASK_ENV=development

# Using docker run
docker run -p 5000:5000 -e FLASK_ENV=development social-addiction-assessment
```

### Port Configuration

To change the port mapping:

```yaml
# In docker-compose.yml
ports:
  - "8080:5000"  # Maps host port 8080 to container port 5000
```

### Volume Mounts

To persist data and logs:

```yaml
# In docker-compose.yml
volumes:
  - ./logs:/app/logs
  - ./user_responses.csv:/app/user_responses.csv
```

## 🌐 Production Deployment

### 1. AWS EC2 Deployment

1. **Launch EC2 instance:**
   - Choose Ubuntu 20.04 LTS
   - Configure security group (open port 5000)
   - Install Docker and Docker Compose

2. **Deploy application:**
   ```bash
   # Clone your repository
   git clone https://github.com/BeingDataScientist/dr_social_addiction.git
   cd dr_social_addiction
   
   # Build and run
   docker-compose up -d
   ```

3. **Access application:**
   - Use EC2 public IP: http://YOUR_EC2_IP:5000

### 2. AWS ECS Deployment

1. **Create ECS cluster**
2. **Create task definition** using the Dockerfile
3. **Create service** with the task definition
4. **Configure load balancer** if needed

### 3. Google Cloud Run

1. **Build and push to Google Container Registry:**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/social-addiction-assessment
   ```

2. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy --image gcr.io/PROJECT_ID/social-addiction-assessment --platform managed
   ```

## 🔍 Monitoring and Logs

### View Logs

```bash
# Using docker-compose
docker-compose logs -f

# Using docker
docker logs -f social-addiction-app
```

### Health Checks

The container includes health checks:
- **Endpoint**: http://localhost:5000/
- **Interval**: 30 seconds
- **Timeout**: 10 seconds
- **Retries**: 3

### Monitor Container Status

```bash
# Check container status
docker ps

# Check health status
docker inspect social-addiction-app | grep Health -A 10
```

## 🛠️ Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Change port in docker-compose.yml
   ports:
     - "8080:5000"
   ```

2. **Permission issues:**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER .
   ```

3. **Build failures:**
   ```bash
   # Clean build
   docker-compose down
   docker system prune -f
   docker-compose up --build
   ```

4. **Memory issues:**
   ```bash
   # Increase memory limit
   docker run -m 2g -p 5000:5000 social-addiction-assessment
   ```

### Debug Mode

Run in debug mode for troubleshooting:

```bash
# Using docker-compose
docker-compose up -e FLASK_ENV=development

# Using docker run
docker run -p 5000:5000 -e FLASK_ENV=development social-addiction-assessment
```

## 📊 Performance Optimization

### Multi-stage Build (Optional)

For smaller images, you can use multi-stage builds:

```dockerfile
# Build stage
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
EXPOSE 5000
CMD ["python", "app.py"]
```

### Resource Limits

Set resource limits in docker-compose.yml:

```yaml
services:
  social-addiction-app:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

## 🎉 Success!

Your application is now containerized and ready for deployment on any system that supports Docker!

### Next Steps:
1. Test locally with `docker-compose up`
2. Deploy to your preferred cloud platform
3. Set up monitoring and logging
4. Configure CI/CD pipeline for automated deployments
