#!/bin/bash

# Social Media Addiction Assessment System - Installation Script
# For Linux and macOS systems

echo "🚀 Installing Social Media Addiction Assessment System..."
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "❌ Please don't run this script as root. Run as regular user."
    exit 1
fi

# Detect operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "❌ Unsupported operating system: $OSTYPE"
    exit 1
fi

echo "📋 Detected OS: $OS"
echo ""

# Function to install Docker on Linux
install_docker_linux() {
    echo "🐳 Installing Docker on Linux..."
    
    # Update package index
    sudo apt update
    
    # Install Docker
    sudo apt install -y docker.io docker-compose
    
    # Start Docker service
    sudo systemctl start docker
    sudo systemctl enable docker
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    echo "✅ Docker installed successfully!"
    echo "⚠️  Please logout and login again for group changes to take effect."
}

# Function to install Docker on macOS
install_docker_macos() {
    echo "🐳 Installing Docker on macOS..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "📦 Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install Docker Desktop
    brew install --cask docker
    
    echo "✅ Docker Desktop installed!"
    echo "⚠️  Please start Docker Desktop from Applications folder."
}

# Function to install Python dependencies
install_python_deps() {
    echo "🐍 Installing Python dependencies..."
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
        exit 1
    fi
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    
    echo "✅ Python dependencies installed!"
}

# Function to download and setup application
setup_application() {
    echo "📥 Setting up application..."
    
    # Check if git is installed
    if ! command -v git &> /dev/null; then
        echo "❌ Git is not installed. Please install Git first."
        exit 1
    fi
    
    # Clone repository if not already present
    if [ ! -d "dr_social_addiction" ]; then
        git clone https://github.com/BeingDataScientist/dr_social_addiction.git
    fi
    
    cd dr_social_addiction
    
    echo "✅ Application setup complete!"
}

# Function to start application with Docker
start_docker() {
    echo "🚀 Starting application with Docker..."
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        echo "❌ Docker is not running. Please start Docker first."
        exit 1
    fi
    
    # Build and start
    docker-compose up --build -d
    
    # Wait for application to start
    echo "⏳ Waiting for application to start..."
    sleep 15
    
    # Check if application is running
    if curl -f http://localhost:5000/ &> /dev/null; then
        echo "✅ Application is running successfully!"
        echo "🌐 Open your browser to: http://localhost:5000"
    else
        echo "❌ Application failed to start. Check logs with: docker-compose logs"
        exit 1
    fi
}

# Function to start application with Python
start_python() {
    echo "🚀 Starting application with Python..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start application
    python app.py &
    
    # Wait for application to start
    echo "⏳ Waiting for application to start..."
    sleep 10
    
    # Check if application is running
    if curl -f http://localhost:5000/ &> /dev/null; then
        echo "✅ Application is running successfully!"
        echo "🌐 Open your browser to: http://localhost:5000"
    else
        echo "❌ Application failed to start. Check the output above for errors."
        exit 1
    fi
}

# Main installation flow
echo "🎯 Choose installation method:"
echo "1) Docker (Recommended)"
echo "2) Python Direct"
echo ""
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        echo "🐳 Installing with Docker..."
        
        # Install Docker based on OS
        if [ "$OS" == "linux" ]; then
            install_docker_linux
        elif [ "$OS" == "macos" ]; then
            install_docker_macos
        fi
        
        # Setup application
        setup_application
        
        # Start with Docker
        start_docker
        ;;
    2)
        echo "🐍 Installing with Python..."
        
        # Install Python dependencies
        install_python_deps
        
        # Setup application
        setup_application
        
        # Start with Python
        start_python
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📊 Useful commands:"
echo "  - View logs: docker-compose logs -f (Docker) or check terminal (Python)"
echo "  - Stop application: docker-compose down (Docker) or Ctrl+C (Python)"
echo "  - Restart: docker-compose restart (Docker) or run python app.py (Python)"
echo ""
echo "📚 For more information, see INSTALLATION_GUIDE.md"
