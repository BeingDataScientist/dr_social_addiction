#!/bin/bash

# System Requirements Checker for Social Media Addiction Assessment System

echo "🔍 Checking system requirements..."
echo ""

# Function to check command and version
check_command() {
    local cmd=$1
    local min_version=$2
    
    if command -v $cmd &> /dev/null; then
        local version=$($cmd --version 2>&1 | head -n1)
        echo "✅ $cmd: $version"
        return 0
    else
        echo "❌ $cmd: Not installed"
        return 1
    fi
}

# Function to check Python version
check_python_version() {
    if command -v python3 &> /dev/null; then
        local version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        local major=$(echo $version | cut -d. -f1)
        local minor=$(echo $version | cut -d. -f2)
        
        if [ $major -eq 3 ] && [ $minor -ge 9 ]; then
            echo "✅ Python: $version (Compatible)"
            return 0
        else
            echo "❌ Python: $version (Requires 3.9+)"
            return 1
        fi
    else
        echo "❌ Python: Not installed"
        return 1
    fi
}

# Function to check system resources
check_resources() {
    echo "📊 System Resources:"
    
    # Check RAM
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        local ram=$(free -h | awk '/^Mem:/ {print $2}')
        echo "   RAM: $ram"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        local ram=$(system_profiler SPHardwareDataType | grep "Memory:" | awk '{print $2, $3}')
        echo "   RAM: $ram"
    fi
    
    # Check disk space
    local disk=$(df -h . | awk 'NR==2 {print $4}')
    echo "   Available Disk Space: $disk"
    
    # Check CPU cores
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        local cores=$(nproc)
        echo "   CPU Cores: $cores"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        local cores=$(sysctl -n hw.ncpu)
        echo "   CPU Cores: $cores"
    fi
}

# Function to check network connectivity
check_network() {
    echo "🌐 Network Connectivity:"
    
    if ping -c 1 google.com &> /dev/null; then
        echo "   ✅ Internet: Connected"
    else
        echo "   ❌ Internet: No connection"
        return 1
    fi
    
    if ping -c 1 github.com &> /dev/null; then
        echo "   ✅ GitHub: Accessible"
    else
        echo "   ❌ GitHub: Not accessible"
        return 1
    fi
}

# Function to check ports
check_ports() {
    echo "🔌 Port Availability:"
    
    # Check port 5000
    if lsof -i :5000 &> /dev/null; then
        echo "   ⚠️  Port 5000: In use"
    else
        echo "   ✅ Port 5000: Available"
    fi
    
    # Check port 80
    if lsof -i :80 &> /dev/null; then
        echo "   ⚠️  Port 80: In use"
    else
        echo "   ✅ Port 80: Available"
    fi
}

# Main checks
echo "🖥️  Operating System:"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "   ✅ Linux: $(uname -r)"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "   ✅ macOS: $(sw_vers -productVersion)"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "   ✅ Windows: $(uname -r)"
else
    echo "   ⚠️  Unknown OS: $OSTYPE"
fi

echo ""
echo "🔧 Required Software:"

# Check Python
python_ok=false
if check_python_version; then
    python_ok=true
fi

# Check Git
git_ok=false
if check_command git; then
    git_ok=true
fi

# Check Docker (optional)
docker_ok=false
if check_command docker; then
    docker_ok=true
    if docker info &> /dev/null; then
        echo "   ✅ Docker: Running"
    else
        echo "   ⚠️  Docker: Installed but not running"
    fi
fi

echo ""
check_resources
echo ""
check_network
echo ""
check_ports

echo ""
echo "📋 Installation Recommendations:"

if [ "$python_ok" = true ] && [ "$git_ok" = true ]; then
    echo "✅ System ready for Python installation"
    echo "   Run: pip install -r requirements.txt && python app.py"
fi

if [ "$docker_ok" = true ]; then
    echo "✅ System ready for Docker installation"
    echo "   Run: docker-compose up --build"
else
    echo "💡 Consider installing Docker for easier deployment"
    echo "   Visit: https://www.docker.com/get-started"
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Choose installation method (Python or Docker)"
echo "2. Run the appropriate installation script"
echo "3. Access application at http://localhost:5000"

echo ""
echo "📚 For detailed instructions, see INSTALLATION_GUIDE.md"
