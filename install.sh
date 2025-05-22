#!/bin/bash
# =============================================================================
# Chimera Installation Script
# =============================================================================
# This script installs the Chimera CLI tool by creating an executable wrapper
# in /usr/local/bin/chimera. The wrapper script:
#
# 1. Uses the official Chimera Docker image (ghcr.io/chimerametta/chimera:latest)
# 2. Automatically pulls the Docker image if not present locally
# 3. Mounts the current working directory to /data in the container
# 4. Executes Chimera with any provided arguments
#
# Usage:
#   sudo ./install.sh
#
# After installation, Chimera can be run from any directory:
#   chimera [arguments]
#
# Requirements:
#   - Docker must be installed and running
#   - Sudo privileges for installation
# =============================================================================

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo "\n${MAGENTA}[STEP]${NC} $1"
}

# Check if running with sudo
if [ "$(id -u)" != "0" ]; then
    log_error "Please run this script with sudo"
    exit 1
fi

# Check if Docker is installed
if ! which docker >/dev/null 2>&1; then
    log_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    log_error "Docker daemon is not running. Please start Docker first."
    exit 1
fi

log_step "Starting Chimera CLI installation..."

log_info "Creating executable wrapper in /usr/local/bin/chimera"
cat > /usr/local/bin/chimera << "EOF"
#!/bin/bash

IMAGE="ghcr.io/chimerametta/chimera:latest"
MOUNT_DIR="$(pwd)"
CONTAINER_DIR="/data"

# Pull the image if it's not already present
if ! docker image inspect "${IMAGE}" > /dev/null 2>&1; then
  echo "Pulling ${IMAGE}..."
  docker pull "${IMAGE}"
fi

# Run the container with arguments and volume mount
docker run --rm -it -v "${MOUNT_DIR}":"${CONTAINER_DIR}" "${IMAGE}" "$@"
EOF

if [ $? -eq 0 ]; then
    log_success "Wrapper script created successfully"
else
    log_error "Failed to create wrapper script"
    exit 1
fi

log_info "Setting executable permissions"
chmod +x /usr/local/bin/chimera

if [ $? -eq 0 ]; then
    log_success "Permissions set successfully"
else
    log_error "Failed to set permissions"
    exit 1
fi

log_step "Verifying installation..."
if command -v chimera >/dev/null 2>&1; then
    log_success "Chimera CLI installed successfully!"
    log_info "You can now use the 'chimera' command from any directory"
else
    log_error "Installation verification failed"
    exit 1
fi

log_info "Testing Docker image pull..."
if docker pull ghcr.io/chimerametta/chimera:latest >/dev/null 2>&1; then
    log_success "Docker image pull test successful"
else
    log_warning "Docker image pull test failed - this is normal if you don't have access to the image yet"
fi

echo "\n${CYAN}Installation complete!${NC}"
echo "Try running: ${GREEN}chimera --help${NC} to get started"