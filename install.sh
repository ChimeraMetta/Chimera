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

sudo bash -c 'cat > /usr/local/bin/chimera <<EOF

IMAGE="ghcr.io/chimerametta/chimera:latest"
MOUNT_DIR="\$(pwd)"
CONTAINER_DIR="/data"

# Pull the image if it's not already present
if ! docker image inspect "\$IMAGE" > /dev/null 2>&1; then
  echo "Pulling \$IMAGE..."
  docker pull "\$IMAGE"
fi

# Run the container with arguments and volume mount
docker run --rm -it -v "\$MOUNT_DIR":"\$CONTAINER_DIR" "\$IMAGE" "\$@"
EOF

chmod +x /usr/local/bin/chimera'