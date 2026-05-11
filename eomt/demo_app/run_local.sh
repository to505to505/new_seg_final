#!/bin/bash
# ==============================================================================
# Run EoMT Segmentation Demo Locally
# ==============================================================================
# Usage:
#   ./run_local.sh
#   ./run_local.sh --port 8502
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=${1:-8502}
if [[ "$1" == "--port" ]]; then
    PORT=${2:-8502}
fi

GREEN='\033[0;32m'
NC='\033[0m'

echo "==========================================================="
echo "  🫀 EoMT Coronary Artery Segmentation Demo"
echo "==========================================================="
echo ""

LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP="localhost"
fi

echo "🚀 Starting Streamlit server..."
echo ""
echo -e "📍 Access URLs:"
echo "   Local:   http://localhost:$PORT"
echo "   Network: http://$LOCAL_IP:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo "==========================================================="
echo ""

streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port "$PORT" \
    --server.headless true
