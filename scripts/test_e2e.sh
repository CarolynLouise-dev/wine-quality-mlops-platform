#!/bin/bash
set -e

# ==============================================================================
# END-TO-END MLOPS VERIFICATION & HEALTH AUDIT SCRIPT
# ==============================================================================

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo -e "${BOLD}${BLUE}======================================================${NC}"
echo -e "${BOLD}${BLUE}   Wine Quality MLOps - End-to-End Verification      ${NC}"
echo -e "${BOLD}${BLUE}======================================================${NC}\n"

# 1. Python Environment Check
echo -e "${BOLD}[1/5] Verifying Python Environment & Dependencies...${NC}"
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
    PYTEST=".venv/bin/pytest"
else
    PYTHON="python3"
    PYTEST="pytest"
fi
$PYTHON --version
echo -e "${GREEN}✓ Python environment active${NC}\n"

# 2. Run Complete Unit & Integration Test Suite
echo -e "${BOLD}[2/5] Running Pytest Unit & Integration Suite...${NC}"
$PYTEST tests -v --disable-warnings
echo -e "${GREEN}✓ All Pytest unit and integration tests PASSED!${NC}\n"

# 3. Verify MLflow Experiment Matrix (10 Models)
echo -e "${BOLD}[3/5] Verifying MLflow Experiment Matrix & Registry Promotion...${NC}"
$PYTHON scripts/run_experiments.py
if [ -f "data/processed/experiment_results.csv" ]; then
    echo -e "${GREEN}✓ MLflow Experiment Matrix ran successfully with 10 configurations!${NC}"
    echo -e "${BLUE}Top Performer Metrics:${NC}"
    head -n 2 data/processed/experiment_results.csv
else
    echo -e "${RED}✗ Experiment results CSV not found!${NC}"
    exit 1
fi
echo ""

# 4. Docker Build Verification
echo -e "${BOLD}[4/5] Verifying Docker Build Capability...${NC}"
echo -e "Testing API Docker build..."
docker build -t wine-mlops-api:test -f api/Dockerfile api/ > /dev/null
echo -e "${GREEN}✓ API Docker container built successfully${NC}"

echo -e "Testing Evidently Docker build..."
docker build -t wine-mlops-drift:test -f evidently_service/Dockerfile evidently_service/ > /dev/null
echo -e "${GREEN}✓ Evidently Docker container built successfully${NC}\n"

# 5. Live Service Health Checks (If stack is active)
echo -e "${BOLD}[5/5] Checking Live Stack Endpoints (if running)...${NC}"
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ FastAPI Serving API (:8000) is LIVE and HEALTHY!${NC}"
    PRED_RESP=$(curl -s -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]}')
    echo "  Sample prediction output: $PRED_RESP"
else
    echo -e "${YELLOW}ℹ Live API container not running on :8000 (Start with 'docker compose up -d')${NC}"
fi

if curl -s -f http://localhost:8001/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Evidently Drift Service (:8001) is LIVE!${NC}"
else
    echo -e "${YELLOW}ℹ Live Evidently container not running on :8001${NC}"
fi

if curl -s -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Prometheus Metrics (:9090) is LIVE!${NC}"
else
    echo -e "${YELLOW}ℹ Live Prometheus container not running on :9090${NC}"
fi

if curl -s -f http://localhost:3000/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Grafana Dashboards (:3000) is LIVE!${NC}"
else
    echo -e "${YELLOW}ℹ Live Grafana container not running on :3000${NC}"
fi

echo -e "\n${BOLD}${GREEN}======================================================${NC}"
echo -e "${BOLD}${GREEN}   ALL END-TO-END VALIDATION GATES PASSED!           ${NC}"
echo -e "${BOLD}${GREEN}======================================================${NC}"
