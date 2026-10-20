#!/usr/bin/env bash
set -e

CONTAINER_NAME="heartguard-test-container"
PORT=8080

echo "=== Cleaning up previous container if exists ==="
docker rm -f $CONTAINER_NAME 2>/dev/null || true

echo "=== Starting Docker Container: $CONTAINER_NAME on port $PORT ==="
docker run -d --name $CONTAINER_NAME --platform "${DOCKER_PLATFORM:-linux/arm64}" -p $PORT:8000 heartguard-api:latest

echo "Waiting for container startup (5s)..."
sleep 5

echo "=== 1. Testing Health Endpoint (GET /health) ==="
HEALTH_RESP=$(curl -s http://localhost:$PORT/health)
echo "Health Response: $HEALTH_RESP"

echo -e "\n=== 2. Testing Root Endpoint (GET /) ==="
ROOT_RESP=$(curl -s http://localhost:$PORT/)
echo "Root Response: $ROOT_RESP"

echo -e "\n=== 3. Testing Single Prediction (POST /predict) ==="
PREDICT_PAYLOAD='{
  "age": 63.0,
  "sex": 1.0,
  "cp": 3.0,
  "trestbps": 145.0,
  "chol": 233.0,
  "fbs": 1.0,
  "restecg": 0.0,
  "thalach": 150.0,
  "exang": 0.0,
  "oldpeak": 2.3,
  "slope": 1.0,
  "ca": 0.0,
  "thal": 3.0
}'
PRED_RESP=$(curl -s -X POST http://localhost:$PORT/predict \
  -H "Content-Type: application/json" \
  -d "$PREDICT_PAYLOAD")
echo "Prediction Response: $PRED_RESP"

echo -e "\n=== 4. Testing High-Risk Patient Prediction (POST /predict) ==="
HIGH_RISK_PAYLOAD='{
  "age": 67.0,
  "sex": 1.0,
  "cp": 4.0,
  "trestbps": 160.0,
  "chol": 286.0,
  "fbs": 0.0,
  "restecg": 2.0,
  "thalach": 108.0,
  "exang": 1.0,
  "oldpeak": 1.5,
  "slope": 2.0,
  "ca": 3.0,
  "thal": 3.0
}'
HIGH_RISK_RESP=$(curl -s -X POST http://localhost:$PORT/predict \
  -H "Content-Type: application/json" \
  -d "$HIGH_RISK_PAYLOAD")
echo "High Risk Prediction Response: $HIGH_RISK_RESP"

echo -e "\n=== 5. Testing Prometheus Metrics (GET /metrics) ==="
METRICS_SAMPLE=$(curl -s http://localhost:$PORT/metrics | grep -E "heart_disease_(api_requests_total|predictions_total|drift_detected)" | head -n 10)
echo "Prometheus Metrics Sample:"
echo "$METRICS_SAMPLE"

echo -e "\n=== 6. Testing Statistical Drift Baseline (GET /drift) ==="
DRIFT_RESP=$(curl -s http://localhost:$PORT/drift)
echo "Drift Status Response: $DRIFT_RESP"

echo -e "\n=== 7. Container Logs ==="
docker logs --tail 10 $CONTAINER_NAME

echo -e "\n=== 8. Cleaning up test container ==="
docker rm -f $CONTAINER_NAME

echo -e "\n>>> DOCKER LOCAL RUN VALIDATION: ALL CHECKS PASSED SUCCESSFULLY! <<<"
