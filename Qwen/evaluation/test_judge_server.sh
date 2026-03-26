#!/bin/bash
# Test script to verify local judge server is accessible

set -e

# Default server URL
SERVER_URL="${1:-http://localhost:8000}"

echo "Testing judge server at: $SERVER_URL"
echo ""

# Test 1: Check /v1/models endpoint
echo "Test 1: Checking /v1/models endpoint..."
if curl -s -f "${SERVER_URL}/v1/models" > /dev/null 2>&1; then
    echo "✓ Server is responding"
    curl -s "${SERVER_URL}/v1/models" | jq '.' 2>/dev/null || echo "(Install jq for formatted output)"
else
    echo "✗ Server is not responding at $SERVER_URL"
    echo "  Make sure your vLLM server is running:"
    echo "  python Qwen/evaluation/judge_server.py --model-path /path/to/model --port 8600"
    exit 1
fi

echo ""
echo "Test 2: Testing /judge..."
response=$(curl -s "${SERVER_URL}/judge" \
    -H "Content-Type: application/json" \
    -d '{
        "question": "What is 2 + 2?",
        "reference": "4",
        "prediction": "The answer is 4."
    }')

if echo "$response" | grep -q '"verdict"'; then
    echo "✓ Judge endpoint working"
    echo "$response" | jq '.' 2>/dev/null || echo "$response"
else
    echo "✗ Judge endpoint failed"
    echo "$response"
    exit 1
fi

echo ""
echo "✓ Judge server is ready for evaluation!"
