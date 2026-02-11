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
    echo "  vllm serve qwen2.5-72b-instruct --port 8000"
    exit 1
fi

echo ""
echo "Test 2: Testing chat completions..."
response=$(curl -s "${SERVER_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5-72b-instruct",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10
    }')

if echo "$response" | grep -q "choices"; then
    echo "✓ Chat completions working"
    echo "$response" | jq '.choices[0].message.content' 2>/dev/null || echo "$response"
else
    echo "✗ Chat completions failed"
    echo "$response"
    exit 1
fi

echo ""
echo "✓ Judge server is ready for evaluation!"
