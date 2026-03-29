// Qwen3-VL Thinking Mode Visualization - Frontend Logic

// State management
const state = {
    imageData: null,
    tokens: [],
    hiddenStates: null,
    attentionWeights: null,
    continuousMask: [],
    tsneCoordinates: null,
    selectedTokenIndex: null,
};

// DOM elements
const elements = {
    imageUpload: document.getElementById('imageUpload'),
    imagePreview: document.getElementById('previewImg'),
    textInput: document.getElementById('textInput'),
    sendBtn: document.getElementById('sendBtn'),
    loadExampleBtn: document.getElementById('loadExampleBtn'),
    autoLoadExample: document.getElementById('autoLoadExample'),
    status: document.getElementById('status'),
    modelInfo: document.getElementById('modelInfo'),
    tokenSequence: document.getElementById('tokenSequence'),
    tsnePlot: document.getElementById('tsnePlot'),
    attentionPlot: document.getElementById('attentionPlot'),
    answer: document.getElementById('answer'),
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    setupEventListeners();
    await checkHealth();

    // Auto-load example if checkbox is checked
    if (elements.autoLoadExample.checked) {
        await loadExample();
    }
});

function setupEventListeners() {
    elements.imageUpload.addEventListener('change', handleImageUpload);
    elements.textInput.addEventListener('input', handleTextInput);
    elements.sendBtn.addEventListener('click', handleGenerate);
    elements.loadExampleBtn.addEventListener('click', loadExample);
}

async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();

        if (data.model_loaded) {
            elements.modelInfo.textContent = `Model: ${data.checkpoint_path || 'Base model'}`;
            setStatus('Model loaded successfully', 'success');
        } else {
            setStatus('Model not loaded - check configuration', 'error');
        }
    } catch (error) {
        setStatus('Failed to connect to server', 'error');
        console.error('Health check failed:', error);
    }
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        state.imageData = e.target.result.split(',')[1]; // Get base64 without prefix
        elements.imagePreview.src = e.target.result;
        elements.imagePreview.style.display = 'block';
        updateSendButton();
    };
    reader.readAsDataURL(file);
}

function handleTextInput() {
    updateSendButton();
}

function updateSendButton() {
    const hasText = elements.textInput.value.trim().length > 0;
    elements.sendBtn.disabled = !hasText;
}

async function loadExample() {
    try {
        setStatus('Loading example...', 'loading');
        const response = await fetch('/api/example');
        const data = await response.json();

        if (data.image_base64) {
            state.imageData = data.image_base64;
            elements.imagePreview.src = `data:image/jpeg;base64,${data.image_base64}`;
            elements.imagePreview.style.display = 'block';
        }

        elements.textInput.value = data.question || '';
        updateSendButton();
        setStatus('Example loaded', 'success');
    } catch (error) {
        setStatus('Failed to load example', 'error');
        console.error('Load example failed:', error);
    }
}

async function handleGenerate() {
    if (elements.sendBtn.disabled) return;

    const request = {
        text: elements.textInput.value.trim(),
        image_base64: state.imageData || null,
        max_tokens: 2048,
        temperature: 0.7,
    };

    setStatus('Generating...', 'loading');
    elements.sendBtn.disabled = true;

    try {
        const response = await fetch('/api/infer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        // Update state
        state.tokens = data.tokens || [];
        state.hiddenStates = data.hidden_states || null;
        state.attentionWeights = data.attention_weights || null;
        state.continuousMask = data.continuous_mask || [];
        state.tsneCoordinates = data.tsne_coordinates || null;

        // Render results
        renderAnswer(data.answer);
        renderTokens();
        renderTSNE();

        if (state.attentionWeights && state.attentionWeights.length > 0) {
            renderAttention();
        } else {
            elements.attentionPlot.innerHTML = '<p class="placeholder">Attention weights not available for this checkpoint</p>';
        }

        setStatus('Generation complete', 'success');
    } catch (error) {
        setStatus(`Error: ${error.message}`, 'error');
        console.error('Generation failed:', error);
    } finally {
        elements.sendBtn.disabled = false;
    }
}

function setStatus(message, type = 'info') {
    elements.status.textContent = message;
    elements.status.className = `status ${type}`;
}

function renderAnswer(answer) {
    elements.answer.textContent = answer || 'No answer generated';
}

function renderTokens() {
    if (!state.tokens || state.tokens.length === 0) {
        elements.tokenSequence.innerHTML = '<p class="placeholder">No tokens generated</p>';
        return;
    }

    elements.tokenSequence.innerHTML = '';

    state.tokens.forEach((token, index) => {
        const tokenEl = document.createElement('span');
        tokenEl.className = 'token';
        tokenEl.textContent = token.text || `[${token.id}]`;
        tokenEl.dataset.index = index;

        if (state.continuousMask[index]) {
            tokenEl.classList.add('continuous');
        }

        tokenEl.addEventListener('click', () => handleTokenClick(index));
        elements.tokenSequence.appendChild(tokenEl);
    });
}

function handleTokenClick(index) {
    // Update selected state
    state.selectedTokenIndex = index;

    // Update visual selection
    document.querySelectorAll('.token').forEach((el, i) => {
        if (i === index) {
            el.classList.add('selected');
        } else {
            el.classList.remove('selected');
        }
    });

    // Update attention heatmap
    renderAttention();

    // Highlight point in t-SNE
    highlightTSNEPoint(index);
}

function renderTSNE() {
    if (!state.tsneCoordinates || state.tsneCoordinates.length === 0) {
        elements.tsnePlot.innerHTML = '<p class="placeholder">No t-SNE data available</p>';
        return;
    }

    // Prepare data
    const tokenTypes = state.tokens.map((token, i) => {
        if (state.continuousMask[i]) return 'continuous';
        if (i < 10) return 'question'; // Early tokens are likely question
        return 'answer';
    });

    const traces = {};
    const colors = {
        question: '#3b82f6',
        continuous: '#22c55e',
        answer: '#f97316',
    };

    // Group by token type
    state.tsneCoordinates.forEach((coord, i) => {
        const type = tokenTypes[i];
        if (!traces[type]) {
            traces[type] = { x: [], y: [], text: [], indices: [] };
        }
        traces[type].x.push(coord[0]);
        traces[type].y.push(coord[1]);
        traces[type].text.push(state.tokens[i]?.text || `token_${i}`);
        traces[type].indices.push(i);
    });

    // Create plot
    const plotData = Object.entries(traces).map(([type, data]) => ({
        x: data.x,
        y: data.y,
        mode: 'markers',
        type: 'scatter',
        name: type.charAt(0).toUpperCase() + type.slice(1),
        text: data.text,
        marker: {
            size: 8,
            color: colors[type],
            line: { width: 1, color: 'white' },
        },
        hoverinfo: 'text+x+y',
        customdata: data.indices,
    }));

    const layout = {
        title: 'Hidden States t-SNE',
        xaxis: { title: 't-SNE 1' },
        yaxis: { title: 't-SNE 2' },
        hovermode: 'closest',
        margin: { l: 50, r: 50, t: 50, b: 50 },
        paper_backgroundcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
    };

    Plotly.newPlot(elements.tsnePlot, plotData, layout, { responsive: true });

    // Add click handler
    elements.tsnePlot.on('plotly_click', (event) => {
        if (event.points && event.points.length > 0) {
            const index = event.points[0].customdata;
            handleTokenClick(index);
        }
    });
}

function highlightTSNEPoint(index) {
    if (!state.tsneCoordinates) return;

    const colors = state.tokens.map((_, i) => (i === index ? '#ef4444' : '#cccccc'));
    const sizes = state.tokens.map((_, i) => (i === index ? 15 : 8));

    Plotly.restyle(elements.tsnePlot, {
        'marker.color': [colors],
        'marker.size': [sizes],
    });
}

function renderAttention() {
    if (!state.attentionWeights || state.attentionWeights.length === 0) {
        elements.attentionPlot.innerHTML = '<p class="placeholder">No attention data available</p>';
        return;
    }

    // Use selected token or show all
    const selectedIndex = state.selectedTokenIndex;
    const attentionMatrix = state.attentionWeights;

    let z, y;
    if (selectedIndex !== null && selectedIndex >= 0 && selectedIndex < attentionMatrix.length) {
        z = [attentionMatrix[selectedIndex]];
        y = [state.tokens[selectedIndex]?.text || `token_${selectedIndex}`];
    } else {
        z = attentionMatrix;
        y = state.tokens.map(t => t?.text || '');
    }

    // Truncate labels
    const x = state.tokens.map(t => {
        const text = t?.text || '';
        return text.length > 20 ? text.substring(0, 20) + '...' : text;
    });

    const trace = {
        z: z,
        x: x,
        y: y,
        type: 'heatmap',
        colorscale: 'Blues',
        showscale: true,
        hoverinfo: 'z',
    };

    const layout = {
        title: selectedIndex !== null ? `Attention from token: ${y[0]}` : 'Attention Heatmap',
        xaxis: { title: 'Key Position' },
        yaxis: { title: 'Query Position' },
        margin: { l: 100, r: 50, t: 50, b: 100 },
        paper_backgroundcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
    };

    Plotly.newPlot(elements.attentionPlot, [trace], layout, { responsive: true });
}
