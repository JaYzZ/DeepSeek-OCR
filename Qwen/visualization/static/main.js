// Qwen3-VL Thinking Mode Visualization - Modern Frontend

// ============================================================================
// State Management
// ============================================================================

const state = {
    imageData: null,
    tokens: [],
    hiddenStates: null,
    attentionWeights: null,
    continuousMask: [],
    tsneCoordinates: null,
    tsneFeatureTypes: [],
    tsnePositionIndices: [],
    selectedTokenIndex: null,
    isGenerating: false,
    serverConnected: false,
    settings: {
        maxTokens: 8192,
        temperature: 0.7,
    },
};

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
    // Status
    serverStatus: document.getElementById('serverStatus'),
    statusText: document.querySelector('.status-text'),

    // Inputs
    fileUpload: document.getElementById('fileUpload'),
    fileUploadContent: document.getElementById('fileUploadContent'),
    imageInput: document.getElementById('imageInput'),
    imagePreview: document.getElementById('imagePreview'),
    previewImage: document.getElementById('previewImage'),
    removeImage: document.getElementById('removeImage'),
    textInput: document.getElementById('textInput'),
    charCount: document.getElementById('charCount'),
    generateBtn: document.getElementById('generateBtn'),
    loadExampleBtn: document.getElementById('loadExampleBtn'),

    // Settings
    maxTokens: document.getElementById('maxTokens'),
    maxTokensValue: document.getElementById('maxTokensValue'),
    temperature: document.getElementById('temperature'),
    temperatureValue: document.getElementById('temperatureValue'),

    // Model Info
    checkpointInfo: document.getElementById('checkpointInfo'),
    thinkingModeInfo: document.getElementById('thinkingModeInfo'),

    // Visualization
    tokenContainer: document.getElementById('tokenContainer'),
    tokenStats: document.getElementById('tokenStats'),
    tsnePlot: document.getElementById('tsnePlot'),
    attentionPlot: document.getElementById('attentionPlot'),
    answerContainer: document.getElementById('answerContainer'),

    // Toast
    toast: document.getElementById('toast'),
    toastMessage: document.getElementById('toastMessage'),
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    setupEventListeners();
    updateCharCount();
    updateGenerateButton();
    await checkServerHealth();
    setupDragAndDrop();
});

function setupEventListeners() {
    // File upload
    elements.fileUploadContent.addEventListener('click', () => elements.imageInput.click());
    elements.imageInput.addEventListener('change', handleImageUpload);
    elements.removeImage.addEventListener('click', removeImage);

    // Text input
    elements.textInput.addEventListener('input', handleTextInput);

    // Buttons
    elements.generateBtn.addEventListener('click', handleGenerate);
    elements.loadExampleBtn.addEventListener('click', loadExample);

    // Settings
    elements.maxTokens.addEventListener('input', (e) => {
        state.settings.maxTokens = parseInt(e.target.value);
        elements.maxTokensValue.textContent = e.target.value;
    });

    elements.temperature.addEventListener('input', (e) => {
        state.settings.temperature = parseFloat(e.target.value);
        elements.temperatureValue.textContent = e.target.value;
    });
}

function setupDragAndDrop() {
    elements.fileUpload.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.fileUploadContent.style.borderColor = 'var(--primary)';
        elements.fileUploadContent.style.background = 'var(--primary-light)';
    });

    elements.fileUpload.addEventListener('dragleave', () => {
        elements.fileUploadContent.style.borderColor = '';
        elements.fileUploadContent.style.background = '';
    });

    elements.fileUpload.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.fileUploadContent.style.borderColor = '';
        elements.fileUploadContent.style.background = '';

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            processImageFile(file);
        }
    });
}

// ============================================================================
// Server Communication
// ============================================================================

async function checkServerHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        if (data.status === 'healthy' && data.model_loaded) {
            state.serverConnected = true;
            updateServerStatus('connected', 'Connected');

            // Update model info
            const checkpoint = data.checkpoint_path || 'Unknown';
            elements.checkpointInfo.textContent = checkpoint.split('/').slice(-2).join('/');
            elements.thinkingModeInfo.textContent = data.thinking_enabled ? 'Enabled' : 'Disabled';

            showToast('Model loaded successfully', 'success');
        } else {
            state.serverConnected = false;
            updateServerStatus('error', 'Model not loaded');
            showToast('Model not loaded - check server configuration', 'error');
        }
    } catch (error) {
        state.serverConnected = false;
        updateServerStatus('error', 'Connection failed');
        showToast('Failed to connect to server', 'error');
        console.error('Health check failed:', error);
    }
}

async function loadExample() {
    try {
        showToast('Loading example...', 'loading');

        const response = await fetch('/api/example');
        if (!response.ok) throw new Error('Failed to load example');

        const data = await response.json();

        if (data.image_base64) {
            state.imageData = data.image_base64;
            elements.previewImage.src = `data:image/jpeg;base64,${data.image_base64}`;
            elements.imagePreview.style.display = 'block';
            elements.fileUploadContent.style.display = 'none';
        }

        if (data.question) {
            elements.textInput.value = data.question;
            updateCharCount();
        }

        updateGenerateButton();
        hideToast();
        showToast('Example loaded', 'success');
    } catch (error) {
        hideToast();
        showToast('Failed to load example', 'error');
        console.error('Load example failed:', error);
    }
}

async function handleGenerate() {
    if (state.isGenerating || !state.serverConnected) return;

    const request = {
        text: elements.textInput.value.trim(),
        image_base64: state.imageData || null,
        max_tokens: state.settings.maxTokens,
        temperature: state.settings.temperature,
    };

    state.isGenerating = true;
    elements.generateBtn.disabled = true;
    showToast('Generating response...', 'loading');

    try {
        const response = await fetch('/api/infer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);

        const data = await response.json();

        // Update state
        state.tokens = data.tokens || [];
        state.hiddenStates = data.hidden_states || null;
        state.attentionWeights = data.attention_weights || null;
        state.continuousMask = data.continuous_mask || [];
        state.tsneCoordinates = data.tsne_coordinates || null;
        state.tsneFeatureTypes = data.tsne_feature_types || [];
        state.tsnePositionIndices = data.tsne_position_indices || [];

        // Render visualizations
        renderAnswer(data.answer);
        renderTokens();
        renderTSNE();
        renderAttention();

        hideToast();
        showToast('Generation complete!', 'success');
    } catch (error) {
        hideToast();
        showToast(`Error: ${error.message}`, 'error');
        console.error('Generation failed:', error);
    } finally {
        state.isGenerating = false;
        elements.generateBtn.disabled = false;
        updateGenerateButton();
    }
}

// ============================================================================
// Input Handlers
// ============================================================================

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    processImageFile(file);
}

function processImageFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        state.imageData = e.target.result.split(',')[1];
        elements.previewImage.src = e.target.result;
        elements.imagePreview.style.display = 'block';
        elements.fileUploadContent.style.display = 'none';
        updateGenerateButton();
    };
    reader.readAsDataURL(file);
}

function removeImage(event) {
    event.stopPropagation();
    state.imageData = null;
    elements.imageInput.value = '';
    elements.previewImage.src = '';
    elements.imagePreview.style.display = 'none';
    elements.fileUploadContent.style.display = 'block';
    updateGenerateButton();
}

function handleTextInput() {
    updateCharCount();
    updateGenerateButton();
}

function updateCharCount() {
    const count = elements.textInput.value.length;
    elements.charCount.textContent = count;
}

function updateGenerateButton() {
    const hasText = elements.textInput.value.trim().length > 0;
    const hasImage = state.imageData !== null;
    const canGenerate = (hasText || hasImage) && state.serverConnected && !state.isGenerating;

    elements.generateBtn.disabled = !canGenerate;
}

// ============================================================================
// Rendering Functions
// ============================================================================

function renderAnswer(answer) {
    if (!answer) {
        elements.answerContainer.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">💭</span>
                <p>No answer generated</p>
            </div>
        `;
        return;
    }

    elements.answerContainer.innerHTML = `<div class="answer-content">${escapeHtml(answer)}</div>`;
}

function renderTokens() {
    if (!state.tokens || state.tokens.length === 0) {
        elements.tokenContainer.classList.add('has-empty-state');
        elements.tokenContainer.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">🔤</span>
                <p>No tokens generated</p>
            </div>
        `;
        elements.tokenStats.textContent = '';
        return;
    }

    elements.tokenContainer.classList.remove('has-empty-state');
    elements.tokenContainer.innerHTML = '';
    const continuousCount = state.continuousMask.filter(Boolean).length;

    state.tokens.forEach((token, index) => {
        const tokenEl = document.createElement('span');
        tokenEl.className = 'token';
        tokenEl.textContent = token.text || `[${token.id}]`;
        tokenEl.dataset.index = index;

        if (state.continuousMask[index]) {
            tokenEl.classList.add('continuous');
        }

        if (state.selectedTokenIndex === index) {
            tokenEl.classList.add('selected');
        }

        tokenEl.addEventListener('click', () => handleTokenClick(index));
        elements.tokenContainer.appendChild(tokenEl);
    });

    elements.tokenStats.textContent = `(${state.tokens.length} total, ${continuousCount} continuous)`;
}

function handleTokenClick(index) {
    console.log('[handleTokenClick] Called with index:', index, 'current selected:', state.selectedTokenIndex);

    // Toggle selection: if clicking same token, deselect
    if (state.selectedTokenIndex === index) {
        state.selectedTokenIndex = null;
        console.log('[handleTokenClick] Deselecting token');
    } else {
        state.selectedTokenIndex = index;
        console.log('[handleTokenClick] Selecting token:', index);
    }

    // Update visual selection
    document.querySelectorAll('.token').forEach((el, i) => {
        el.classList.toggle('selected', i === state.selectedTokenIndex);
    });

    // Update visualizations
    renderAttention();
    highlightTSNEPoint(state.selectedTokenIndex);

    console.log('[handleTokenClick] Finished, new selectedTokenIndex:', state.selectedTokenIndex);
}

function renderTSNE() {
    if (!state.tsneCoordinates || state.tsneCoordinates.length === 0) {
        elements.tsnePlot.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">📈</span>
                <p>No t-SNE data available</p>
            </div>
        `;
        return;
    }

    // Debug: Check what data structure we have
    console.log('[TSNE] tsneFeatureTypes:', state.tsneFeatureTypes);
    console.log('[TSNE] tsnePositionIndices:', state.tsnePositionIndices);
    console.log('[TSNE] continuousMask:', state.continuousMask);

    // New data structure: tsneCoordinates, tsneFeatureTypes, tsnePositionIndices
    // For each position, we have up to 3 dots: token_emb, hidden_state, vae_sample
    const hasNewStructure = state.tsneFeatureTypes && state.tsnePositionIndices &&
                          Array.isArray(state.tsneFeatureTypes) &&
                          Array.isArray(state.tsnePositionIndices) &&
                          state.tsneFeatureTypes.length > 0;

    if (hasNewStructure) {
        // Group by feature type only (not by position type)
        const traces = {};

        // Define colors and markers for feature types
        const featureConfig = {
            token_emb: { color: '#3b82f6', symbol: 'circle', name: 'Token Embedding' },
            hidden_state: { color: '#f97316', symbol: 'diamond', name: 'Hidden State' },
            vae_sample: { color: '#22c55e', symbol: 'star', name: 'VAE Sample' },
        };

        state.tsneCoordinates.forEach((coord, i) => {
            const featureType = state.tsneFeatureTypes[i];
            const posIdx = state.tsnePositionIndices[i];
            const token = state.tokens[posIdx];

            if (!featureConfig[featureType]) return; // Skip unknown feature types

            const config = featureConfig[featureType];
            if (!traces[featureType]) {
                traces[featureType] = {
                    x: [],
                    y: [],
                    text: [],
                    indices: [],
                    color: config.color,
                    symbol: config.symbol,
                    name: config.name,
                };
            }

            traces[featureType].x.push(coord[0]);
            traces[featureType].y.push(coord[1]);
            traces[featureType].text.push(token?.text || `token_${posIdx}`);
            traces[featureType].indices.push(posIdx);
        });

        // Create plot with separate traces for each feature type
        const plotData = Object.values(traces).map(data => ({
            x: data.x,
            y: data.y,
            mode: 'markers',
            type: 'scatter',
            name: data.name,
            text: data.text,
            marker: {
                size: 10,
                color: data.color,
                symbol: data.symbol,
                line: { width: 1.5, color: 'white' },
                opacity: 1.0,
            },
            hoverinfo: 'text+x+y',
            customdata: data.indices,
        }));

        const allX = state.tsneCoordinates.map(coord => coord[0]);
        const allY = state.tsneCoordinates.map(coord => coord[1]);
        const minX = Math.min(...allX);
        const maxX = Math.max(...allX);
        const minY = Math.min(...allY);
        const maxY = Math.max(...allY);
        const padX = Math.max((maxX - minX) * 0.08, 1e-3);
        const padY = Math.max((maxY - minY) * 0.08, 1e-3);

        const layout = {
            xaxis: {
                title: 't-SNE 1',
                range: [minX - padX, maxX + padX],
                autorange: false,
                zeroline: false,
                automargin: true,
                fixedrange: true,
            },
            yaxis: {
                title: 't-SNE 2',
                range: [minY - padY, maxY + padY],
                autorange: false,
                zeroline: false,
                automargin: true,
                fixedrange: true,
            },
            hovermode: 'closest',
            margin: { l: 56, r: 20, t: 20, b: 48 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            autosize: true,
            showlegend: true,
            legend: {
                orientation: 'h',
                x: 0,
                y: 1.08,
                xanchor: 'left',
                yanchor: 'bottom',
            },
        };

        const config = { responsive: true, displayModeBar: false };
        elements.tsnePlot.innerHTML = '';
        Plotly.newPlot(elements.tsnePlot, plotData, layout, config).then(() => {
            Plotly.Plots.resize(elements.tsnePlot);
        });
    } else {
        // Legacy behavior for old data structure (single dot per position)
        const tokenTypes = state.tokens.map((token, i) => {
            if (state.continuousMask[i]) return 'continuous';
            if (i < 10) return 'question';
            return 'answer';
        });

        const traces = {};
        const colors = {
            question: '#3b82f6',
            continuous: '#22c55e',
            answer: '#f97316',
        };

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

        const plotData = Object.entries(traces).map(([type, data]) => ({
            x: data.x,
            y: data.y,
            mode: 'markers',
            type: 'scatter',
            name: type.charAt(0).toUpperCase() + type.slice(1),
            text: data.text,
            marker: {
                size: 10,
                color: colors[type],
                line: { width: 2, color: 'white' },
                opacity: 1.0,
            },
            hoverinfo: 'text+x+y',
            customdata: data.indices,
        }));

        const allX = state.tsneCoordinates.map(coord => coord[0]);
        const allY = state.tsneCoordinates.map(coord => coord[1]);
        const minX = Math.min(...allX);
        const maxX = Math.max(...allX);
        const minY = Math.min(...allY);
        const maxY = Math.max(...allY);
        const padX = Math.max((maxX - minX) * 0.08, 1e-3);
        const padY = Math.max((maxY - minY) * 0.08, 1e-3);

        const layout = {
            xaxis: {
                title: 't-SNE 1',
                range: [minX - padX, maxX + padX],
                autorange: false,
                zeroline: false,
                automargin: true,
                fixedrange: true,
            },
            yaxis: {
                title: 't-SNE 2',
                range: [minY - padY, maxY + padY],
                autorange: false,
                zeroline: false,
                automargin: true,
                fixedrange: true,
            },
            hovermode: 'closest',
            margin: { l: 56, r: 20, t: 20, b: 48 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            autosize: true,
            showlegend: true,
            legend: {
                orientation: 'h',
                x: 0,
                y: 1.08,
                xanchor: 'left',
                yanchor: 'bottom',
            },
        };

        const config = { responsive: true, displayModeBar: false };
        elements.tsnePlot.innerHTML = '';
        Plotly.newPlot(elements.tsnePlot, plotData, layout, config).then(() => {
            Plotly.Plots.resize(elements.tsnePlot);
        });
    }

    // Add click handler
    elements.tsnePlot.on('plotly_click', (event) => {
        if (event.points && event.points.length > 0) {
            const index = event.points[0].customdata;
            if (index !== undefined && index !== null) {
                handleTokenClick(index);
            } else {
                // Clicked on empty space, reset selection
                state.selectedTokenIndex = null;
                document.querySelectorAll('.token').forEach(el => {
                    el.classList.remove('selected');
                });
                renderAttention();
                highlightTSNEPoint(null);
            }
        } else {
            // Clicked on empty space, reset selection
            state.selectedTokenIndex = null;
            document.querySelectorAll('.token').forEach(el => {
                el.classList.remove('selected');
            });
            renderAttention();
            highlightTSNEPoint(null);
        }
    });
}

function highlightTSNEPoint(index) {
    if (!state.tsneCoordinates) return;

    console.log('[highlightTSNEPoint] Called with index:', index);

    const plotData = elements.tsnePlot.data;
    if (!plotData || plotData.length === 0) return;

    // If index is null, reset: remove highlight traces and reset opacity
    if (index === null) {
        // Remove any highlight traces (they have 'highlight' in name)
        const indicesToRemove = [];
        plotData.forEach((trace, i) => {
            if (trace.name && trace.name.includes('highlight')) {
                indicesToRemove.push(i);
            }
        });
        if (indicesToRemove.length > 0) {
            // Sort in descending order to remove from end first
            indicesToRemove.sort((a, b) => b - a);
            Plotly.deleteTraces(elements.tsnePlot, indicesToRemove);
        }

        // Reset opacity of remaining traces
        plotData.forEach((trace) => {
            if (!trace.name || !trace.name.includes('highlight')) {
                const update = {
                    'marker.opacity': Array(trace.x.length).fill(1.0),
                };
                Plotly.restyle(elements.tsnePlot, update, [plotData.indexOf(trace)]);
            }
        });
        return;
    }

    // First, remove any existing highlight traces BEFORE adding new ones
    const indicesToRemove = [];
    plotData.forEach((trace, i) => {
        if (trace.name && trace.name.includes('highlight')) {
            indicesToRemove.push(i);
        }
    });
    if (indicesToRemove.length > 0) {
        indicesToRemove.sort((a, b) => b - a);
        Plotly.deleteTraces(elements.tsnePlot, indicesToRemove);
    }

    // Then, fade markers in the plot
    plotData.forEach((trace) => {
        if (!trace.name || !trace.name.includes('highlight')) {
            const opacities = trace.customdata?.map((posIdx) => {
                // Fade unselected, keep selected at full opacity
                return (posIdx === index) ? 1.0 : 0.2;
            }) || Array(trace.x.length).fill(0.2);

            const update = {
                'marker.opacity': opacities,
            };
            Plotly.restyle(elements.tsnePlot, update, [plotData.indexOf(trace)]);
        }
    });

    // Finally, add highlight traces on top for selected markers
    const highlightTraces = [];

    plotData.forEach((trace) => {
        if (!trace.name || trace.name.includes('highlight')) return;

        const selectedX = [];
        const selectedY = [];
        const selectedText = [];

        trace.customdata?.forEach((posIdx, i) => {
            if (posIdx === index) {
                selectedX.push(trace.x[i]);
                selectedY.push(trace.y[i]);
                selectedText.push(trace.text[i]);
            }
        });

        if (selectedX.length > 0) {
            highlightTraces.push({
                x: selectedX,
                y: selectedY,
                mode: 'markers',
                type: 'scatter',
                name: `${trace.name} (highlight)`,
                text: selectedText,
                marker: {
                    size: 20,
                    color: trace.marker.color,
                    symbol: trace.marker.symbol,
                    line: { width: 1.5, color: 'white' },
                    opacity: 1.0,
                },
                hoverinfo: 'text+x+y',
                showlegend: false,
            });
        }
    });

    // Add highlight traces on top (they will appear above original traces)
    if (highlightTraces.length > 0) {
        console.log('[highlightTSNEPoint] Adding', highlightTraces.length, 'highlight traces on top');
        Plotly.addTraces(elements.tsnePlot, highlightTraces);
    }
}

function renderAttention() {
    if (!state.attentionWeights || state.attentionWeights.length === 0) {
        elements.attentionPlot.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">🎨</span>
                <p>Attention weights not available for this checkpoint</p>
            </div>
        `;
        return;
    }

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
        title: {
            text: selectedIndex !== null
                ? `Attention from: ${escapeHtml(y[0])}`
                : 'Attention Heatmap',
            font: { size: 16 }
        },
        xaxis: { title: 'Key Position' },
        yaxis: { title: 'Query Position' },
        margin: { l: 100, r: 50, t: 50, b: 100 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
    };

    const config = { responsive: true, displayModeBar: false };
    elements.attentionPlot.innerHTML = '';
    Plotly.newPlot(elements.attentionPlot, [trace], layout, config);
}

// ============================================================================
// UI Utilities
// ============================================================================

function updateServerStatus(status, message) {
    elements.serverStatus.className = `status-badge ${status}`;
    elements.statusText.textContent = message;
}

function showToast(message, type = 'info') {
    elements.toastMessage.textContent = message;
    elements.toast.style.display = 'block';

    const icon = elements.toast.querySelector('.toast-icon');
    if (type === 'loading') {
        icon.textContent = '⏳';
        icon.style.animation = 'spin 2s linear infinite';
    } else if (type === 'success') {
        icon.textContent = '✓';
        icon.style.animation = 'none';
    } else if (type === 'error') {
        icon.textContent = '✕';
        icon.style.animation = 'none';
    } else {
        icon.textContent = 'ℹ';
        icon.style.animation = 'none';
    }

    // Auto-hide for success messages
    if (type === 'success') {
        setTimeout(hideToast, 3000);
    }
}

function hideToast() {
    elements.toast.style.display = 'none';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
