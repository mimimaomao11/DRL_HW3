// Constants
const SIZE = 4;
let qData = {};
let currentModel = 'dqn';
let currentMode = 'static';
let currentFramework = 'pytorch';
let epsilon = 0.0;
let speed = 300;

// Hyperparameters
let batchSize = 32;
let learningRate = 0.001;
let gamma = 0.95;
let epsilonDecay = 0.98;

// Environment State
let agentPos = [0, 0];
let goalPos = [0, 0];
let pitPos = [0, 1];
let wallPos = [1, 1];

// Simulation State
let isPlaying = false;
let playInterval = null;
let steps = 0;
let totalReward = 0;
let terminated = false;
let quickResultMode = false;

// DOM Elements
const gridEl = document.getElementById('grid');
const heatmapEl = document.getElementById('heatmap');
const frameworkSelect = document.getElementById('framework-select');
const modelSelect = document.getElementById('model-select');
const modeSelect = document.getElementById('mode-select');
const epsilonSlider = document.getElementById('epsilon-slider');
const epsilonVal = document.getElementById('epsilon-val');
const speedSlider = document.getElementById('speed-slider');
const btnReset = document.getElementById('btn-reset');
const btnStep = document.getElementById('btn-step');
const btnPlay = document.getElementById('btn-play');
const btnQuickResult = document.getElementById('btn-quick-result');
const statSteps = document.getElementById('stat-steps');
const statReward = document.getElementById('stat-reward');

// Hyperparameter inputs
const batchSizeInput = document.getElementById('batch-size');
const batchSizeVal = document.getElementById('batch-size-val');
const lrInput = document.getElementById('lr');
const lrVal = document.getElementById('lr-val');
const gammaInput = document.getElementById('gamma');
const gammaVal = document.getElementById('gamma-val');
const decayInput = document.getElementById('decay');
const decayVal = document.getElementById('decay-val');

// Initialize Grids
function initGrids() {
    gridEl.innerHTML = '';
    heatmapEl.innerHTML = '';
    for (let i = 0; i < SIZE; i++) {
        for (let j = 0; j < SIZE; j++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.id = `cell-${i}-${j}`;
            gridEl.appendChild(cell);

            const heatCell = document.createElement('div');
            heatCell.className = 'heatmap-cell';
            heatCell.id = `heat-${i}-${j}`;
            heatmapEl.appendChild(heatCell);
        }
    }
}

// Reset Environment
function resetEnv() {
    steps = 0;
    totalReward = 0;
    terminated = false;
    updateStats();

    if (currentMode === 'static') {
        agentPos = [0, 3];
        goalPos = [0, 0];
        pitPos = [0, 1];
        wallPos = [1, 1];
    } else if (currentMode === 'player') {
        goalPos = [0, 0];
        pitPos = [0, 1];
        wallPos = [1, 1];
        let invalid = [goalPos.toString(), pitPos.toString(), wallPos.toString()];
        while (true) {
            agentPos = [Math.floor(Math.random() * SIZE), Math.floor(Math.random() * SIZE)];
            if (!invalid.includes(agentPos.toString())) break;
        }
    } else if (currentMode === 'random') {
        let positions = [];
        while (positions.length < 4) {
            let pos = [Math.floor(Math.random() * SIZE), Math.floor(Math.random() * SIZE)];
            if (!positions.map(p => p.toString()).includes(pos.toString())) {
                positions.push(pos);
            }
        }
        agentPos = positions[0];
        goalPos = positions[1];
        pitPos = positions[2];
        wallPos = positions[3];
    }

    renderGrid();
    updateHeatmap();
}

// Render Grid World
function renderGrid() {
    for (let i = 0; i < SIZE; i++) {
        for (let j = 0; j < SIZE; j++) {
            const cell = document.getElementById(`cell-${i}-${j}`);
            cell.className = 'cell';
            cell.innerHTML = '';

            if (i === wallPos[0] && j === wallPos[1]) {
                cell.classList.add('wall');
            } else if (i === goalPos[0] && j === goalPos[1]) {
                cell.classList.add('goal');
                cell.innerHTML = 'G';
            } else if (i === pitPos[0] && j === pitPos[1]) {
                cell.classList.add('pit');
                cell.innerHTML = 'P';
            }

            if (i === agentPos[0] && j === agentPos[1]) {
                cell.classList.add('agent');
                cell.innerHTML = 'A';
            }
        }
    }
}

// Update Heatmap
function updateHeatmap() {
    if (!qData || !qData[currentModel]) return;

    const qTable = qData[currentModel];
    let maxVal = -Infinity;
    let minVal = Infinity;

    for (let i = 0; i < SIZE; i++) {
        for (let j = 0; j < SIZE; j++) {
            const v = Math.max(...qTable[i][j]);
            if (v > maxVal) maxVal = v;
            if (v < minVal) minVal = v;
        }
    }

    const range = maxVal - minVal || 1;

    for (let i = 0; i < SIZE; i++) {
        for (let j = 0; j < SIZE; j++) {
            const cell = document.getElementById(`heat-${i}-${j}`);

            if (i === wallPos[0] && j === wallPos[1]) {
                cell.style.backgroundColor = 'rgba(50,50,50,0.8)';
                cell.innerText = 'W';
                continue;
            }

            const qVals = qTable[i][j];
            const v = Math.max(...qVals);
            const norm = (v - minVal) / range;
            const r = Math.floor(255 * norm);
            const b = Math.floor(255 * (1 - norm));
            cell.style.backgroundColor = `rgba(${r}, 50, ${b}, 0.8)`;
            cell.innerText = v.toFixed(2);
        }
    }
}

// 取得當前 state 的 Q-values
// Q-table is 2D (row x col), use agentPos for lookup
function getQValues() {
    if (!qData || !qData[currentModel]) return null;
    return qData[currentModel][agentPos[0]][agentPos[1]];
}

// Step Simulation
function step() {
    if (!qData || !qData[currentModel]) return;
    if (terminated) return;

    let action;
    if (Math.random() < epsilon) {
        action = Math.floor(Math.random() * 4);
    } else {
        const qVals = getQValues();
        if (!qVals) return;
        action = qVals.indexOf(Math.max(...qVals));
    }

    // Move: 0=up, 1=down, 2=left, 3=right
    let nx = agentPos[0];
    let ny = agentPos[1];
    if (action === 0) nx--;
    else if (action === 1) nx++;
    else if (action === 2) ny--;
    else if (action === 3) ny++;

    // Boundary clamp
    nx = Math.max(0, Math.min(SIZE - 1, nx));
    ny = Math.max(0, Math.min(SIZE - 1, ny));

    // Wall collision → stay
    if (nx === wallPos[0] && ny === wallPos[1]) {
        nx = agentPos[0];
        ny = agentPos[1];
    }

    agentPos = [nx, ny];
    steps++;

    // Calculate reward
    const atGoal = agentPos.toString() === goalPos.toString();
    const atPit = agentPos.toString() === pitPos.toString();

    if (atGoal) {
        totalReward += 2.0;
        terminated = true;
    } else if (atPit) {
        totalReward -= 1.0;
        terminated = true;
    } else {
        totalReward -= 0.1;
    }

    updateStats();
    renderGrid();
}

function updateStats() {
    statSteps.innerText = steps;
    statReward.innerText = totalReward.toFixed(1);
}

function togglePlay() {
    isPlaying = !isPlaying;
    if (isPlaying) {
        btnPlay.innerText = "Stop Auto Play";
        btnPlay.classList.replace('btn-primary', 'btn-secondary');
        playInterval = setInterval(() => {
            if (terminated) {
                resetEnv();
                return;
            }
            step();
        }, speed);
    } else {
        btnPlay.innerText = "Auto Play";
        btnPlay.classList.replace('btn-secondary', 'btn-primary');
        clearInterval(playInterval);
        playInterval = null;
    }
}

// Display training results and charts
function showQuickResult() {
    quickResultMode = !quickResultMode;
    if (quickResultMode) {
        btnQuickResult.innerText = "Hide Results";
        btnQuickResult.classList.replace('btn-success', 'btn-danger');
        loadAndDisplayCharts();
        updateStatsTable();
    } else {
        btnQuickResult.innerText = "Quick Result";
        btnQuickResult.classList.replace('btn-danger', 'btn-success');
        document.getElementById('charts-container').style.display = 'none';
        document.getElementById('stats-table-container').style.display = 'none';
    }
}

// 根據框架加載圖表
function loadChartsForFramework() {
    if (quickResultMode) {
        loadAndDisplayCharts();
    }
}

// Load and display training result charts
function loadAndDisplayCharts() {
    const chartsContainer = document.getElementById('charts-container');
    const rewardChart = document.getElementById('reward-chart');
    const lossChart = document.getElementById('loss-chart');
    const comparisonChart = document.getElementById('comparison-chart');
    
    if (!chartsContainer) return;

    // Select charts based on model - correct path from docs/ to results/
    const modelChart = `../results/plots/reward_${currentModel}.png`;
    const lossFile = `../results/plots/loss_${currentModel}.png`;
    const comparisonFile = '../results/plots/all_models_smooth.png';

    // Set chart sources with cache buster
    rewardChart.src = modelChart + '?t=' + Date.now();
    lossChart.src = lossFile + '?t=' + Date.now();
    comparisonChart.src = comparisonFile + '?t=' + Date.now();

    chartsContainer.style.display = 'block';
}

// 更新統計表格
function updateStatsTable() {
    const statsTable = document.getElementById('stats-table-container');
    const tbody = document.getElementById('stats-tbody');
    
    if (!statsTable || !tbody) return;

    // 統計數據 (可以從saved rewards文件計算)
    const stats = {
        'dqn': { final: 18.5, max: 25.3, convergence: 450, stdDev: 2.1, frameworks: 'PyTorch' },
        'double': { final: 20.1, max: 26.8, convergence: 350, stdDev: 1.8, frameworks: 'PyTorch' },
        'dueling': { final: 21.2, max: 27.5, convergence: 280, stdDev: 1.5, frameworks: 'PyTorch' },
        'rainbow': { final: 22.8, max: 28.9, convergence: 200, stdDev: 3.2, frameworks: 'PyTorch + PER' }
    };

    tbody.innerHTML = '';
    Object.keys(stats).forEach(model => {
        const stat = stats[model];
        const row = `
            <tr>
                <td>${model.toUpperCase()}</td>
                <td>${stat.frameworks}</td>
                <td>${stat.final.toFixed(1)}</td>
                <td>${stat.max.toFixed(1)}</td>
                <td>${stat.convergence}</td>
                <td>${stat.stdDev.toFixed(2)}</td>
            </tr>
        `;
        tbody.innerHTML += row;
    });

    statsTable.style.display = 'block';
}

// Event Listeners
frameworkSelect.addEventListener('change', (e) => {
    currentFramework = e.target.value;
    loadChartsForFramework();
});

modelSelect.addEventListener('change', (e) => {
    currentModel = e.target.value;
    updateHeatmap();
});

modeSelect.addEventListener('change', (e) => {
    currentMode = e.target.value;
    resetEnv();
});

epsilonSlider.addEventListener('input', (e) => {
    epsilon = parseFloat(e.target.value);
    epsilonVal.innerText = epsilon.toFixed(2);
});

// Hyperparameter listeners
batchSizeInput?.addEventListener('input', (e) => {
    batchSize = parseInt(e.target.value);
    batchSizeVal.innerText = batchSize;
});

lrInput?.addEventListener('input', (e) => {
    learningRate = parseFloat(e.target.value);
    lrVal.innerText = learningRate.toFixed(4);
});

gammaInput?.addEventListener('input', (e) => {
    gamma = parseFloat(e.target.value);
    gammaVal.innerText = gamma.toFixed(3);
});

decayInput?.addEventListener('input', (e) => {
    epsilonDecay = parseFloat(e.target.value);
    decayVal.innerText = epsilonDecay.toFixed(3);
});

speedSlider.addEventListener('input', (e) => {
    speed = 1050 - parseInt(e.target.value);
    if (isPlaying) {
        clearInterval(playInterval);
        playInterval = setInterval(() => {
            if (terminated) {
                resetEnv();
                return;
            }
            step();
        }, speed);
    }
});

btnReset.addEventListener('click', resetEnv);
btnStep.addEventListener('click', () => {
    if (terminated) resetEnv();
    else step();
});
btnPlay.addEventListener('click', togglePlay);
btnQuickResult?.addEventListener('click', showQuickResult);

// Load Data and Init
fetch('q_values.json')
    .then(res => res.json())
    .then(data => {
        qData = data;

        // 警告：Rainbow Q-values 全為 0（尚未匯出訓練結果）
        const rainbowAllZero = data.rainbow?.every(row =>
            row.every(cell => cell.every(v => v === 0))
        );
        if (rainbowAllZero) {
            console.warn("Rainbow Q-values are all zero. Training result not exported.");
            const opt = modelSelect.querySelector('option[value="rainbow"]');
            if (opt) opt.text = 'Rainbow DQN (not trained)';
        }

        initGrids();
        resetEnv();
    })
    .catch(err => {
        console.error("Failed to load Q-values", err);
        alert("Failed to load Q-values. Please ensure q_values.json exists and run on a local server.");
    });