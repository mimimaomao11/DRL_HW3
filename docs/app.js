// Constants
const SIZE = 4;
let qData = {};
let currentModel = 'dqn';
let currentMode = 'static';
let epsilon = 0.0;
let speed = 300;

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
let terminated = false; // 新增：追蹤是否已終止

// DOM Elements
const gridEl = document.getElementById('grid');
const heatmapEl = document.getElementById('heatmap');
const modelSelect = document.getElementById('model-select');
const modeSelect = document.getElementById('mode-select');
const epsilonSlider = document.getElementById('epsilon-slider');
const epsilonVal = document.getElementById('epsilon-val');
const speedSlider = document.getElementById('speed-slider');
const btnReset = document.getElementById('btn-reset');
const btnStep = document.getElementById('btn-step');
const btnPlay = document.getElementById('btn-play');
const statSteps = document.getElementById('stat-steps');
const statReward = document.getElementById('stat-reward');

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
                cell.innerHTML = '🏁';
            } else if (i === pitPos[0] && j === pitPos[1]) {
                cell.classList.add('pit');
                cell.innerHTML = '🔥';
            }

            if (i === agentPos[0] && j === agentPos[1]) {
                cell.classList.add('agent');
                cell.innerHTML = '🤖';
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
// BUG FIX: Q-table 是 2D (row x col)，只用 agentPos 查詢即可（符合 JSON 結構）
function getQValues() {
    if (!qData || !qData[currentModel]) return null;
    return qData[currentModel][agentPos[0]][agentPos[1]];
}

// Step Simulation
// BUG FIX: 移除 step() 內的 setTimeout(resetEnv)，改由呼叫端統一處理終止
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

// BUG FIX: Auto Play 統一在 setInterval 裡處理終止 → reset，不用 setTimeout
function togglePlay() {
    isPlaying = !isPlaying;
    if (isPlaying) {
        btnPlay.innerText = "Stop Auto Play";
        btnPlay.classList.replace('btn-primary', 'btn-secondary');
        playInterval = setInterval(() => {
            if (terminated) {
                resetEnv(); // 終止後先 reset，下一 tick 再繼續
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

// Event Listeners
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

// BUG FIX: Speed slider 移除 dir="rtl"（在 HTML），這裡翻轉數值讓右 = 快
speedSlider.addEventListener('input', (e) => {
    // slider range 50~1000，翻轉後：右邊(1000)→ delay=50ms(快)，左邊(50)→ delay=1000ms(慢)
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
            if (opt) opt.text = 'Rainbow DQN (⚠️ 未訓練)';
        }

        initGrids();
        resetEnv();
    })
    .catch(err => {
        console.error("Failed to load Q-values", err);
        alert("Failed to load Q-values. Please ensure q_values.json exists and run on a local server.");
    });