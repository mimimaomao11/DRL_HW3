// Constants
const SIZE = 4;
let qData = {}; // Stores all q-tables
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
            let strPos = pos.toString();
            if (!positions.map(p => p.toString()).includes(strPos)) {
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

            if (i === wallPos[0] && j === wallPos[1]) cell.classList.add('wall');
            else if (i === goalPos[0] && j === goalPos[1]) {
                cell.classList.add('goal');
                cell.innerHTML = '🏁';
            }
            else if (i === pitPos[0] && j === pitPos[1]) {
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

// Update Heatmap based on Q-values
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

    for (let i = 0; i < SIZE; i++) {
        for (let j = 0; j < SIZE; j++) {
            const cell = document.getElementById(`heat-${i}-${j}`);
            const qVals = qTable[i][j];
            const v = Math.max(...qVals);
            
            // Normalize for color mapping
            let norm = (v - minVal) / (maxVal - minVal || 1);
            
            // Map to blue-red scale
            let r = Math.floor(255 * norm);
            let b = Math.floor(255 * (1 - norm));
            
            cell.style.backgroundColor = `rgba(${r}, 50, ${b}, 0.8)`;
            cell.innerText = v.toFixed(2);
            
            if (i === wallPos[0] && j === wallPos[1]) {
                cell.style.backgroundColor = 'rgba(50,50,50,0.8)';
                cell.innerText = 'W';
            }
        }
    }
}

// Step Simulation
function step() {
    if (!qData || !qData[currentModel]) return;
    
    // Check if terminal
    if (agentPos.toString() === goalPos.toString() || agentPos.toString() === pitPos.toString()) {
        resetEnv();
        return;
    }

    let action;
    if (Math.random() < epsilon) {
        action = Math.floor(Math.random() * 4); // Explore
    } else {
        const qVals = qData[currentModel][agentPos[0]][agentPos[1]];
        action = qVals.indexOf(Math.max(...qVals)); // Exploit
    }

    // Move (0: up, 1: down, 2: left, 3: right)
    let nx = agentPos[0];
    let ny = agentPos[1];
    
    if (action === 0) nx--;
    else if (action === 1) nx++;
    else if (action === 2) ny--;
    else if (action === 3) ny++;

    // Boundaries
    nx = Math.max(0, Math.min(SIZE - 1, nx));
    ny = Math.max(0, Math.min(SIZE - 1, ny));

    // Wall collision
    if (nx === wallPos[0] && ny === wallPos[1]) {
        nx = agentPos[0];
        ny = agentPos[1];
    }

    agentPos = [nx, ny];
    steps++;

    // Calculate Reward
    if (agentPos.toString() === goalPos.toString()) {
        totalReward += 2.0;
        if (isPlaying) {
            setTimeout(resetEnv, speed);
        }
    } else if (agentPos.toString() === pitPos.toString()) {
        totalReward -= 1.0;
        if (isPlaying) {
            setTimeout(resetEnv, speed);
        }
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
            if (agentPos.toString() !== goalPos.toString() && agentPos.toString() !== pitPos.toString()) {
                step();
            }
        }, speed);
    } else {
        btnPlay.innerText = "Auto Play";
        btnPlay.classList.replace('btn-secondary', 'btn-primary');
        clearInterval(playInterval);
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

speedSlider.addEventListener('input', (e) => {
    speed = parseInt(e.target.value);
    if (isPlaying) {
        clearInterval(playInterval);
        playInterval = setInterval(step, speed);
    }
});

btnReset.addEventListener('click', resetEnv);
btnStep.addEventListener('click', step);
btnPlay.addEventListener('click', togglePlay);

// Load Data and Init
fetch('q_values.json')
    .then(res => res.json())
    .then(data => {
        qData = data;
        initGrids();
        resetEnv();
    })
    .catch(err => {
        console.error("Failed to load Q-values", err);
        alert("Failed to load Q-values. Please ensure q_values.json exists and run on a local server.");
    });
