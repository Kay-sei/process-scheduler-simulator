// Process class to store process information
class Process {
    constructor(name, arrivalTime, burstTime, priority = 0) {
        this.name = name;
        this.arrivalTime = parseInt(arrivalTime);
        this.burstTime = parseInt(burstTime);
        this.priority = parseInt(priority);
        this.remainingTime = parseInt(burstTime);
        this.startTime = null;
        this.finishTime = null;
        this.status = 'Waiting';
    }

    get waitingTime() {
        return this.startTime - this.arrivalTime;
    }

    get turnaroundTime() {
        return this.finishTime - this.arrivalTime;
    }
}

// Global variables
let processes = [];
let currentTime = 0;
let isSimulationRunning = false;
let selectedAlgorithm = 'fcfs';
let timeQuantum = 2;

// Constants for Gantt chart
const PIXELS_PER_TIME_UNIT = 40;
const MAX_TIME_UNITS_VISIBLE = 20;
const SIMULATION_DELAY = 300; // Fixed 300ms delay for moderate speed

// Process color mapping
const PROCESS_COLORS = {
    'p1': { color: '#3B82F6', name: 'Blue' },
    'p2': { color: '#10B981', name: 'Green' },
    'p3': { color: '#F59E0B', name: 'Yellow' },
    'p4': { color: '#EF4444', name: 'Red' },
    'p5': { color: '#8B5CF6', name: 'Purple' },
    'p6': { color: '#EC4899', name: 'Pink' },
    'p7': { color: '#06B6D4', name: 'Cyan' },
    'p8': { color: '#F97316', name: 'Orange' },
    'p9': { color: '#6366F1', name: 'Indigo' },
    'p10': { color: '#14B8A6', name: 'Teal' }
};

// Track used process colors
let usedProcessColors = new Map();

// Event Listeners
document.getElementById('algorithm').addEventListener('change', function(e) {
    selectedAlgorithm = e.target.value;
    const timeQuantumContainer = document.getElementById('timeQuantumContainer');
    timeQuantumContainer.style.display = selectedAlgorithm === 'rr' ? 'block' : 'none';
});

// Add process to the queue
function addProcess() {
    const name = document.getElementById('processName').value;
    const arrivalTime = document.getElementById('arrivalTime').value;
    const burstTime = document.getElementById('burstTime').value;
    const priority = document.getElementById('priority').value;

    if (!name || !arrivalTime || !burstTime) {
        alert('Please fill in all required fields');
        return;
    }

    const process = new Process(name, arrivalTime, burstTime, priority);
    processes.push(process);
    updateProcessTable();
    clearInputFields();
}

// Clear input fields after adding process
function clearInputFields() {
    document.getElementById('processName').value = '';
    document.getElementById('arrivalTime').value = '';
    document.getElementById('burstTime').value = '';
    document.getElementById('priority').value = '';
}

// Update process table
function updateProcessTable() {
    const tableBody = document.getElementById('processTableBody');
    tableBody.innerHTML = '';

    processes.forEach(process => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap text-white">${process.name}</td>
            <td class="px-6 py-4 whitespace-nowrap text-white">${process.arrivalTime}</td>
            <td class="px-6 py-4 whitespace-nowrap text-white">${process.burstTime}</td>
            <td class="px-6 py-4 whitespace-nowrap text-white">${process.priority}</td>
            <td class="px-6 py-4 whitespace-nowrap">
                <span class="px-2 py-1 text-xs font-medium rounded-full 
                    ${process.status === 'Running' ? 'bg-green-600' : 
                    process.status === 'Completed' ? 'bg-blue-600' : 'bg-yellow-600'}">
                    ${process.status}
                </span>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

// Start simulation
function startSimulation() {
    if (processes.length === 0) {
        alert('Please add processes before starting simulation');
        return;
    }

    if (selectedAlgorithm === 'rr') {
        timeQuantum = parseInt(document.getElementById('timeQuantum').value) || 2;
    }

    isSimulationRunning = true;
    currentTime = 0;
    resetProcesses();

    // Sort processes by arrival time
    processes.sort((a, b) => a.arrivalTime - b.arrivalTime);

    switch (selectedAlgorithm) {
        case 'fcfs':
            runFCFS();
            break;
        case 'sjf':
            runSJF();
            break;
        case 'priority':
            runPriority();
            break;
        case 'rr':
            runRoundRobin();
            break;
    }
}

// Reset simulation
function resetSimulation() {
    processes = [];
    currentTime = 0;
    isSimulationRunning = false;
    updateProcessTable();
    updateStatistics();
    clearGanttChart();
}

// Reset processes for new simulation
function resetProcesses() {
    processes.forEach(process => {
        process.remainingTime = process.burstTime;
        process.startTime = null;
        process.finishTime = null;
        process.status = 'Waiting';
    });
    updateProcessTable();
}

// FCFS Algorithm
function runFCFS() {
    let currentIndex = 0;
    
    function executeNextProcess() {
        if (currentIndex >= processes.length) {
            finishSimulation();
            return;
        }

        const process = processes[currentIndex];
        
        if (currentTime < process.arrivalTime) {
            currentTime = process.arrivalTime;
        }

        process.startTime = currentTime;
        process.status = 'Running';
        updateProcessTable();
        updateGanttChart(process);

        setTimeout(() => {
            currentTime += process.burstTime;
            process.finishTime = currentTime;
            process.status = 'Completed';
            currentIndex++;
            updateProcessTable();
            executeNextProcess();
        }, SIMULATION_DELAY);
    }

    executeNextProcess();
}

// SJF Algorithm
function runSJF() {
    function getNextProcess() {
        return processes
            .filter(p => p.arrivalTime <= currentTime && p.remainingTime > 0)
            .sort((a, b) => a.remainingTime - b.remainingTime)[0];
    }

    function executeNextProcess() {
        const process = getNextProcess();
        
        if (!process) {
            if (processes.every(p => p.remainingTime === 0)) {
                finishSimulation();
                return;
            }
            currentTime++;
            setTimeout(executeNextProcess, SIMULATION_DELAY);
            return;
        }

        if (process.startTime === null) {
            process.startTime = currentTime;
        }
        
        process.status = 'Running';
        updateProcessTable();
        updateGanttChart(process);

        setTimeout(() => {
            process.remainingTime--;
            currentTime++;
            
            if (process.remainingTime === 0) {
                process.finishTime = currentTime;
                process.status = 'Completed';
            }
            
            updateProcessTable();
            executeNextProcess();
        }, SIMULATION_DELAY);
    }

    executeNextProcess();
}

// Priority Algorithm
function runPriority() {
    function getNextProcess() {
        return processes
            .filter(p => p.arrivalTime <= currentTime && p.remainingTime > 0)
            .sort((a, b) => b.priority - a.priority)[0];
    }

    function executeNextProcess() {
        const process = getNextProcess();
        
        if (!process) {
            if (processes.every(p => p.remainingTime === 0)) {
                finishSimulation();
                return;
            }
            currentTime++;
            setTimeout(executeNextProcess, SIMULATION_DELAY);
            return;
        }

        if (process.startTime === null) {
            process.startTime = currentTime;
        }
        
        process.status = 'Running';
        updateProcessTable();
        updateGanttChart(process);

        setTimeout(() => {
            process.remainingTime--;
            currentTime++;
            
            if (process.remainingTime === 0) {
                process.finishTime = currentTime;
                process.status = 'Completed';
            }
            
            updateProcessTable();
            executeNextProcess();
        }, SIMULATION_DELAY);
    }

    executeNextProcess();
}

// Round Robin Algorithm
function runRoundRobin() {
    let queue = [];
    let timeSlice = 0;

    function updateQueue() {
        processes
            .filter(p => p.arrivalTime <= currentTime && p.remainingTime > 0 && !queue.includes(p))
            .forEach(p => queue.push(p));
    }

    function executeNextProcess() {
        updateQueue();

        if (queue.length === 0) {
            if (processes.every(p => p.remainingTime === 0)) {
                finishSimulation();
                return;
            }
            currentTime++;
            setTimeout(executeNextProcess, SIMULATION_DELAY);
            return;
        }

        const process = queue.shift();
        
        if (process.startTime === null) {
            process.startTime = currentTime;
        }
        
        process.status = 'Running';
        updateProcessTable();
        updateGanttChart(process);

        setTimeout(() => {
            process.remainingTime--;
            currentTime++;
            timeSlice++;
            
            if (process.remainingTime === 0) {
                process.finishTime = currentTime;
                process.status = 'Completed';
                timeSlice = 0;
            } else if (timeSlice === timeQuantum) {
                queue.push(process);
                process.status = 'Waiting';
                timeSlice = 0;
            } else {
                queue.unshift(process);
            }
            
            updateProcessTable();
            executeNextProcess();
        }, SIMULATION_DELAY);
    }

    executeNextProcess();
}

// Update time axis with more detailed markers
function updateTimeAxis(maxTime) {
    const timeAxis = document.getElementById('timeAxis');
    timeAxis.innerHTML = '';
    
    for (let i = 0; i <= maxTime; i++) {
        const marker = document.createElement('div');
        marker.className = 'time-marker';
        marker.style.left = `${i * PIXELS_PER_TIME_UNIT}px`;
        
        // Add tick mark and number
        marker.innerHTML = `
            <div class="tick-mark"></div>
            <div class="time-number">${i}</div>
        `;
        
        timeAxis.appendChild(marker);
    }
}

// Get next available color class
function getProcessColorClass(processName) {
    if (!usedProcessColors.has(processName)) {
        const usedClasses = new Set(usedProcessColors.values());
        const availableClass = Object.keys(PROCESS_COLORS).find(key => !usedClasses.has(key));
        usedProcessColors.set(processName, availableClass || 'p1');
        updateProcessLegend();
    }
    return usedProcessColors.get(processName);
}

// Update process legend
function updateProcessLegend() {
    const legend = document.getElementById('processLegend');
    legend.innerHTML = '';
    
    // Add active processes to legend
    for (const [processName, colorClass] of usedProcessColors) {
        const colorInfo = PROCESS_COLORS[colorClass];
        const legendItem = document.createElement('div');
        legendItem.className = 'legend-item';
        legendItem.innerHTML = `
            <div class="legend-color" style="background-color: ${colorInfo.color}"></div>
            <span>${processName}</span>
        `;
        legend.appendChild(legendItem);
    }
}

// Update Gantt Chart
function updateGanttChart(process) {
    const ganttChart = document.getElementById('ganttChart');
    
    // Handle idle time
    if (process.arrivalTime > currentTime) {
        const idleBlock = document.createElement('div');
        idleBlock.className = 'process-block idle-block';
        const idleWidth = (process.arrivalTime - currentTime) * PIXELS_PER_TIME_UNIT;
        idleBlock.style.width = `${idleWidth}px`;
        idleBlock.style.left = `${currentTime * PIXELS_PER_TIME_UNIT}px`;
        idleBlock.innerHTML = `
            <span class="process-name">Idle</span>
        `;
        ganttChart.appendChild(idleBlock);
    }

    // Create process block
    const processBlock = document.createElement('div');
    const colorClass = getProcessColorClass(process.name);
    processBlock.className = `process-block ${colorClass}`;
    
    // Calculate time span and position
    let timeSpan;
    if (selectedAlgorithm === 'fcfs') {
        timeSpan = process.burstTime;
    } else {
        timeSpan = 1; // For other algorithms that execute one time unit at a time
    }
    
    const startTime = Math.max(currentTime, process.arrivalTime);
    const endTime = startTime + timeSpan;
    const width = timeSpan * PIXELS_PER_TIME_UNIT;
    
    processBlock.style.width = `${width}px`;
    processBlock.style.left = `${startTime * PIXELS_PER_TIME_UNIT}px`;
    
    processBlock.innerHTML = `
        <span class="process-name">${process.name}</span>
    `;
    
    // Add tooltip with process details
    processBlock.title = `Process ${process.name} (${startTime} → ${endTime})`;
    
    ganttChart.appendChild(processBlock);
    
    // Update time axis if needed
    updateTimeAxis(Math.max(endTime, ganttChart.dataset.maxTime || 0));
    ganttChart.dataset.maxTime = Math.max(endTime, ganttChart.dataset.maxTime || 0);
    
    // Scroll to keep current time visible
    const container = ganttChart.parentElement.parentElement;
    const scrollPosition = (endTime * PIXELS_PER_TIME_UNIT) - container.clientWidth + 100;
    if (scrollPosition > container.scrollLeft) {
        container.scrollLeft = scrollPosition;
    }
}

// Clear Gantt Chart
function clearGanttChart() {
    const ganttChart = document.getElementById('ganttChart');
    const timeAxis = document.getElementById('timeAxis');
    const legend = document.getElementById('processLegend');
    ganttChart.innerHTML = '';
    timeAxis.innerHTML = '';
    legend.innerHTML = '';
    ganttChart.dataset.maxTime = '0';
    usedProcessColors.clear();
}

// Update Statistics
function updateStatistics() {
    const completedProcesses = processes.filter(p => p.status === 'Completed');
    
    if (completedProcesses.length === 0) {
        document.getElementById('avgWaitingTime').textContent = '0.00';
        document.getElementById('avgTurnaroundTime').textContent = '0.00';
        return;
    }

    const avgWaitingTime = completedProcesses.reduce((sum, p) => sum + p.waitingTime, 0) / completedProcesses.length;
    const avgTurnaroundTime = completedProcesses.reduce((sum, p) => sum + p.turnaroundTime, 0) / completedProcesses.length;

    document.getElementById('avgWaitingTime').textContent = avgWaitingTime.toFixed(2);
    document.getElementById('avgTurnaroundTime').textContent = avgTurnaroundTime.toFixed(2);
}

// Finish simulation
function finishSimulation() {
    isSimulationRunning = false;
    updateStatistics();
} 