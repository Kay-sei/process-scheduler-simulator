// Initialize vis.js network
let network = null;
let data = null;
let isInitialized = false;

// Network options
const options = {
    nodes: {
        shape: 'circle',
        font: {
            color: '#ffffff',
            size: 16,
            face: 'Helvetica'
        },
        borderWidth: 2,
        shadow: true
    },
    edges: {
        arrows: 'to',
        smooth: {
            type: 'continuous',
            roundness: 0.1
        },
        width: 2,
        shadow: true
    },
    physics: {
        enabled: true,
        stabilization: {
            enabled: true,
            iterations: 1000,
            updateInterval: 100
        },
        barnesHut: {
            gravitationalConstant: -1000,
            centralGravity: 0.1,
            springLength: 200,
            springConstant: 0.01,
            damping: 0.5
        }
    },
    interaction: {
        zoomView: false,
        dragView: true,
        hover: true
    }
};

// Initialize network function
function initializeNetwork() {
    if (!isInitialized) {
        const container = document.getElementById('graph');
        if (container) {
            data = {
                nodes: new vis.DataSet([]),
                edges: new vis.DataSet([])
            };
            network = new vis.Network(container, data, options);
            isInitialized = true;
            console.log('Network initialized successfully');
        } else {
            console.error('Graph container not found');
        }
    }
}

// Helper function for API calls
async function apiCall(endpoint, method = 'GET', body = null) {
    try {
        const response = await fetch(endpoint, {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            },
            body: body ? JSON.stringify(body) : null
        });
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        showError('An error occurred while communicating with the server');
    }
}

// Show error message
function showError(message) {
    const status = document.getElementById('deadlockStatus');
    status.innerHTML = `<div class="bg-red-600 text-white p-4 rounded-lg">${message}</div>`;
    status.classList.remove('hidden');
}

// Show success message
function showSuccess(message) {
    const status = document.getElementById('deadlockStatus');
    status.innerHTML = `<div class="bg-green-600 text-white p-4 rounded-lg">${message}</div>`;
    status.classList.remove('hidden');
}

// Add process
async function addProcess() {
    if (!isInitialized) {
        showError('Network not initialized. Please refresh the page.');
        return;
    }

    const processId = document.getElementById('processId').value;
    if (!processId) {
        showError('Please enter a process ID');
        return;
    }

    try {
        console.log('Adding process with ID:', processId);
        const result = await apiCall('/api/process', 'POST', { pid: processId });
        console.log('Server response:', result);
        
        if (result.status === 'success') {
            const nodeData = {
                id: `P${processId}`,
                label: `P${processId}`,
                color: {
                    background: '#3B82F6',
                    border: '#2563EB'
                }
            };
            console.log('Adding node to visualization:', nodeData);
            data.nodes.add(nodeData);
            document.getElementById('processId').value = '';
            showSuccess(`Process P${processId} added successfully`);
        } else {
            showError(result.message || 'Failed to add process');
        }
    } catch (error) {
        console.error('Error adding process:', error);
        showError('An error occurred while adding the process');
    }
}

// Add resource
async function addResource() {
    if (!isInitialized) {
        showError('Network not initialized. Please refresh the page.');
        return;
    }

    const resourceId = document.getElementById('resourceId').value;
    const instances = document.getElementById('instances').value;
    
    if (!resourceId || !instances) {
        showError('Please enter both resource ID and number of instances');
        return;
    }

    const result = await apiCall('/api/resource', 'POST', { 
        rid: resourceId,
        instances: instances
    });

    if (result.status === 'success') {
        data.nodes.add({
            id: resourceId,
            label: `${resourceId}\n(${instances})`,
            shape: 'square',
            color: {
                background: '#10B981',
                border: '#059669'
            }
        });
        document.getElementById('resourceId').value = '';
        document.getElementById('instances').value = '';
        showSuccess(`Resource ${resourceId} with ${instances} instances added successfully`);
    } else {
        showError(result.message || 'Failed to add resource');
    }
}

// Add edge (allocation or request)
async function addEdge() {
    const edgeType = document.getElementById('edgeType').value;
    const processId = document.getElementById('edgeProcessId').value;
    const resourceId = document.getElementById('edgeResourceId').value;
    const instances = document.getElementById('edgeInstances').value;

    if (!processId || !resourceId || !instances) {
        showError('Please fill in all fields');
        return;
    }

    const endpoint = edgeType === 'allocation' ? '/api/allocate' : '/api/request';
    const result = await apiCall(endpoint, 'POST', {
        pid: processId,
        rid: resourceId,
        instances: instances
    });

    if (result.status === 'success') {
        const edgeColor = edgeType === 'allocation' ? '#10B981' : '#EF4444';
        const from = edgeType === 'allocation' ? resourceId : `P${processId}`;
        const to = edgeType === 'allocation' ? `P${processId}` : resourceId;

        data.edges.add({
            from: from,
            to: to,
            color: edgeColor,
            label: instances.toString()
        });

        document.getElementById('edgeProcessId').value = '';
        document.getElementById('edgeResourceId').value = '';
        document.getElementById('edgeInstances').value = '';
        showSuccess(`${edgeType} edge added successfully`);
    } else {
        showError(result.message || `Failed to add ${edgeType}`);
    }
}

// Detect deadlock
async function detectDeadlock() {
    const result = await apiCall('/api/detect_deadlock');
    
    if (result.has_deadlock) {
        // Highlight deadlock cycle
        const cycles = result.cycles;
        const cycleStr = cycles.map(cycle => cycle.join(' → ')).join('\n');
        
        const status = document.getElementById('deadlockStatus');
        status.innerHTML = `
            <div class="bg-red-600 text-white p-4 rounded-lg">
                <h4 class="font-bold text-lg mb-2">Deadlock Detected!</h4>
                <p class="font-mono">${cycleStr}</p>
            </div>
        `;
        status.classList.remove('hidden');

        // Highlight deadlocked nodes and edges
        const deadlockedNodes = new Set(cycles.flat());
        data.nodes.forEach(node => {
            if (deadlockedNodes.has(node.id)) {
                data.nodes.update({
                    id: node.id,
                    color: {
                        background: '#EF4444',
                        border: '#DC2626'
                    }
                });
            }
        });
    } else {
        showSuccess('No deadlock detected in the system');
    }
}

// Reset graph
async function resetGraph() {
    const result = await apiCall('/api/reset', 'POST');
    if (result.status === 'success') {
        data.nodes.clear();
        data.edges.clear();
        document.getElementById('deadlockStatus').classList.add('hidden');
        showSuccess('Graph reset successfully');
    } else {
        showError('Failed to reset graph');
    }
}

// Initialize event listeners
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded, initializing network...');
    initializeNetwork();
    
    // Enter key listeners
    document.getElementById('processId').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') addProcess();
    });
    
    document.getElementById('resourceId').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            if (document.getElementById('instances').value) {
                addResource();
            } else {
                document.getElementById('instances').focus();
            }
        }
    });
    
    document.getElementById('instances').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') addResource();
    });
}); 