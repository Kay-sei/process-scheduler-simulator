# Process Scheduler and Deadlock Detection Simulator

A web-based application that simulates process scheduling algorithms and deadlock detection in operating systems.

## Features

- **Process Scheduler Simulator**
  - First Come First Serve (FCFS)
  - Shortest Job First (SJF)
  - Priority Scheduling
  - Round Robin

- **Resource Allocation Graph (RAG) Simulator**
  - Visual representation of resource allocation
  - Deadlock detection
  - Interactive process and resource management

## Quick Setup Guide

### Prerequisites
- Python 3.7 or higher
- Git (optional - for cloning)

### Method 1: Download ZIP
1. Go to `https://github.com/Kay-sei/process-scheduler-simulator`
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file to your desired location

### Method 2: Clone with Git
```bash
git clone https://github.com/Kay-sei/process-scheduler-simulator
cd process-scheduler-simulator
```

### Setting Up the Environment
1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
- On Windows:
```bash
.\.venv\Scripts\activate
```
- On Mac/Linux:
```bash
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Application
1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and go to:
- http://localhost:5000 - Home page
- http://localhost:5000/scheduler - Process Scheduler
- http://localhost:5000/rag - Resource Allocation Graph

## Troubleshooting
If you encounter any issues:
1. Make sure Python is installed and in your PATH
2. Ensure all requirements are installed correctly
3. Check if the virtual environment is activated (you should see (.venv) in your terminal)

## License

MIT License 