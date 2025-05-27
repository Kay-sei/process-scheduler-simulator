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

## Technologies Used

- Python
- Flask
- NetworkX
- HTML/CSS (Tailwind CSS)
- JavaScript

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd <repository-name>
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\activate
   # On Unix or MacOS:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your web browser and navigate to:
   - http://localhost:5000 - Home page
   - http://localhost:5000/scheduler - Process Scheduler
   - http://localhost:5000/rag - Resource Allocation Graph

## Usage

### Process Scheduler
1. Select scheduling algorithm
2. Add processes with burst time, arrival time, and priority (if applicable)
3. Set time quantum for Round Robin
4. Start simulation to view Gantt chart and statistics

### RAG Simulator
1. Add processes and resources
2. Create allocations and requests
3. Detect deadlocks using the graph visualization

## License

MIT License 