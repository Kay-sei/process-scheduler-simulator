from flask import Flask, render_template, request, jsonify
import networkx as nx
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import io
import base64
import json

app = Flask(__name__)

class ResourceAllocationGraph:
    def __init__(self):
        self.processes = set()
        self.resources = {}
        self.allocation = {}
        self.requests = {}
        
    def add_process(self, pid):
        self.processes.add(pid)
        
    def add_resource(self, rid, instances):
        self.resources[rid] = instances
        
    def allocate(self, pid, rid, instances):
        if instances <= self.resources.get(rid, 0):
            self.allocation[(pid, rid)] = instances
            return True
        return False
        
    def request(self, pid, rid, instances):
        if instances <= self.resources.get(rid, 0):
            self.requests[(pid, rid)] = instances
            return True
        return False
        
    def detect_deadlock(self):
        G = nx.DiGraph()
        
        for p in self.processes:
            G.add_node(f"P{p}", type="process")
        for r in self.resources:
            G.add_node(f"R{r}", type="resource")
            
        for (pid, rid), instances in self.allocation.items():
            G.add_edge(f"R{rid}", f"P{pid}", weight=instances)
            
        for (pid, rid), instances in self.requests.items():
            G.add_edge(f"P{pid}", f"R{rid}", weight=instances)
            
        try:
            cycles = list(nx.simple_cycles(G))
            return len(cycles) > 0, cycles, self.get_graph_data(G)
        except:
            return False, [], self.get_graph_data(G)
    
    def get_graph_data(self, G):
        nodes = []
        edges = []
        
        for node in G.nodes():
            node_type = "process" if node.startswith("P") else "resource"
            nodes.append({
                "id": node,
                "label": node,
                "type": node_type
            })
        
        for (u, v) in G.edges():
            edge_type = "allocation" if u.startswith("R") else "request"
            edges.append({
                "from": u,
                "to": v,
                "type": edge_type
            })
            
        return {"nodes": nodes, "edges": edges}
            
    def clear(self):
        self.processes.clear()
        self.resources.clear()
        self.allocation.clear()
        self.requests.clear()

# Global RAG instance
rag = ResourceAllocationGraph()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scheduler')
def scheduler():
    return render_template('scheduler.html')

@app.route('/rag')
def rag_simulator():
    return render_template('rag.html')

@app.route('/api/process', methods=['POST'])
def add_process():
    data = request.json
    pid = data.get('pid')
    if pid:
        rag.add_process(int(pid))
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Invalid process ID"})

@app.route('/api/resource', methods=['POST'])
def add_resource():
    data = request.json
    rid = data.get('rid')
    instances = data.get('instances')
    if rid and instances:
        rag.add_resource(rid, int(instances))
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Invalid resource data"})

@app.route('/api/allocate', methods=['POST'])
def allocate_resource():
    data = request.json
    pid = data.get('pid')
    rid = data.get('rid')
    instances = data.get('instances')
    if pid and rid and instances:
        success = rag.allocate(int(pid), rid, int(instances))
        return jsonify({"status": "success" if success else "error"})
    return jsonify({"status": "error", "message": "Invalid allocation data"})

@app.route('/api/request', methods=['POST'])
def request_resource():
    data = request.json
    pid = data.get('pid')
    rid = data.get('rid')
    instances = data.get('instances')
    if pid and rid and instances:
        success = rag.request(int(pid), rid, int(instances))
        return jsonify({"status": "success" if success else "error"})
    return jsonify({"status": "error", "message": "Invalid request data"})

@app.route('/api/detect_deadlock', methods=['GET'])
def detect_deadlock():
    has_deadlock, cycles, graph_data = rag.detect_deadlock()
    return jsonify({
        "has_deadlock": has_deadlock,
        "cycles": cycles,
        "graph_data": graph_data
    })

@app.route('/api/reset', methods=['POST'])
def reset_rag():
    rag.clear()
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True) 