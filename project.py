import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import random
import networkx as nx

# Modern color scheme
COLORS = {
    'bg_dark': '#1e1e2e',
    'bg_light': '#313244',
    'accent1': '#89b4fa',  # Blue
    'accent2': '#f5c2e7',  # Pink
    'accent3': '#a6e3a1',  # Green
    'accent4': '#fab387',  # Orange
    'text': '#cdd6f4',
    'text_dim': '#9399b2'
}

class ModernButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config(
            relief=tk.FLAT,
            borderwidth=0,
            padx=20,
            pady=10,
            font=('Helvetica', 12),
            cursor='hand2'
        )
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

    def on_enter(self, e):
        self['background'] = self.darker(self['background'], 20)

    def on_leave(self, e):
        self['background'] = self.lighter(self['background'], 20)

    def darker(self, color, factor=30):
        rgb = self.winfo_rgb(color)
        return '#{:02x}{:02x}{:02x}'.format(
            *[max(0, min(255, int(c/256 - factor))) for c in rgb]
        )

    def lighter(self, color, factor=30):
        rgb = self.winfo_rgb(color)
        return '#{:02x}{:02x}{:02x}'.format(
            *[max(0, min(255, int(c/256 + factor))) for c in rgb]
        )

class ModernListbox(tk.Listbox):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config(
            relief=tk.FLAT,
            borderwidth=0,
            selectmode=tk.SINGLE,
            activestyle='none'
        )

class CustomNotebook(ttk.Notebook):
    def __init__(self, *args, **kwargs):
        ttk.Notebook.__init__(self, *args, **kwargs)
        self._active = None
        
        self.bind("<ButtonPress-1>", self.on_tab_press, True)
        self.bind("<ButtonRelease-1>", self.on_tab_release)

    def on_tab_press(self, event):
        try:
            element = self.identify(event.x, event.y)
            self._active = self.index("@%d,%d" % (event.x, event.y))
        except:
            pass

    def on_tab_release(self, event):
        if not self._active:
            return
        try:
            index = self.index("@%d,%d" % (event.x, event.y))
        except:
            index = -1
        if index != -1 and index == self._active:
            self.select(index)
        self._active = None

class ResourceAllocationGraph:
    def __init__(self):
        self.processes = set()
        self.resources = {}  # resource_id: total_instances
        self.allocation = {}  # (process_id, resource_id): instances
        self.requests = {}   # (process_id, resource_id): instances
        
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
        
        # Add all processes and resources as nodes
        for p in self.processes:
            G.add_node(f"P{p}", type="process")
        for r in self.resources:
            G.add_node(f"R{r}", type="resource")
            
        # Add edges for allocations (Resource → Process)
        for (pid, rid), instances in self.allocation.items():
            G.add_edge(f"R{rid}", f"P{pid}", weight=instances)
            
        # Add edges for requests (Process → Resource)
        for (pid, rid), instances in self.requests.items():
            G.add_edge(f"P{pid}", f"R{rid}", weight=instances)
            
        # Detect cycles in the graph
        try:
            cycles = list(nx.simple_cycles(G))
            return len(cycles) > 0, cycles
        except:
            return False, []
            
    def clear(self):
        self.processes.clear()
        self.resources.clear()
        self.allocation.clear()
        self.requests.clear()

class SchedulerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Process Scheduler with Deadlock Detection")
        self.root.geometry("1400x900")
        self.root.config(bg=COLORS['bg_dark'])
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TNotebook', background=COLORS['bg_dark'])
        self.style.configure('TNotebook.Tab', padding=[20, 10], font=('Helvetica', 12))
        self.style.configure('TFrame', background=COLORS['bg_dark'])
        
        self.font_large = ("Helvetica", 32, "bold")
        self.font_med = ("Helvetica", 24)
        self.font_small = ("Helvetica", 18)
        
        self.processes = []
        self.quantum = 2
        self.rag = ResourceAllocationGraph()

        self.setup_ui()

    def setup_ui(self):
        # Create notebook with custom styling
        self.notebook = CustomNotebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Scheduler Tab with modern styling
        self.scheduler_frame = ttk.Frame(self.notebook)
        self.scheduler_frame.configure(style='TFrame')
        self.notebook.add(self.scheduler_frame, text="Process Scheduler")
        
        # Deadlock Tab with modern styling
        self.deadlock_frame = ttk.Frame(self.notebook)
        self.deadlock_frame.configure(style='TFrame')
        self.notebook.add(self.deadlock_frame, text="Deadlock Detection")
        
        self.setup_scheduler_ui()
        self.setup_deadlock_ui()

    def setup_scheduler_ui(self):
        # Title with modern styling
        title = tk.Label(
            self.scheduler_frame,
            text="Process Scheduling Simulator",
            font=self.font_large,
            bg=COLORS['bg_dark'],
            fg=COLORS['accent1']
        )
        title.pack(pady=20)

        # Control Panel Frame
        control_frame = tk.Frame(self.scheduler_frame, bg=COLORS['bg_dark'])
        control_frame.pack(fill='x', padx=20)

        # Add Process Button with modern styling
        add_btn = ModernButton(
            control_frame,
            text="Add Process",
            font=self.font_small,
            bg=COLORS['accent1'],
            fg=COLORS['bg_dark'],
            command=self.add_process
        )
        add_btn.pack(side=tk.LEFT, padx=10)

        # Quantum Input Frame
        quantum_frame = tk.Frame(control_frame, bg=COLORS['bg_dark'])
        quantum_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(
            quantum_frame,
            text="RR Quantum:",
            font=self.font_small,
            bg=COLORS['bg_dark'],
            fg=COLORS['text']
        ).pack(side=tk.LEFT)
        
        self.quantum_entry = tk.Spinbox(
            quantum_frame,
            from_=1,
            to=20,
            width=3,
            font=self.font_small,
            bg=COLORS['bg_light'],
            fg=COLORS['text'],
            buttonbackground=COLORS['accent1'],
            relief=tk.FLAT
        )
        self.quantum_entry.pack(side=tk.LEFT, padx=5)
        self.quantum_entry.delete(0, "end")
        self.quantum_entry.insert(0, "2")

        # Process List Frame
        list_frame = tk.Frame(self.scheduler_frame, bg=COLORS['bg_dark'])
        list_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Modern Listbox
        self.proc_listbox = ModernListbox(
            list_frame,
            width=50,
            height=8,
            font=self.font_small,
            bg=COLORS['bg_light'],
            fg=COLORS['text'],
            selectbackground=COLORS['accent1'],
            selectforeground=COLORS['bg_dark']
        )
        self.proc_listbox.pack(fill='both', expand=True)

        # Button Frame
        btn_frame = tk.Frame(self.scheduler_frame, bg=COLORS['bg_dark'])
        btn_frame.pack(pady=20)

        # Run Button
        self.start_btn = ModernButton(
            btn_frame,
            text="Run Simulation",
            font=self.font_small,
            bg=COLORS['accent3'],
            fg=COLORS['bg_dark'],
            command=self.start_sim
        )
        self.start_btn.pack(side=tk.LEFT, padx=10)

        # Reset Button
        self.reset_btn = ModernButton(
            btn_frame,
            text="Reset",
            font=self.font_small,
            bg=COLORS['accent2'],
            fg=COLORS['bg_dark'],
            command=self.reset_all
        )
        self.reset_btn.pack(side=tk.LEFT, padx=10)

        # Stats Frame
        self.stats_frame = tk.Frame(self.scheduler_frame, bg=COLORS['bg_dark'])
        self.stats_frame.pack(pady=20)
        
        # Stats Labels with modern styling
        self.avg_waiting_var = tk.StringVar(value="Average Waiting Time: N/A")
        self.avg_turnaround_var = tk.StringVar(value="Average Turnaround Time: N/A")
        self.avg_response_var = tk.StringVar(value="Average Response Time: N/A")
        
        for var in [self.avg_waiting_var, self.avg_turnaround_var, self.avg_response_var]:
            tk.Label(
                self.stats_frame,
                textvariable=var,
                font=self.font_small,
                bg=COLORS['bg_dark'],
                fg=COLORS['text']
            ).pack(pady=5)

        # Gantt Chart
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        self.setup_gantt_chart()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.scheduler_frame)

    def setup_deadlock_ui(self):
        # Title
        title = tk.Label(
            self.deadlock_frame,
            text="Resource Allocation Graph",
            font=self.font_large,
            bg=COLORS['bg_dark'],
            fg=COLORS['accent1']
        )
        title.pack(pady=20)

        # Control Panel
        btn_frame = tk.Frame(self.deadlock_frame, bg=COLORS['bg_dark'])
        btn_frame.pack(pady=20)

        # Modern buttons for deadlock detection
        buttons = [
            ("Add Resource", COLORS['accent1'], self.add_resource),
            ("Add Request", COLORS['accent2'], self.add_request),
            ("Add Allocation", COLORS['accent3'], self.add_allocation),
            ("Detect Deadlock", COLORS['accent4'], self.detect_deadlock),
            ("Reset", COLORS['accent2'], self.reset_rag)
        ]

        for text, color, command in buttons:
            ModernButton(
                btn_frame,
                text=text,
                font=self.font_small,
                bg=color,
                fg=COLORS['bg_dark'],
                command=command
            ).pack(side=tk.LEFT, padx=10)

        # RAG Visualization Frame
        self.rag_frame = tk.Frame(self.deadlock_frame, bg=COLORS['bg_dark'])
        self.rag_frame.pack(pady=20, fill=tk.BOTH, expand=True)

        # Create figure for RAG visualization with modern styling
        self.rag_fig, self.rag_ax = plt.subplots(figsize=(12, 8))
        self.rag_canvas = FigureCanvasTkAgg(self.rag_fig, master=self.rag_frame)
        self.setup_rag_chart()
        self.rag_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    def setup_rag_chart(self):
        self.rag_ax.clear()
        self.rag_ax.set_facecolor(COLORS['bg_light'])
        self.rag_fig.patch.set_facecolor(COLORS['bg_dark'])
        self.rag_ax.set_title("Resource Allocation Graph", fontsize=24, color=COLORS['text'])
        self.rag_ax.set_xticks([])
        self.rag_ax.set_yticks([])
        self.rag_canvas.draw_idle()

    def setup_gantt_chart(self):
        self.ax.clear()
        self.ax.set_facecolor(COLORS['bg_light'])
        self.fig.patch.set_facecolor(COLORS['bg_dark'])
        self.ax.set_title("Gantt Chart", fontsize=24, color=COLORS['text'])
        self.ax.set_xlabel("Time", fontsize=20, color=COLORS['text'])
        self.ax.set_ylabel("Processes", fontsize=20, color=COLORS['text'])
        self.ax.tick_params(colors=COLORS['text'])
        self.ax.grid(True, which='both', axis='x', linestyle='--', color=COLORS['text_dim'], alpha=0.3)

    def add_resource(self):
        rid = simpledialog.askstring("Input", "Enter Resource ID (e.g., R1):", parent=self.root)
        if not rid:
            return
        instances = simpledialog.askinteger("Input", "Enter number of instances:", parent=self.root, minvalue=1)
        if not instances:
            return
        
        self.rag.add_resource(rid, instances)
        self.update_rag_visualization()
        messagebox.showinfo("Success", f"Resource {rid} with {instances} instances added.")

    def add_request(self):
        if not self.processes:
            messagebox.showerror("Error", "No processes available. Add processes first.")
            return
            
        pid = simpledialog.askinteger("Input", "Enter Process ID:", parent=self.root, minvalue=1)
        if not pid:
            return
            
        rid = simpledialog.askstring("Input", "Enter Resource ID:", parent=self.root)
        if not rid:
            return
            
        instances = simpledialog.askinteger("Input", "Enter number of instances:", parent=self.root, minvalue=1)
        if not instances:
            return
            
        if self.rag.request(pid, rid, instances):
            self.update_rag_visualization()
            messagebox.showinfo("Success", f"Request from P{pid} for {instances} instances of {rid} added.")
        else:
            messagebox.showerror("Error", "Invalid request. Check resource availability.")

    def add_allocation(self):
        if not self.processes:
            messagebox.showerror("Error", "No processes available. Add processes first.")
            return
            
        pid = simpledialog.askinteger("Input", "Enter Process ID:", parent=self.root, minvalue=1)
        if not pid:
            return
            
        rid = simpledialog.askstring("Input", "Enter Resource ID:", parent=self.root)
        if not rid:
            return
            
        instances = simpledialog.askinteger("Input", "Enter number of instances:", parent=self.root, minvalue=1)
        if not instances:
            return
            
        if self.rag.allocate(pid, rid, instances):
            self.update_rag_visualization()
            messagebox.showinfo("Success", f"Allocation of {instances} instances of {rid} to P{pid} added.")
        else:
            messagebox.showerror("Error", "Invalid allocation. Check resource availability.")

    def detect_deadlock(self):
        has_deadlock, cycles = self.rag.detect_deadlock()
        if has_deadlock:
            cycle_str = "\n".join([" → ".join(cycle) for cycle in cycles])
            messagebox.showwarning("Deadlock Detected", f"Deadlock detected!\nCycles found:\n{cycle_str}")
        else:
            messagebox.showinfo("No Deadlock", "No deadlock detected in the system.")

    def reset_rag(self):
        self.rag.clear()
        self.setup_rag_chart()
        messagebox.showinfo("Reset", "Resource Allocation Graph has been reset.")

    def update_rag_visualization(self):
        self.rag_ax.clear()
        self.rag_ax.set_facecolor(COLORS['bg_light'])
        G = nx.DiGraph()
        
        # Add nodes
        process_nodes = [f"P{p}" for p in self.rag.processes]
        resource_nodes = list(self.rag.resources.keys())
        
        pos = {}
        
        # Position processes on the left
        for i, p in enumerate(process_nodes):
            pos[p] = (-1, -i)
            G.add_node(p, node_type='process')
            
        # Position resources on the right
        for i, r in enumerate(resource_nodes):
            pos[r] = (1, -i)
            G.add_node(r, node_type='resource')
            
        # Add edges
        for (pid, rid), instances in self.rag.allocation.items():
            G.add_edge(rid, f"P{pid}", weight=instances, edge_type='allocation')
            
        for (pid, rid), instances in self.rag.requests.items():
            G.add_edge(f"P{pid}", rid, weight=instances, edge_type='request')
            
        # Draw the graph with modern styling
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=process_nodes, 
                             node_color=COLORS['accent1'],
                             node_size=1500, 
                             node_shape='o', 
                             ax=self.rag_ax)
        
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=resource_nodes, 
                             node_color=COLORS['accent3'],
                             node_size=1500, 
                             node_shape='s', 
                             ax=self.rag_ax)
        
        # Draw edges with different styles for allocation and request
        allocation_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('edge_type') == 'allocation']
        request_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('edge_type') == 'request']
        
        nx.draw_networkx_edges(G, pos, 
                             edgelist=allocation_edges, 
                             edge_color=COLORS['accent3'],
                             arrows=True,
                             arrowsize=30, 
                             ax=self.rag_ax,
                             width=2)
        
        nx.draw_networkx_edges(G, pos, 
                             edgelist=request_edges, 
                             edge_color=COLORS['accent2'],
                             arrows=True,
                             arrowsize=30, 
                             ax=self.rag_ax,
                             width=2)
        
        # Draw labels with modern styling
        nx.draw_networkx_labels(G, pos, 
                              font_size=14,
                              font_weight='bold',
                              font_color=COLORS['bg_dark'],
                              ax=self.rag_ax)
        
        self.rag_ax.set_title("Resource Allocation Graph", 
                             fontsize=24, 
                             color=COLORS['text'],
                             pad=20)
        self.rag_ax.set_xticks([])
        self.rag_ax.set_yticks([])
        self.rag_canvas.draw_idle()

    def add_process(self):
        try:
            arrival_time = simpledialog.askinteger("Input", "Enter Arrival Time (non-negative):", minvalue=0, parent=self.root)
            if arrival_time is None:
                return
            burst_time = simpledialog.askinteger("Input", "Enter Burst Time (positive):", minvalue=1, parent=self.root)
            if burst_time is None:
                return
            priority = simpledialog.askinteger("Input", "Enter Priority (1=Highest):", minvalue=1, parent=self.root)
            if priority is None:
                return
            pid = len(self.processes) + 1
            proc_desc = f"P{pid} | Arrival: {arrival_time} | Burst: {burst_time} | Priority: {priority}"
            self.processes.append({'pid': pid, 'arrival': arrival_time, 'burst': burst_time, 'priority': priority})
            self.proc_listbox.insert(tk.END, proc_desc)
            self.rag.add_process(pid)
        except Exception:
            messagebox.showerror("Error", "Invalid input! Please enter valid integers.", parent=self.root)

    def reset_all(self):
        self.processes.clear()
        self.proc_listbox.delete(0, tk.END)
        self.clear_canvas()
        self.canvas.get_tk_widget().pack_forget()
        self.avg_waiting_var.set("Average Waiting Time: N/A")
        self.avg_turnaround_var.set("Average Turnaround Time: N/A")
        self.avg_response_var.set("Average Response Time: N/A")
        self.reset_rag()

    def start_sim(self):
        if not self.processes:
            messagebox.showwarning("No Processes", "Please add at least one process.", parent=self.root)
            return
        try:
            self.quantum = int(self.quantum_entry.get())
        except Exception:
            messagebox.showerror("Invalid Quantum", "Quantum must be an integer.", parent=self.root)
            return
        
        # Suggest the best algorithm automatically
        algorithm = self.suggest_best_algorithm()
        messagebox.showinfo("Suggested Algorithm", f"The suggested algorithm is: {algorithm}", parent=self.root)

        threading.Thread(target=self.run_scheduler, args=(algorithm,), daemon=True).start()

    def suggest_best_algorithm(self):
        if len(self.processes) == 0:
            return "First Come First Serve"

        burst_times = [p['burst'] for p in self.processes]
        priorities = [p['priority'] for p in self.processes]
        arrival_times = [p['arrival'] for p in self.processes]

        arrival_range = max(arrival_times) - min(arrival_times) if arrival_times else 0
        burst_range = max(burst_times) - min(burst_times) if burst_times else 0

        if arrival_range <= 1 and burst_range == 0:
            return "First Come First Serve"
        elif burst_range > 0 and len(set(burst_times)) == len(burst_times):
            return "Shortest Job First"
        elif len(set(priorities)) < len(priorities):
            return "Priority Scheduling"
        else:
            return "Round Robin"

    def run_scheduler(self, algo):
        proc_list = [dict(p) for p in self.processes]
        if algo == "First Come First Serve":
            schedule, stats = self.FCFS(proc_list)
        elif algo == "Shortest Job First":
            schedule, stats = self.SJF(proc_list)
        elif algo == "Round Robin":
            schedule, stats = self.RR(proc_list, self.quantum)
        elif algo == "Priority Scheduling":
            schedule, stats = self.Priority(proc_list)
        else:
            schedule, stats = [], {}
        self.root.after(0, lambda: self.show_and_animate_gantt(schedule, stats))

    def FCFS(self, procs):
        schedule = []
        procs = sorted(procs, key=lambda p: p['arrival'])
        current_time = 0
        waiting_times = {}
        response_times = {}
        start_times = {}
        for p in procs:
            start = max(current_time, p['arrival'])
            if p['pid'] not in start_times:
                start_times[p['pid']] = start
                response_times[p['pid']] = start - p['arrival']
            finish = start + p['burst']
            waiting_times[p['pid']] = start - p['arrival']
            schedule.append({'pid': p['pid'], 'start': start, 'end': finish})
            current_time = finish
        turnaround_times = {s['pid']: s['end'] - next(p['arrival'] for p in procs if p['pid'] == s['pid']) for s in schedule}
        avg_wt = sum(waiting_times.values()) / len(waiting_times) if waiting_times else 0
        avg_tat = sum(turnaround_times.values()) / len(turnaround_times) if turnaround_times else 0
        avg_rt = sum(response_times.values()) / len(response_times) if response_times else 0
        stats = {'avg_wt': avg_wt, 'avg_tat': avg_tat, 'avg_rt': avg_rt}
        return schedule, stats

    def SJF(self, procs):
        procs = sorted(procs, key=lambda p: p['arrival'])
        ready_queue = []
        schedule = []
        time = 0
        left = procs.copy()
        waiting_times = {}
        response_times = {}
        first_response = {}
        while left or ready_queue:
            for p in left:
                if p['arrival'] <= time:
                    ready_queue.append(p)
            left = [p for p in left if p['arrival'] > time]
            if ready_queue:
                p = min(ready_queue, key=lambda p: p['burst'])
                ready_queue.remove(p)
                start = max(time, p['arrival'])
                if p['pid'] not in first_response:
                    first_response[p['pid']] = start - p['arrival']
                finish = start + p['burst']
                waiting_times[p['pid']] = start - p['arrival']
                schedule.append({'pid': p['pid'], 'start': start, 'end': finish})
                time = finish
            else:
                time += 1
        turnaround_times = {s['pid']: s['end'] - next(p['arrival'] for p in procs if p['pid'] == s['pid']) for s in schedule}
        avg_wt = sum(waiting_times.values()) / len(waiting_times) if waiting_times else 0
        avg_tat = sum(turnaround_times.values()) / len(turnaround_times) if turnaround_times else 0
        avg_rt = sum(first_response.values()) / len(first_response) if first_response else 0
        stats = {'avg_wt': avg_wt, 'avg_tat': avg_tat, 'avg_rt': avg_rt}
        return schedule, stats

    def RR(self, procs, quantum):
        procs = [dict(p) for p in procs]
        for p in procs:
            p['remaining'] = p['burst']
        ready_queue = []
        schedule = []
        time = 0
        left = procs
        waiting_times = {p['pid']: 0 for p in procs}
        first_response = {}

        while left or ready_queue:
            for p in left:
                if p['arrival'] <= time and p not in ready_queue:
                    ready_queue.append(p)
            left = [p for p in left if p['arrival'] > time]
            if ready_queue:
                p = ready_queue.pop(0)
                start_time = time
                if p['pid'] not in first_response:
                    first_response[p['pid']] = start_time - p['arrival']
                exec_time = min(p['remaining'], quantum)
                time += exec_time
                p['remaining'] -= exec_time
                schedule.append({'pid': p['pid'], 'start': start_time, 'end': time})
                for q in ready_queue:
                    waiting_times[q['pid']] += exec_time
                if p['remaining'] > 0:
                    ready_queue.append(p)
                else:
                    waiting_times[p['pid']] = waiting_times.get(p['pid'], 0) + time - p['arrival'] - p['burst']
            else:
                time += 1

        turnaround_times = {s['pid']: s['end'] - next(p['arrival'] for p in procs if p['pid'] == s['pid']) for s in schedule}
        avg_wt = sum(waiting_times.values()) / len(waiting_times) if waiting_times else 0
        avg_tat = sum(turnaround_times.values()) / len(turnaround_times) if turnaround_times else 0
        avg_rt = sum(first_response.values()) / len(first_response) if first_response else 0
        stats = {'avg_wt': avg_wt, 'avg_tat': avg_tat, 'avg_rt': avg_rt}
        return schedule, stats

    def Priority(self, procs):
        schedule = []
        time = 0
        procs = sorted(procs, key=lambda p: (p['arrival'], p['priority']))
        left = procs.copy()
        waiting_times = {}
        first_response = {}
        while left:
            available = [p for p in left if p['arrival'] <= time]
            if not available:
                time = left[0]['arrival']
                continue
            p = min(available, key=lambda x: x['priority'])
            start = max(time, p['arrival'])
            if p['pid'] not in first_response:
                first_response[p['pid']] = start - p['arrival']
            finish = start + p['burst']
            waiting_times[p['pid']] = start - p['arrival']
            schedule.append({'pid': p['pid'], 'start': start, 'end': finish})
            time = finish
            left.remove(p)

        turnaround_times = {s['pid']: s['end'] - next(p['arrival'] for p in procs if p['pid'] == s['pid']) for s in schedule}
        avg_wt = sum(waiting_times.values()) / len(waiting_times) if waiting_times else 0
        avg_tat = sum(turnaround_times.values()) / len(turnaround_times) if turnaround_times else 0
        avg_rt = sum(first_response.values()) / len(first_response) if first_response else 0
        stats = {'avg_wt': avg_wt, 'avg_tat': avg_tat, 'avg_rt': avg_rt}
        return schedule, stats

    def show_and_animate_gantt(self, schedule, stats):
        self.canvas.get_tk_widget().pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        self.clear_canvas()
        
        # Modern color palette for processes
        colors = [COLORS['accent1'], COLORS['accent2'], COLORS['accent3'], 
                 COLORS['accent4'], '#f38ba8', '#fab387', '#89dceb']
        p_colors = {}
        for p in self.processes:
            if p['pid'] not in p_colors:
                p_colors[p['pid']] = random.choice(colors)

        max_time = max([task['end'] for task in schedule]) if schedule else 1
        bars = []
        for task in schedule:
            pid = task['pid']
            start = task['start']
            end = task['end']
            color = p_colors.get(pid, COLORS['accent1'])
            bar = mpatches.Rectangle((start, 0.5), 0, 0.5, 
                                   facecolor=color,
                                   edgecolor=COLORS['text'],
                                   linewidth=2)
            self.ax.add_patch(bar)
            bars.append((bar, start, end, f"P{pid}"))

        self.ax.set_xlim(0, max_time + 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks(range(0, int(max_time) + 2))
        self.ax.tick_params(axis='x', colors=COLORS['text'], labelsize=14)
        self.ax.tick_params(axis='y', colors=COLORS['text'], labelsize=14)
        self.ax.set_yticks([])
        self.ax.set_facecolor(COLORS['bg_light'])

        # Modern legend
        patches = [mpatches.Patch(color=color, 
                                label=f"P{pid}",
                                edgecolor=COLORS['text'],
                                linewidth=1) 
                  for pid, color in p_colors.items()]
        
        self.ax.legend(handles=patches,
                      bbox_to_anchor=(1.05, 1),
                      loc='upper left',
                      fontsize=14,
                      facecolor=COLORS['bg_light'],
                      edgecolor=COLORS['text_dim'],
                      labelcolor=COLORS['text'])
        
        self.ax.set_title("Gantt Chart", 
                         fontsize=24,
                         color=COLORS['text'],
                         pad=20)
        self.ax.set_xlabel("Time",
                          fontsize=20,
                          color=COLORS['text'])
        self.canvas.draw_idle()

        def animate_bar(idx=0):
            if idx >= len(bars):
                self.avg_waiting_var.set(f"Average Waiting Time: {stats['avg_wt']:.2f}")
                self.avg_turnaround_var.set(f"Average Turnaround Time: {stats['avg_tat']:.2f}")
                self.avg_response_var.set(f"Average Response Time: {stats['avg_rt']:.2f}")
                return
            bar, start, end, label = bars[idx]
            width = end - start
            step = width / 30.0
            cur_width = 0

            def grow():
                nonlocal cur_width
                if cur_width >= width:
                    self.ax.text(start + width / 2, 0.75,
                               label,
                               ha='center',
                               va='center',
                               fontsize=16,
                               color=COLORS['text'],
                               fontweight='bold')
                    self.canvas.draw_idle()
                    animate_bar(idx + 1)
                    return
                cur_width += step
                bar.set_width(cur_width)
                self.canvas.draw_idle()
                self.root.after(30, grow)
            grow()

        animate_bar()

    def clear_canvas(self):
        self.ax.cla()
        self.ax.set_title("Gantt Chart", fontsize=28, color='black')
        self.ax.set_xlabel("Time", fontsize=24, color='black')
        self.ax.set_ylabel("Processes", fontsize=22, color='black')
        self.ax.set_yticks([])
        self.ax.grid(True, which='both', axis='x', linestyle='--', color='gray', alpha=0.5)
        self.ax.set_facecolor('#FFFFFF')
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = SchedulerApp(root)
    root.mainloop()