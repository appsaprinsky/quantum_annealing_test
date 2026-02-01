import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dwave.samplers import SimulatedAnnealingSampler
import dimod
from collections import defaultdict
import random
import time

class QuantumShortestPath:
    """Quantum shortest path solver using QUBO formulation."""
    
    def __init__(self, graph, source, target):
        """
        Initialize the quantum shortest path solver.
        
        Args:
            graph: NetworkX graph with 'weight' attribute on edges
            source: Source node
            target: Target node
        """
        self.graph = graph.copy()
        self.source = source
        self.target = target
        self.nodes = list(graph.nodes())
        self.edges = list(graph.edges())
        
        # Map edges to indices
        self.edge_to_idx = {edge: i for i, edge in enumerate(self.edges)}
        self.idx_to_edge = {i: edge for i, edge in enumerate(self.edges)}
        
        # Store positions for consistent plotting
        self.pos = nx.spring_layout(self.graph, seed=42)
    
    def build_qubo(self, penalty_strength=10.0):
        """
        Build QUBO model for shortest path problem.
        
        H = H_cost + penalty * (H_flow + H_source_target)
        
        Args:
            penalty_strength: Penalty for flow conservation constraints
            
        Returns:
            dimod.BinaryQuadraticModel
        """
        bqm = dimod.BinaryQuadraticModel(vartype='BINARY')
        
        # 1. Minimize path cost (objective)
        print("Adding objective function...")
        for (u, v), idx in self.edge_to_idx.items():
            weight = self.graph[u][v].get('weight', 1.0)
            bqm.add_linear(idx, weight)
        
        # 2. Flow conservation constraints
        print("Adding flow conservation constraints...")
        for node in self.nodes:
            # Get outgoing and incoming edges
            outgoing = [idx for (u, v), idx in self.edge_to_idx.items() if u == node]
            incoming = [idx for (u, v), idx in self.edge_to_idx.items() if v == node]
            
            if node == self.source:
                # Source: outflow = 1, inflow = 0
                # Constraint: (sum(outgoing) - 1)^2 + (sum(incoming))^2
                
                # Outflow constraint: (sum(outgoing) - 1)^2
                if outgoing:
                    # Linear terms from squares: x^2 = x for binary variables
                    for edge_idx in outgoing:
                        current = bqm.get_linear(edge_idx)
                        bqm.set_linear(edge_idx, current + penalty_strength)
                    
                    # Quadratic terms from cross products
                    for i in range(len(outgoing)):
                        for j in range(i+1, len(outgoing)):
                            bqm.add_quadratic(outgoing[i], outgoing[j], 2*penalty_strength)
                    
                    # Terms from -2*sum(x)*1
                    for edge_idx in outgoing:
                        current = bqm.get_linear(edge_idx)
                        bqm.set_linear(edge_idx, current - 2*penalty_strength)
                    
                    # Constant term from 1^2
                    bqm.offset += penalty_strength
                
                # Inflow constraint: (sum(incoming))^2
                if incoming:
                    # Linear terms
                    for edge_idx in incoming:
                        current = bqm.get_linear(edge_idx)
                        bqm.set_linear(edge_idx, current + penalty_strength)
                    
                    # Quadratic terms
                    for i in range(len(incoming)):
                        for j in range(i+1, len(incoming)):
                            bqm.add_quadratic(incoming[i], incoming[j], 2*penalty_strength)
            
            elif node == self.target:
                # Target: inflow = 1, outflow = 0
                # Constraint: (sum(incoming) - 1)^2 + (sum(outgoing))^2
                
                # Inflow constraint: (sum(incoming) - 1)^2
                if incoming:
                    # Linear terms
                    for edge_idx in incoming:
                        current = bqm.get_linear(edge_idx)
                        bqm.set_linear(edge_idx, current + penalty_strength)
                    
                    # Quadratic terms
                    for i in range(len(incoming)):
                        for j in range(i+1, len(incoming)):
                            bqm.add_quadratic(incoming[i], incoming[j], 2*penalty_strength)
                    
                    # Terms from -2*sum(x)*1
                    for edge_idx in incoming:
                        current = bqm.get_linear(edge_idx)
                        bqm.set_linear(edge_idx, current - 2*penalty_strength)
                    
                    # Constant term
                    bqm.offset += penalty_strength
                
                # Outflow constraint: (sum(outgoing))^2
                if outgoing:
                    # Linear terms
                    for edge_idx in outgoing:
                        current = bqm.get_linear(edge_idx)
                        bqm.set_linear(edge_idx, current + penalty_strength)
                    
                    # Quadratic terms
                    for i in range(len(outgoing)):
                        for j in range(i+1, len(outgoing)):
                            bqm.add_quadratic(outgoing[i], outgoing[j], 2*penalty_strength)
            
            else:
                # Intermediate nodes: inflow = outflow = 0 or 1
                # Constraint: (sum(incoming) - sum(outgoing))^2
                
                if incoming or outgoing:
                    # Linear terms from squares of incoming
                    for edge_idx in incoming:
                        current = bqm.get_linear(edge_idx)
                        bqm.set_linear(edge_idx, current + penalty_strength)
                    
                    # Quadratic terms for incoming pairs
                    for i in range(len(incoming)):
                        for j in range(i+1, len(incoming)):
                            bqm.add_quadratic(incoming[i], incoming[j], 2*penalty_strength)
                    
                    # Linear terms from squares of outgoing
                    for edge_idx in outgoing:
                        current = bqm.get_linear(edge_idx)
                        bqm.set_linear(edge_idx, current + penalty_strength)
                    
                    # Quadratic terms for outgoing pairs
                    for i in range(len(outgoing)):
                        for j in range(i+1, len(outgoing)):
                            bqm.add_quadratic(outgoing[i], outgoing[j], 2*penalty_strength)
                    
                    # Cross terms between incoming and outgoing (with negative sign)
                    for in_idx in incoming:
                        for out_idx in outgoing:
                            bqm.add_quadratic(in_idx, out_idx, -2*penalty_strength)
        
        return bqm
    
    def solve_quantum(self, num_reads=1000, annealing_time=20):
        """
        Solve the shortest path using quantum annealing.
        
        Args:
            num_reads: Number of samples
            annealing_time: Annealing time per sample
            
        Returns:
            dict: Results including best path and statistics
        """
        print("Building QUBO model...")
        bqm = self.build_qubo(penalty_strength=15.0)
        
        print(f"QUBO size: {len(bqm.variables)} variables, {bqm.num_interactions} interactions")
        print(f"Solving with {num_reads} reads...")
        
        sampler = SimulatedAnnealingSampler()
        start_time = time.time()
        response = sampler.sample(bqm, num_reads=num_reads, annealing_time=annealing_time)
        solve_time = time.time() - start_time
        
        print(f"Solving completed in {solve_time:.2f} seconds")
        
        # Find best feasible solution
        best_solution = None
        best_energy = float('inf')
        feasible_count = 0
        
        for sample, energy in response.data(['sample', 'energy']):
            if self.is_valid_path(sample):
                feasible_count += 1
                if energy < best_energy:
                    best_energy = energy
                    best_solution = sample
        
        # Convert to path
        path = None
        if best_solution:
            path = self.solution_to_path(best_solution)
        
        return {
            'best_solution': best_solution,
            'best_energy': best_energy,
            'best_path': path,
            'feasible_count': feasible_count,
            'total_samples': num_reads,
            'solve_time': solve_time,
            'response': response
        }
    
    def is_valid_path(self, solution):
        """Check if solution represents a valid path."""
        selected_edges = []
        for idx, value in solution.items():
            if value == 1:
                selected_edges.append(self.idx_to_edge[idx])
        
        if not selected_edges:
            return False
        
        # Build subgraph
        subgraph = nx.Graph()
        subgraph.add_edges_from(selected_edges)
        
        # Check if it's a simple path from source to target
        if not nx.has_path(subgraph, self.source, self.target):
            return False
        
        # Check if it's exactly one path (no branches)
        path_nodes = set()
        for edge in selected_edges:
            path_nodes.update(edge)
        
        # Each intermediate node should have degree 2, endpoints degree 1
        for node in path_nodes:
            degree = subgraph.degree(node)
            if node == self.source or node == self.target:
                if degree != 1:
                    return False
            else:
                if degree != 2:
                    return False
        
        return True
    
    def solution_to_path(self, solution):
        """Convert binary solution to node path."""
        selected_edges = []
        for idx, value in solution.items():
            if value == 1:
                selected_edges.append(self.idx_to_edge[idx])
        
        if not selected_edges:
            return None
        
        # Build graph from selected edges
        G = nx.Graph()
        G.add_edges_from(selected_edges)
        
        try:
            path = nx.shortest_path(G, self.source, self.target)
            return path
        except:
            return None
    
    def calculate_path_cost(self, path):
        """Calculate total cost of a path."""
        if not path:
            return float('inf')
        
        total_cost = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_cost += self.graph[u][v].get('weight', 1.0)
        
        return total_cost
    
    def plot_graph(self, title="Graph", highlight_path=None, save_path=None):
        """Plot the graph with optional path highlighting."""
        plt.figure(figsize=(14, 10))
        
        # Draw all nodes
        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            if node == self.source:
                node_colors.append('green')
                node_sizes.append(1000)
            elif node == self.target:
                node_colors.append('red')
                node_sizes.append(1000)
            else:
                node_colors.append('lightblue')
                node_sizes.append(700)
        
        nx.draw_networkx_nodes(self.graph, self.pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9)
        
        # Draw all edges with weights
        all_edges = list(self.graph.edges())
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=all_edges,
                              edge_color='lightgray', width=2, alpha=0.5)
        
        # Highlight path if provided
        if highlight_path:
            path_edges = [(highlight_path[i], highlight_path[i+1]) 
                         for i in range(len(highlight_path)-1)]
            nx.draw_networkx_edges(self.graph, self.pos, edgelist=path_edges,
                                  edge_color='red', width=4, alpha=0.9)
            
            # Highlight path nodes
            nx.draw_networkx_nodes(self.graph, self.pos, nodelist=highlight_path,
                                  node_color='orange', node_size=800, alpha=0.9)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, self.pos, font_size=12, font_weight='bold')
        
        # Add edge weights
        edge_labels = {}
        for (u, v) in self.graph.edges():
            weight = self.graph[u][v].get('weight', 1)
            edge_labels[(u, v)] = f"{weight}"
        
        nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels,
                                    font_size=10, font_color='darkblue')
        
        plt.title(title, fontsize=18, fontweight='bold', pad=20)
        plt.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.9, label='Source'),
            Patch(facecolor='red', alpha=0.9, label='Target'),
            Patch(facecolor='orange', alpha=0.9, label='Path Node'),
            Patch(facecolor='white', edgecolor='red', linewidth=3, label='Selected Path'),
            Patch(facecolor='white', edgecolor='lightgray', linewidth=2, label='Other Edges')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def plot_comparison(self, quantum_path, classical_path, quantum_cost, classical_cost):
        """Plot comparison between quantum and classical solutions."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        
        # Plot quantum solution
        ax1 = axes[0]
        self._plot_solution(ax1, quantum_path, f"Quantum Solution\nPath: {quantum_path}\nCost: {quantum_cost}")
        
        # Plot classical solution
        ax2 = axes[1]
        self._plot_solution(ax2, classical_path, f"Classical Solution\nPath: {classical_path}\nCost: {classical_cost}")
        
        # Add overall title
        plt.suptitle(f"Shortest Path Comparison\nSource: {self.source}, Target: {self.target}", 
                    fontsize=20, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_solution(self, ax, path, title):
        """Helper method to plot a single solution."""
        # Draw all nodes
        node_colors = []
        for node in self.graph.nodes():
            if node == self.source:
                node_colors.append('green')
            elif node == self.target:
                node_colors.append('red')
            elif path and node in path:
                node_colors.append('orange')
            else:
                node_colors.append('lightblue')
        
        nx.draw_networkx_nodes(self.graph, self.pos, ax=ax, node_color=node_colors, 
                              node_size=600, alpha=0.9)
        
        # Draw all edges
        nx.draw_networkx_edges(self.graph, self.pos, ax=ax, edge_color='lightgray', 
                              width=2, alpha=0.4)
        
        # Highlight solution path
        if path:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(self.graph, self.pos, ax=ax, edgelist=path_edges,
                                  edge_color='red', width=4, alpha=0.9)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, self.pos, ax=ax, font_size=10)
        
        # Add edge weights
        edge_labels = {}
        for (u, v) in self.graph.edges():
            weight = self.graph[u][v].get('weight', 1)
            edge_labels[(u, v)] = f"{weight}"
        
        nx.draw_networkx_edge_labels(self.graph, self.pos, ax=ax, edge_labels=edge_labels,
                                    font_size=9)
        
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        ax.axis('off')


def create_large_graph(num_nodes=15, edge_probability=0.3, seed=42):
    """
    Create a larger random graph for testing.
    
    Args:
        num_nodes: Number of nodes in the graph
        edge_probability: Probability of creating an edge between nodes
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX graph
    """
    print(f"Creating random graph with {num_nodes} nodes...")
    random.seed(seed)
    np.random.seed(seed)
    
    # Create a connected random graph
    while True:
        G = nx.erdos_renyi_graph(num_nodes, edge_probability, seed=seed)
        if nx.is_connected(G):
            break
        seed += 1
    
    # Convert to directed graph (for easier path finding)
    G = G.to_directed()
    
    # Add reverse edges to ensure connectivity
    edges_to_add = []
    for u, v in list(G.edges()):
        if not G.has_edge(v, u):
            edges_to_add.append((v, u))
    
    for u, v in edges_to_add:
        G.add_edge(u, v)
    
    # Assign random weights (1-20 for variety)
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(1, 20)
    
    return G


def create_grid_graph(rows=5, cols=5):
    """
    Create a grid graph for more structured testing.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        
    Returns:
        NetworkX graph
    """
    print(f"Creating grid graph {rows}x{cols}...")
    G = nx.grid_2d_graph(rows, cols)
    
    # Convert to directed and add reverse edges
    G = G.to_directed()
    edges_to_add = []
    for u, v in list(G.edges()):
        if not G.has_edge(v, u):
            edges_to_add.append((v, u))
    
    for u, v in edges_to_add:
        G.add_edge(u, v)
    
    # Assign random weights (1-10)
    random.seed(42)
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(1, 10)
    
    # Convert to integer node labels for easier handling
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    return G


def create_known_graph():
    """Create a known graph where we can predict the shortest path."""
    print("Creating known test graph...")
    G = nx.DiGraph()
    
    # Add 10 nodes
    for i in range(10):
        G.add_node(i)
    
    # Add edges with specific weights
    edges = [
        (0, 1, 2), (0, 2, 4), (0, 3, 3),
        (1, 4, 5), (1, 5, 2),
        (2, 4, 1), (2, 5, 6),
        (3, 5, 4), (3, 6, 3),
        (4, 7, 3), (4, 8, 2),
        (5, 7, 1), (5, 8, 5),
        (6, 8, 4), (6, 9, 2),
        (7, 9, 3),
        (8, 9, 1)
    ]
    
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
        # Add reverse edges with higher weights to avoid cycles
        G.add_edge(v, u, weight=w+5)
    
    return G


def find_classical_shortest_path(graph, source, target):
    """Find shortest path using classical algorithms."""
    try:
        path = nx.shortest_path(graph, source, target, weight='weight')
        cost = sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
        return path, cost
    except:
        return None, float('inf')


def analyze_solutions(solver, result):
    """Analyze and print solution statistics."""
    if result['best_path']:
        quantum_path = result['best_path']
        quantum_cost = solver.calculate_path_cost(quantum_path)
        
        print(f"\nQuantum Results:")
        print(f"  Path found: {quantum_path}")
        print(f"  Path cost: {quantum_cost}")
        print(f"  Energy: {result['best_energy']:.2f}")
        print(f"  Feasible solutions: {result['feasible_count']}/{result['total_samples']} ({result['feasible_count']/result['total_samples']*100:.1f}%)")
        
        # Show a few samples
        print(f"\nSample analysis:")
        samples_shown = 0
        for sample, energy in result['response'].data(['sample', 'energy']):
            if samples_shown < 3:
                selected = [solver.idx_to_edge[idx] for idx, val in sample.items() if val == 1]
                print(f"  Sample energy {energy:.2f}: {len(selected)} edges selected")
                samples_shown += 1
        
        return quantum_path, quantum_cost
    else:
        print("No feasible quantum solution found!")
        return None, float('inf')


def main():
    """Main execution function."""
    print("=" * 70)
    print("QUANTUM SHORTEST PATH SOLVER")
    print("=" * 70)
    
    # Choose graph type
    print("\nChoose graph type:")
    print("1. Random graph (12 nodes)")
    print("2. Grid graph (4x4 = 16 nodes)")
    print("3. Known test graph (10 nodes)")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        # Create random graph
        G = create_large_graph(num_nodes=12, edge_probability=0.3)
        source = 0
        target = 11
    elif choice == "2":
        # Create grid graph
        G = create_grid_graph(rows=4, cols=4)
        source = 0  # Top-left corner
        target = 15  # Bottom-right corner
    else:
        # Create known test graph
        G = create_known_graph()
        source = 0
        target = 9
    
    print(f"\nGraph created:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Source: {source}")
    print(f"  Target: {target}")
    
    # Create solver
    solver = QuantumShortestPath(G, source, target)
    
    # Plot the original graph
    print("\nPlotting original graph...")
    solver.plot_graph(title=f"Original Graph\nNodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Find classical solution
    print("\nFinding classical shortest path...")
    classical_path, classical_cost = find_classical_shortest_path(G, source, target)
    
    if classical_path:
        print(f"  Classical shortest path: {classical_path}")
        print(f"  Classical path cost: {classical_cost}")
    else:
        print("  No classical path found!")
        return
    
    # Solve with quantum annealing
    print("\nSolving with quantum annealing...")
    print("This may take a moment...")
    
    # Adjust parameters based on graph size
    if G.number_of_nodes() > 12:
        num_reads = 800
        annealing_time = 25
    else:
        num_reads = 1000
        annealing_time = 20
    
    result = solver.solve_quantum(num_reads=num_reads, annealing_time=annealing_time)
    
    # Analyze quantum results
    quantum_path, quantum_cost = analyze_solutions(solver, result)
    
    if quantum_path:
        # Compare with classical
        print(f"\nComparison:")
        print(f"  Classical cost: {classical_cost}")
        print(f"  Quantum cost: {quantum_cost}")
        
        if abs(quantum_cost - classical_cost) < 0.01:
            print(f"  ✓ Quantum found optimal solution!")
        else:
            print(f"  ✗ Quantum solution is not optimal")
            print(f"    Difference: {quantum_cost - classical_cost:.2f}")
        
        # Plot quantum solution
        print("\nPlotting quantum solution...")
        solver.plot_graph(title=f"Quantum Solution\nPath: {quantum_path}\nCost: {quantum_cost}",
                         highlight_path=quantum_path)
        
        # Plot classical solution
        print("\nPlotting classical solution...")
        solver.plot_graph(title=f"Classical Optimal Solution\nPath: {classical_path}\nCost: {classical_cost}",
                         highlight_path=classical_path)
        
        # Plot comparison
        print("\nPlotting comparison...")
        solver.plot_comparison(quantum_path, classical_path, quantum_cost, classical_cost)
        
        # Additional statistics
        print("\nAdditional Statistics:")
        print(f"  Solve time: {result['solve_time']:.2f} seconds")
        
        # Find all unique feasible paths in quantum results
        feasible_paths = {}
        for sample, energy in result['response'].data(['sample', 'energy']):
            if solver.is_valid_path(sample):
                path = solver.solution_to_path(sample)
                if path:
                    path_tuple = tuple(path)
                    if path_tuple not in feasible_paths:
                        feasible_paths[path_tuple] = {
                            'energy': energy,
                            'cost': solver.calculate_path_cost(path)
                        }
        
        print(f"  Unique feasible paths found: {len(feasible_paths)}")
        
        if len(feasible_paths) > 1:
            print("  Top feasible paths by energy:")
            sorted_paths = sorted(feasible_paths.items(), key=lambda x: x[1]['energy'])
            for i, (path_tuple, info) in enumerate(sorted_paths[:3]):  # Top 3
                path = list(path_tuple)
                if path != quantum_path:
                    print(f"    Path {i+1}: {path} (cost: {info['cost']}, energy: {info['energy']:.2f})")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


def test_small_graph():
    """Test with a very small graph to verify algorithm works."""
    print("Testing with very small graph...")
    
    # Create a simple triangle graph
    G = nx.DiGraph()
    
    # Add 4 nodes in a diamond shape
    edges = [
        (0, 1, {'weight': 3}),
        (0, 2, {'weight': 2}),
        (1, 3, {'weight': 4}),
        (2, 3, {'weight': 1}),
    ]
    
    G.add_edges_from(edges)
    
    # Add reverse edges with higher weights
    reverse_edges = [(1, 0, 5), (2, 0, 5), (3, 1, 6), (3, 2, 6)]
    for u, v, w in reverse_edges:
        G.add_edge(u, v, weight=w)
    
    source = 0
    target = 3
    
    print(f"Simple graph: 4 nodes, 8 edges")
    print(f"Source: {source}, Target: {target}")
    
    solver = QuantumShortestPath(G, source, target)
    
    # Plot
    solver.plot_graph(title="Simple Test Graph")
    
    # Find classical solution
    classical_path, classical_cost = find_classical_shortest_path(G, source, target)
    print(f"Classical shortest path: {classical_path} (cost: {classical_cost})")
    
    # Solve quantum
    print("\nSolving with quantum...")
    result = solver.solve_quantum(num_reads=500, annealing_time=10)
    
    quantum_path, quantum_cost = analyze_solutions(solver, result)
    
    if quantum_path:
        print(f"\nComparison:")
        print(f"  Classical: {classical_path} (cost: {classical_cost})")
        print(f"  Quantum: {quantum_path} (cost: {quantum_cost})")
        
        if quantum_path == classical_path:
            print("✓ Quantum found optimal solution!")
        else:
            print("✗ Quantum did not find optimal solution")
        
        # Plot results
        solver.plot_comparison(quantum_path, classical_path, quantum_cost, classical_cost)


if __name__ == "__main__":
    print("Quantum Shortest Path Solver")
    print("=" * 50)
    print("This program solves the shortest path problem using quantum annealing.")
    print("It compares the quantum solution with classical Dijkstra's algorithm.")
    print("=" * 50)
    
    print("\nOptions:")
    print("1. Run main program with choice of graphs")
    print("2. Run test with very small graph")
    
    main()