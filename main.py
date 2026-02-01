import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from dwave.system import LeapHybridSampler
from dwave.samplers import SimulatedAnnealingSampler
import dimod
from collections import defaultdict
import itertools

class ConstrainedShortestPathQuantum:
    """
    Solves the constrained shortest path problem using quantum annealing.
    Uses penalty methods for constraints and incorporates Grover-like mixing.
    """
    
    def __init__(self, graph, source, target, resource_constraints=None):
        """
        Initialize the constrained shortest path solver.
        
        Args:
            graph: NetworkX graph with 'weight' and optional resource attributes on edges
            source: Source node
            target: Target node
            resource_constraints: Dict of {resource_name: max_limit}
        """
        self.graph = graph.copy()  # Work with a copy
        self.source = source
        self.target = target
        self.nodes = list(graph.nodes())
        self.edges = list(graph.edges())
        self.resource_constraints = resource_constraints or {}
        
        # Map edges to indices
        self.edge_to_idx = {edge: i for i, edge in enumerate(self.edges)}
        self.idx_to_edge = {i: edge for i, edge in enumerate(self.edges)}
        
    def build_qubo_model(self, penalty_strength=10.0, resource_penalty=5.0):
        """
        Build QUBO model for constrained shortest path.
        
        Hamiltonian: H = H_cost + penalty * (H_flow + H_source_target) + resource_penalty * H_resources
        
        Args:
            penalty_strength: Penalty for flow conservation constraints
            resource_penalty: Penalty for resource constraints
            
        Returns:
            dimod.BinaryQuadraticModel
        """
        n_edges = len(self.edges)
        bqm = dimod.BinaryQuadraticModel(vartype='BINARY')
        
        # 1. Minimize path cost (main objective)
        for (u, v), idx in self.edge_to_idx.items():
            weight = self.graph[u][v].get('weight', 1.0)
            bqm.add_variable(idx, weight)
        
        # 2. Flow conservation constraints
        for node in self.nodes:
            if node == self.source:
                # Source: outflow - inflow = 1
                outflow = [self.edge_to_idx[(node, v)] 
                          for v in self.graph.neighbors(node) if (node, v) in self.edge_to_idx]
                inflow = [self.edge_to_idx[(u, node)] 
                         for u in self.graph.predecessors(node) if (u, node) in self.edge_to_idx]
                
                # Create constraint: (sum(outflow) - sum(inflow) - 1)^2
                linear_coeffs = {}
                quadratic_coeffs = {}
                
                # Linear terms
                for edge_idx in outflow:
                    linear_coeffs[edge_idx] = linear_coeffs.get(edge_idx, 0) + 1
                for edge_idx in inflow:
                    linear_coeffs[edge_idx] = linear_coeffs.get(edge_idx, 0) - 1
                
                # Quadratic terms
                all_edges = outflow + inflow
                for i in range(len(all_edges)):
                    for j in range(i+1, len(all_edges)):
                        edge_i = all_edges[i]
                        edge_j = all_edges[j]
                        quadratic_coeffs[(edge_i, edge_j)] = quadratic_coeffs.get((edge_i, edge_j), 0) + 2
                
                # Add to BQM with penalty
                for var, coeff in linear_coeffs.items():
                    current_linear = bqm.get_linear(var)
                    bqm.set_linear(var, current_linear + penalty_strength * 2 * coeff * (-1))
                
                for (i, j), coeff in quadratic_coeffs.items():
                    current_quadratic = bqm.get_quadratic(i, j, default=0)
                    bqm.set_quadratic(i, j, current_quadratic + penalty_strength * coeff)
                
                # Constant term
                bqm.offset += penalty_strength * 1  # (-1)^2 = 1
                
            elif node == self.target:
                # Target: inflow - outflow = 1
                outflow = [self.edge_to_idx[(node, v)] 
                          for v in self.graph.neighbors(node) if (node, v) in self.edge_to_idx]
                inflow = [self.edge_to_idx[(u, node)] 
                         for u in self.graph.predecessors(node) if (u, node) in self.edge_to_idx]
                
                linear_coeffs = {}
                quadratic_coeffs = {}
                
                for edge_idx in inflow:
                    linear_coeffs[edge_idx] = linear_coeffs.get(edge_idx, 0) + 1
                for edge_idx in outflow:
                    linear_coeffs[edge_idx] = linear_coeffs.get(edge_idx, 0) - 1
                
                all_edges = outflow + inflow
                for i in range(len(all_edges)):
                    for j in range(i+1, len(all_edges)):
                        edge_i = all_edges[i]
                        edge_j = all_edges[j]
                        quadratic_coeffs[(edge_i, edge_j)] = quadratic_coeffs.get((edge_i, edge_j), 0) + 2
                
                for var, coeff in linear_coeffs.items():
                    current_linear = bqm.get_linear(var)
                    bqm.set_linear(var, current_linear + penalty_strength * 2 * coeff * (-1))
                
                for (i, j), coeff in quadratic_coeffs.items():
                    current_quadratic = bqm.get_quadratic(i, j, default=0)
                    bqm.set_quadratic(i, j, current_quadratic + penalty_strength * coeff)
                
                bqm.offset += penalty_strength * 1
                
            else:
                # Intermediate nodes: outflow = inflow
                outflow = [self.edge_to_idx[(node, v)] 
                          for v in self.graph.neighbors(node) if (node, v) in self.edge_to_idx]
                inflow = [self.edge_to_idx[(u, node)] 
                         for u in self.graph.predecessors(node) if (u, node) in self.edge_to_idx]
                
                linear_coeffs = {}
                quadratic_coeffs = {}
                
                for edge_idx in outflow:
                    linear_coeffs[edge_idx] = linear_coeffs.get(edge_idx, 0) + 1
                for edge_idx in inflow:
                    linear_coeffs[edge_idx] = linear_coeffs.get(edge_idx, 0) - 1
                
                all_edges = outflow + inflow
                for i in range(len(all_edges)):
                    for j in range(i+1, len(all_edges)):
                        edge_i = all_edges[i]
                        edge_j = all_edges[j]
                        quadratic_coeffs[(edge_i, edge_j)] = quadratic_coeffs.get((edge_i, edge_j), 0) + 2
                
                for var, coeff in linear_coeffs.items():
                    current_linear = bqm.get_linear(var)
                    bqm.set_linear(var, current_linear + penalty_strength * 2 * coeff * 0)
                
                for (i, j), coeff in quadratic_coeffs.items():
                    current_quadratic = bqm.get_quadratic(i, j, default=0)
                    bqm.set_quadratic(i, j, current_quadratic + penalty_strength * coeff)
        
        # 3. Resource constraints
        if self.resource_constraints:
            for resource, max_limit in self.resource_constraints.items():
                # Add penalty for exceeding resource limit
                for (u, v), idx in self.edge_to_idx.items():
                    resource_val = self.graph[u][v].get(resource, 0)
                    if resource_val > 0:
                        # Linear penalty proportional to resource usage
                        current_linear = bqm.get_linear(idx)
                        penalty = resource_penalty * (resource_val / max_limit) ** 2
                        bqm.set_linear(idx, current_linear + penalty)
        
        return bqm
    
    def grover_mixer_feasible_space(self, bqm, num_iterations=3):
        """
        Apply Grover-like mixing to enhance probability of feasible solutions.
        This is a classical pre-processing step that modifies the QUBO to favor feasible solutions.
        
        Args:
            bqm: Original BinaryQuadraticModel
            num_iterations: Number of mixing iterations
            
        Returns:
            Modified BinaryQuadraticModel
        """
        print("Applying Grover-like mixer to enhance feasible solution space...")
        
        # Create a copy of the BQM
        mixed_bqm = bqm.copy()
        
        # Sample some solutions to identify feasible regions
        sampler = SimulatedAnnealingSampler()
        
        for iteration in range(num_iterations):
            # Sample from current QUBO
            response = sampler.sample(mixed_bqm, num_reads=50)
            
            # Analyze samples for feasibility
            feasible_solutions = []
            for sample, energy in response.data(['sample', 'energy']):
                if self.is_feasible(sample):
                    feasible_solutions.append(sample)
            
            if feasible_solutions:
                # Calculate average of feasible solutions
                avg_feasible = self.average_solutions(feasible_solutions)
                
                # Modify QUBO to favor solutions similar to feasible ones
                for var in mixed_bqm.variables:
                    if avg_feasible.get(var, 0) > 0.7:  # Strongly favored in feasible solutions
                        # Reduce energy for this variable being 1
                        current_linear = mixed_bqm.get_linear(var)
                        mixed_bqm.set_linear(var, current_linear - 0.3)
                    elif avg_feasible.get(var, 0) < 0.3:  # Rarely in feasible solutions
                        # Increase energy for this variable being 1
                        current_linear = mixed_bqm.get_linear(var)
                        mixed_bqm.set_linear(var, current_linear + 0.3)
            
            print(f"Iteration {iteration + 1}: Found {len(feasible_solutions)} feasible solutions")
        
        return mixed_bqm
    
    def is_feasible(self, solution):
        """Check if a solution represents a valid path satisfying constraints."""
        # Convert solution to edge selection
        selected_edges = []
        for idx, value in solution.items():
            if value == 1:
                selected_edges.append(self.idx_to_edge[idx])
        
        # Check if it forms a path from source to target
        if not selected_edges:
            return False
        
        # Build subgraph from selected edges
        subgraph = nx.DiGraph()
        subgraph.add_edges_from(selected_edges)
        
        # Check connectivity
        if not nx.has_path(subgraph, self.source, self.target):
            return False
        
        # Check flow conservation
        for node in subgraph.nodes():
            if node == self.source:
                if subgraph.out_degree(node) != 1 or subgraph.in_degree(node) != 0:
                    return False
            elif node == self.target:
                if subgraph.in_degree(node) != 1 or subgraph.out_degree(node) != 0:
                    return False
            else:
                if not (subgraph.in_degree(node) == 1 and subgraph.out_degree(node) == 1):
                    return False
        
        # Check resource constraints
        total_resources = {resource: 0 for resource in self.resource_constraints.keys()}
        for u, v in selected_edges:
            for resource in self.resource_constraints.keys():
                total_resources[resource] += self.graph[u][v].get(resource, 0)
        
        for resource, limit in self.resource_constraints.items():
            if total_resources[resource] > limit:
                return False
        
        return True
    
    def average_solutions(self, solutions):
        """Calculate average values for each variable across solutions."""
        if not solutions:
            return {}
        
        avg = defaultdict(float)
        for sol in solutions:
            for var, val in sol.items():
                avg[var] += val
        for var in avg:
            avg[var] /= len(solutions)
        return dict(avg)
    
    def solve_with_dwave_simulator(self, num_reads=1000, annealing_time=10):
        """
        Solve using D-Wave's simulated annealing sampler.
        
        Args:
            num_reads: Number of reads/samples
            annealing_time: Annealing time per read
            
        Returns:
            dict: Best solution and statistics
        """
        print("Building QUBO model...")
        bqm = self.build_qubo_model()
        
        print("Applying Grover-like mixer...")
        bqm = self.grover_mixer_feasible_space(bqm)
        
        print("Solving with simulated annealing sampler...")
        sampler = SimulatedAnnealingSampler()
        
        response = sampler.sample(bqm, num_reads=num_reads, annealing_time=annealing_time)
        
        # Find best feasible solution
        best_solution = None
        best_energy = float('inf')
        feasible_count = 0
        
        for sample, energy in response.data(['sample', 'energy']):
            if self.is_feasible(sample):
                feasible_count += 1
                if energy < best_energy:
                    best_energy = energy
                    best_solution = sample
        
        # Convert solution to path
        path = None
        if best_solution:
            path = self.solution_to_path(best_solution)
        
        return {
            'best_solution': best_solution,
            'best_energy': best_energy,
            'best_path': path,
            'feasible_count': feasible_count,
            'total_samples': num_reads,
            'response': response
        }
    
    def solve_with_hybrid_sampler(self, time_limit=10):
        """
        Solve using D-Wave's hybrid sampler (requires Leap access).
        
        Args:
            time_limit: Maximum run time in seconds
            
        Returns:
            dict: Best solution and statistics
        """
        print("Building QUBO model...")
        bqm = self.build_qubo_model()
        
        print("Applying Grover-like mixer...")
        bqm = self.grover_mixer_feasible_space(bqm)
        
        print("Solving with hybrid sampler...")
        try:
            sampler = LeapHybridSampler()
            response = sampler.sample(bqm, time_limit=time_limit)
            
            # Find best feasible solution
            best_solution = None
            best_energy = float('inf')
            
            for sample, energy in response.data(['sample', 'energy']):
                if self.is_feasible(sample) and energy < best_energy:
                    best_energy = energy
                    best_solution = sample
            
            path = None
            if best_solution:
                path = self.solution_to_path(best_solution)
            
            return {
                'best_solution': best_solution,
                'best_energy': best_energy,
                'best_path': path,
                'response': response
            }
            
        except Exception as e:
            print(f"Hybrid sampler error: {e}")
            print("Falling back to simulated annealing...")
            return self.solve_with_dwave_simulator()
    
    def solution_to_path(self, solution):
        """Convert binary solution to actual path."""
        selected_edges = []
        for idx, value in solution.items():
            if value == 1:
                selected_edges.append(self.idx_to_edge[idx])
        
        if not selected_edges:
            return None
        
        # Reconstruct path from selected edges
        G = nx.DiGraph()
        G.add_edges_from(selected_edges)
        
        try:
            path = nx.shortest_path(G, self.source, self.target)
            return path
        except:
            return None
    
    def calculate_path_metrics(self, path):
        """Calculate total cost and resource usage for a path."""
        if not path:
            return None
        
        total_cost = 0
        total_resources = {resource: 0 for resource in self.resource_constraints.keys()}
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_cost += self.graph[u][v].get('weight', 0)
            for resource in self.resource_constraints.keys():
                total_resources[resource] += self.graph[u][v].get(resource, 0)
        
        return {
            'cost': total_cost,
            'resources': total_resources,
            'path': path
        }
    
    def plot_graph(self, title="Original Graph", highlight_path=None, save_path=None):
        """
        Plot the graph with optional path highlighting.
        
        Args:
            title: Plot title
            highlight_path: List of nodes representing path to highlight
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            if node == self.source:
                node_colors.append('green')
                node_sizes.append(800)
            elif node == self.target:
                node_colors.append('red')
                node_sizes.append(800)
            else:
                node_colors.append('lightblue')
                node_sizes.append(600)
        
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8)
        
        # Draw edges
        edge_labels = {}
        for (u, v) in self.graph.edges():
            weight = self.graph[u][v].get('weight', 1)
            time = self.graph[u][v].get('time', 0)
            edge_labels[(u, v)] = f"w:{weight}\nt:{time}"
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(self.graph, pos, edge_color='lightgray', 
                              width=2, alpha=0.5, arrows=True)
        
        # Highlight path if provided
        if highlight_path:
            path_edges = [(highlight_path[i], highlight_path[i+1]) 
                         for i in range(len(highlight_path)-1)]
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges,
                                  edge_color='red', width=4, alpha=0.8,
                                  arrows=True, arrowstyle='-|>', arrowsize=20)
            
            # Highlight path nodes
            path_nodes = highlight_path
            nx.draw_networkx_nodes(self.graph, pos, nodelist=path_nodes,
                                  node_color='orange', node_size=800, alpha=0.8)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_weight='bold')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels,
                                    font_size=9, font_color='darkblue')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.8, label='Source'),
            Patch(facecolor='red', alpha=0.8, label='Target'),
            Patch(facecolor='lightblue', alpha=0.8, label='Intermediate'),
            Patch(facecolor='orange', alpha=0.8, label='Path Node'),
            Patch(facecolor='white', edgecolor='red', linewidth=2, label='Selected Path'),
            Patch(facecolor='white', edgecolor='lightgray', linewidth=2, label='Other Edges')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def plot_solution_comparison(self, quantum_path, classical_path=None, save_path=None):
        """
        Plot comparison between quantum and classical solutions.
        
        Args:
            quantum_path: Path found by quantum solver
            classical_path: Path found by classical solver (optional)
            save_path: Path to save the figure (optional)
        """
        fig, axes = plt.subplots(1, 2 if classical_path else 1, figsize=(16, 8))
        
        if not classical_path:
            axes = [axes]
        
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Plot quantum solution
        ax = axes[0]
        self._plot_single_solution(ax, pos, quantum_path, "Quantum Solution")
        
        # Plot classical solution if provided
        if classical_path:
            ax = axes[1]
            self._plot_single_solution(ax, pos, classical_path, "Classical Solution")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison figure saved to {save_path}")
        
        plt.show()
    
    def _plot_single_solution(self, ax, pos, path, title):
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
        
        nx.draw_networkx_nodes(self.graph, pos, ax=ax, node_color=node_colors, 
                              node_size=600, alpha=0.8)
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(self.graph, pos, ax=ax, edge_color='lightgray', 
                              width=2, alpha=0.3, arrows=True)
        
        # Highlight solution path
        if path:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(self.graph, pos, ax=ax, edgelist=path_edges,
                                  edge_color='red', width=4, alpha=0.8,
                                  arrows=True, arrowstyle='-|>', arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, ax=ax, font_size=10)
        
        # Add edge labels with weights and times
        edge_labels = {}
        for (u, v) in self.graph.edges():
            weight = self.graph[u][v].get('weight', 1)
            time = self.graph[u][v].get('time', 0)
            edge_labels[(u, v)] = f"w:{weight}\nt:{time}"
        
        nx.draw_networkx_edge_labels(self.graph, pos, ax=ax, edge_labels=edge_labels,
                                    font_size=8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add metrics if path exists
        if path:
            metrics = self.calculate_path_metrics(path)
            textstr = f"Path: {path}\nCost: {metrics['cost']}"
            for resource, usage in metrics['resources'].items():
                limit = self.resource_constraints.get(resource, float('inf'))
                textstr += f"\n{resource}: {usage}/{limit}"
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props)


def create_example_graph():
    """Create an example graph for testing."""
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(6):
        G.add_node(i)
    
    # Add edges with weights and resource constraints
    edges = [
        (0, 1, {'weight': 2, 'time': 3}),
        (0, 2, {'weight': 4, 'time': 2}),
        (1, 2, {'weight': 1, 'time': 1}),
        (1, 3, {'weight': 5, 'time': 4}),
        (2, 3, {'weight': 3, 'time': 2}),
        (2, 4, {'weight': 2, 'time': 3}),
        (3, 4, {'weight': 1, 'time': 2}),
        (3, 5, {'weight': 3, 'time': 3}),
        (4, 5, {'weight': 2, 'time': 1})
    ]
    
    G.add_edges_from(edges)
    return G


def main():
    """Main example usage."""
    print("=" * 60)
    print("Constrained Shortest Path using Quantum Annealing")
    print("=" * 60)
    
    # Create example graph
    print("\n1. Creating example graph...")
    G = create_example_graph()
    
    print(f"Graph nodes: {list(G.nodes())}")
    print(f"Graph edges: {list(G.edges())}")
    
    # Define problem
    source = 0
    target = 5
    resource_constraints = {'time': 6}  # Maximum total time
    
    print(f"\n2. Problem setup:")
    print(f"   Source: {source}")
    print(f"   Target: {target}")
    print(f"   Resource constraints: {resource_constraints}")
    
    # Create solver
    solver = ConstrainedShortestPathQuantum(
        G, source, target, resource_constraints
    )
    
    # Plot original graph
    print("\n3. Plotting original graph...")
    solver.plot_graph(title="Original Graph with Edge Weights (w) and Times (t)")
    
    # Solve using simulated annealing (works without Leap access)
    print("\n4. Solving with simulated annealing sampler...")
    result_sa = solver.solve_with_dwave_simulator(num_reads=200, annealing_time=5)
    
    print(f"\nSimulated Annealing Results:")
    print(f"   Feasible solutions found: {result_sa['feasible_count']}/{result_sa['total_samples']}")
    print(f"   Best energy: {result_sa['best_energy']:.2f}")
    
    quantum_path = None
    if result_sa['best_path']:
        metrics = solver.calculate_path_metrics(result_sa['best_path'])
        quantum_path = result_sa['best_path']
        print(f"\n   Best path: {metrics['path']}")
        print(f"   Path cost: {metrics['cost']}")
        print(f"   Resource usage: {metrics['resources']}")
        
        # Check constraints
        for resource, limit in resource_constraints.items():
            usage = metrics['resources'][resource]
            status = "✓" if usage <= limit else "✗"
            print(f"   {resource} constraint: {usage}/{limit} {status}")
        
        # Plot quantum solution
        print("\n5. Plotting quantum solution...")
        solver.plot_graph(title=f"Quantum Solution\nPath: {quantum_path}\nCost: {metrics['cost']}, Time: {metrics['resources']['time']}",
                         highlight_path=quantum_path)
    else:
        print("   No feasible path found!")
    
    # Compare with classical solution
    print("\n6. Finding classical solution for comparison...")
    classical_path = None
    try:
        # Find all simple paths
        all_paths = []
        for path in nx.all_simple_paths(G, source, target):
            cost = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            time = sum(G[path[i]][path[i+1]].get('time', 0) for i in range(len(path)-1))
            all_paths.append((path, cost, time))
        
        # Filter feasible paths
        feasible_paths = [p for p in all_paths if p[2] <= resource_constraints['time']]
        
        if feasible_paths:
            # Sort by cost
            feasible_paths.sort(key=lambda x: x[1])
            best_classical = feasible_paths[0]
            classical_path = best_classical[0]
            
            print(f"\n   Classical best path: {best_classical[0]}")
            print(f"   Path cost: {best_classical[1]}")
            print(f"   Time usage: {best_classical[2]}/{resource_constraints['time']}")
            
            # Plot classical solution
            print("\n7. Plotting classical solution...")
            solver.plot_graph(title=f"Classical Optimal Solution\nPath: {classical_path}\nCost: {best_classical[1]}, Time: {best_classical[2]}",
                             highlight_path=classical_path)
            
            # Check if quantum found optimal
            if quantum_path:
                quantum_cost = solver.calculate_path_metrics(quantum_path)['cost']
                is_optimal = abs(quantum_cost - best_classical[1]) < 0.01
                print(f"\n   Quantum solution optimal: {'✓' if is_optimal else '✗'}")
                
                if not is_optimal:
                    print(f"   Quantum cost: {quantum_cost}, Classical cost: {best_classical[1]}")
                    print(f"   Difference: {abs(quantum_cost - best_classical[1]):.2f}")
                
                # Plot comparison
                print("\n8. Plotting solution comparison...")
                solver.plot_solution_comparison(quantum_path, classical_path)
        else:
            print("   No feasible classical paths found!")
            
    except Exception as e:
        print(f"   Error in classical solution: {e}")
    
    # Print all feasible classical paths for reference
    print("\n9. All feasible classical paths:")
    try:
        all_paths = []
        for path in nx.all_simple_paths(G, source, target):
            cost = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            time = sum(G[path[i]][path[i+1]].get('time', 0) for i in range(len(path)-1))
            all_paths.append((path, cost, time))
        
        feasible_paths = [p for p in all_paths if p[2] <= resource_constraints['time']]
        
        if feasible_paths:
            feasible_paths.sort(key=lambda x: x[1])
            for i, (path, cost, time) in enumerate(feasible_paths):
                print(f"   {i+1}. Path: {path}, Cost: {cost}, Time: {time}")
        else:
            print("   No feasible paths found!")
    except Exception as e:
        print(f"   Error listing paths: {e}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()