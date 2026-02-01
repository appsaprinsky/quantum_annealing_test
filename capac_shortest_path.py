import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import LeapHybridSampler
import dimod
from collections import defaultdict
import random
import time
import itertools
from typing import Dict, List, Tuple, Optional

USE_GROVER_MIXER = True  # Set to False to disable Grover mixer

class CapacitatedShortestPathSolver:
    """
    Solves the capacitated shortest path problem using quantum annealing.
    Finds the shortest path from source to target while respecting resource constraints.
    """
    
    def __init__(self, graph: nx.DiGraph, source: int, target: int, 
                 resource_constraints: Dict[str, float]):
        """
        Initialize the capacitated shortest path solver.
        
        Args:
            graph: Directed graph with 'weight' and resource attributes on edges
            source: Source node
            target: Target node
            resource_constraints: Dictionary of {resource_name: max_limit}
        """
        self.graph = graph.copy()
        self.source = source
        self.target = target
        self.resource_constraints = resource_constraints
        self.nodes = list(graph.nodes())
        self.edges = list(graph.edges())
        
        # Map edges to indices for QUBO
        self.edge_to_idx = {edge: i for i, edge in enumerate(self.edges)}
        self.idx_to_edge = {i: edge for i, edge in enumerate(self.edges)}
        
        # Store for Grover mixer
        self.feasible_solutions_history = []
        
        # Layout for consistent plotting
        self.pos = nx.spring_layout(self.graph, seed=42)
    
    def build_qubo(self, penalty_strength: float = 15.0, 
                   resource_penalty: float = 10.0) -> dimod.BinaryQuadraticModel:
        """
        Build QUBO model for capacitated shortest path.
        
        H = H_cost + penalty * H_flow + resource_penalty * H_resources
        
        Args:
            penalty_strength: Penalty for flow conservation constraints
            resource_penalty: Penalty for resource constraint violations
            
        Returns:
            BinaryQuadraticModel
        """
        print("Building QUBO model...")
        bqm = dimod.BinaryQuadraticModel(vartype='BINARY')
        
        # 1. Objective: Minimize path cost
        print("  Adding objective function...")
        for (u, v), idx in self.edge_to_idx.items():
            weight = self.graph[u][v].get('weight', 1.0)
            bqm.add_linear(idx, weight)
        
        # 2. Flow conservation constraints
        print("  Adding flow conservation constraints...")
        for node in self.nodes:
            outgoing = [idx for (u, v), idx in self.edge_to_idx.items() if u == node]
            incoming = [idx for (u, v), idx in self.edge_to_idx.items() if v == node]
            
            if node == self.source:
                # Source: outflow = 1, inflow = 0
                self._add_source_constraints(bqm, outgoing, incoming, penalty_strength)
                
            elif node == self.target:
                # Target: inflow = 1, outflow = 0
                self._add_target_constraints(bqm, outgoing, incoming, penalty_strength)
                
            else:
                # Intermediate: inflow = outflow (both 0 or both 1)
                self._add_intermediate_constraints(bqm, outgoing, incoming, penalty_strength)
        
        # 3. Resource constraints
        print("  Adding resource constraints...")
        self._add_resource_constraints(bqm, resource_penalty)
        
        print(f"  QUBO built: {len(bqm.variables)} variables, {bqm.num_interactions} interactions")
        return bqm
    
    def _add_source_constraints(self, bqm: dimod.BinaryQuadraticModel, 
                               outgoing: List[int], incoming: List[int], 
                               penalty: float):
        """Add constraints for source node."""
        # Outflow = 1 constraint: (sum(outgoing) - 1)^2
        for edge_idx in outgoing:
            current = bqm.get_linear(edge_idx)
            bqm.set_linear(edge_idx, current + penalty)
        
        for i in range(len(outgoing)):
            for j in range(i+1, len(outgoing)):
                bqm.add_quadratic(outgoing[i], outgoing[j], 2*penalty)
        
        for edge_idx in outgoing:
            current = bqm.get_linear(edge_idx)
            bqm.set_linear(edge_idx, current - 2*penalty)
        
        bqm.offset += penalty
        
        # Inflow = 0 constraint: (sum(incoming))^2
        for edge_idx in incoming:
            current = bqm.get_linear(edge_idx)
            bqm.set_linear(edge_idx, current + penalty)
        
        for i in range(len(incoming)):
            for j in range(i+1, len(incoming)):
                bqm.add_quadratic(incoming[i], incoming[j], 2*penalty)
    
    def _add_target_constraints(self, bqm: dimod.BinaryQuadraticModel,
                               outgoing: List[int], incoming: List[int],
                               penalty: float):
        """Add constraints for target node."""
        # Inflow = 1 constraint: (sum(incoming) - 1)^2
        for edge_idx in incoming:
            current = bqm.get_linear(edge_idx)
            bqm.set_linear(edge_idx, current + penalty)
        
        for i in range(len(incoming)):
            for j in range(i+1, len(incoming)):
                bqm.add_quadratic(incoming[i], incoming[j], 2*penalty)
        
        for edge_idx in incoming:
            current = bqm.get_linear(edge_idx)
            bqm.set_linear(edge_idx, current - 2*penalty)
        
        bqm.offset += penalty
        
        # Outflow = 0 constraint: (sum(outgoing))^2
        for edge_idx in outgoing:
            current = bqm.get_linear(edge_idx)
            bqm.set_linear(edge_idx, current + penalty)
        
        for i in range(len(outgoing)):
            for j in range(i+1, len(outgoing)):
                bqm.add_quadratic(outgoing[i], outgoing[j], 2*penalty)
    
    def _add_intermediate_constraints(self, bqm: dimod.BinaryQuadraticModel,
                                     outgoing: List[int], incoming: List[int],
                                     penalty: float):
        """Add constraints for intermediate nodes."""
        if not outgoing and not incoming:
            return
        
        # Constraint: (sum(incoming) - sum(outgoing))^2
        
        # Terms from sum(incoming)^2
        for edge_idx in incoming:
            current = bqm.get_linear(edge_idx)
            bqm.set_linear(edge_idx, current + penalty)
        
        for i in range(len(incoming)):
            for j in range(i+1, len(incoming)):
                bqm.add_quadratic(incoming[i], incoming[j], 2*penalty)
        
        # Terms from sum(outgoing)^2
        for edge_idx in outgoing:
            current = bqm.get_linear(edge_idx)
            bqm.set_linear(edge_idx, current + penalty)
        
        for i in range(len(outgoing)):
            for j in range(i+1, len(outgoing)):
                bqm.add_quadratic(outgoing[i], outgoing[j], 2*penalty)
        
        # Cross terms: -2 * sum(incoming) * sum(outgoing)
        for in_idx in incoming:
            for out_idx in outgoing:
                bqm.add_quadratic(in_idx, out_idx, -2*penalty)
    
    def _add_resource_constraints(self, bqm: dimod.BinaryQuadraticModel,
                                 resource_penalty: float):
        """Add constraints for resource limits."""
        for resource, max_limit in self.resource_constraints.items():
            # Quadratic penalty for exceeding resource limit
            # We'll use a soft constraint: penalty * max(0, sum(resource_usage) - max_limit)^2
            
            # First, collect edges with this resource
            resource_edges = []
            resource_weights = []
            
            for (u, v), idx in self.edge_to_idx.items():
                resource_val = self.graph[u][v].get(resource, 0)
                if resource_val > 0:
                    resource_edges.append(idx)
                    resource_weights.append(resource_val)
            
            if not resource_edges:
                continue
            
            # Create a linear approximation of the quadratic penalty
            # For each edge, add penalty proportional to (resource_val^2 / max_limit)
            for idx, resource_val in zip(resource_edges, resource_weights):
                current = bqm.get_linear(idx)
                # Quadratic penalty approximation
                penalty_term = resource_penalty * (resource_val ** 2) / (max_limit ** 2)
                bqm.set_linear(idx, current + penalty_term)
            
            # Add pairwise penalties for edges that together might exceed limit
            total_possible = sum(resource_weights)
            if total_possible > max_limit:
                # Add penalties for combinations that exceed limit
                for i in range(len(resource_edges)):
                    for j in range(i+1, len(resource_edges)):
                        combined = resource_weights[i] + resource_weights[j]
                        if combined > max_limit:
                            penalty_term = resource_penalty * (combined - max_limit) / max_limit
                            bqm.add_quadratic(resource_edges[i], resource_edges[j], penalty_term)
    
    def apply_grover_mixer(self, bqm: dimod.BinaryQuadraticModel, 
                          num_iterations: int = 3,
                          num_feasible_to_keep: int = 10) -> dimod.BinaryQuadraticModel:
        """
        Apply Grover-like mixing to amplify probability of feasible solutions.
        
        Args:
            bqm: Original QUBO model
            num_iterations: Number of mixing iterations
            num_feasible_to_keep: Number of feasible solutions to remember
            
        Returns:
            Modified QUBO model with amplified feasible solution space
        """
        if not USE_GROVER_MIXER:
            print("Grover mixer disabled (USE_GROVER_MIXER = False)")
            return bqm
        
        print("\nApplying Grover-like mixer...")
        print(f"  Number of iterations: {num_iterations}")
        
        mixed_bqm = bqm.copy()
        sampler = SimulatedAnnealingSampler()
        
        for iteration in range(num_iterations):
            print(f"  Iteration {iteration + 1}/{num_iterations}...")
            
            # Sample from current QUBO
            response = sampler.sample(mixed_bqm, num_reads=100, annealing_time=10)
            
            # Collect feasible solutions
            iteration_feasible = []
            for sample, energy in response.data(['sample', 'energy']):
                if self.is_feasible(sample):
                    iteration_feasible.append((sample, energy))
            
            if iteration_feasible:
                # Keep best feasible solutions
                iteration_feasible.sort(key=lambda x: x[1])
                best_feasible = iteration_feasible[:num_feasible_to_keep]
                self.feasible_solutions_history.extend([s for s, _ in best_feasible])
                
                # Calculate amplification pattern from feasible solutions
                amplification_pattern = self._calculate_amplification_pattern(
                    [s for s, _ in best_feasible]
                )
                
                # Apply amplification to QUBO
                mixed_bqm = self._amplify_feasible_space(mixed_bqm, amplification_pattern)
                
                print(f"    Found {len(iteration_feasible)} feasible solutions")
                print(f"    Best energy: {iteration_feasible[0][1]:.2f}")
            else:
                print(f"    No feasible solutions found in this iteration")
        
        if self.feasible_solutions_history:
            print(f"\n  Grover mixer completed. Collected {len(self.feasible_solutions_history)} feasible solutions.")
        else:
            print("\n  Grover mixer completed but found no feasible solutions.")
        
        return mixed_bqm
    
    def _calculate_amplification_pattern(self, feasible_solutions: List[dict]) -> Dict[int, float]:
        """
        Calculate which variables are frequently 1 in feasible solutions.
        
        Args:
            feasible_solutions: List of feasible solution samples
            
        Returns:
            Dictionary mapping variable index to amplification strength
        """
        if not feasible_solutions:
            return {}
        
        # Count how often each variable is 1 in feasible solutions
        var_counts = defaultdict(int)
        for solution in feasible_solutions:
            for idx, value in solution.items():
                if value == 1:
                    var_counts[idx] += 1
        
        # Convert to probabilities
        total_solutions = len(feasible_solutions)
        amplification = {}
        for idx, count in var_counts.items():
            probability = count / total_solutions
            # Strongly amplify variables that appear in >70% of feasible solutions
            if probability > 0.7:
                amplification[idx] = -0.5  # Negative to favor
            # Suppress variables that appear in <30% of feasible solutions
            elif probability < 0.3:
                amplification[idx] = 0.3   # Positive to discourage
        
        return amplification
    
    def _amplify_feasible_space(self, bqm: dimod.BinaryQuadraticModel,
                               amplification_pattern: Dict[int, float]) -> dimod.BinaryQuadraticModel:
        """
        Modify QUBO to amplify probability of feasible solutions.
        
        Args:
            bqm: Original QUBO
            amplification_pattern: Dictionary of variable amplifications
            
        Returns:
            Modified QUBO
        """
        modified_bqm = bqm.copy()
        
        for idx, strength in amplification_pattern.items():
            if idx in modified_bqm.variables:
                current = modified_bqm.get_linear(idx)
                modified_bqm.set_linear(idx, current + strength)
        
        return modified_bqm
    
    def is_feasible(self, solution: dict) -> bool:
        """
        Check if a solution satisfies all constraints.
        
        Args:
            solution: Binary solution dictionary
            
        Returns:
            True if solution is feasible
        """
        # Convert to edge selection
        selected_edges = []
        for idx, value in solution.items():
            if value == 1:
                selected_edges.append(self.idx_to_edge[idx])
        
        # Check if it forms a valid path
        if not selected_edges:
            return False
        
        # Build subgraph
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
            if total_resources[resource] > limit + 0.001:  # Small tolerance
                return False
        
        return True
    
    def solution_to_path(self, solution: dict) -> Optional[List[int]]:
        """
        Convert binary solution to node path.
        
        Args:
            solution: Binary solution dictionary
            
        Returns:
            List of nodes representing the path, or None if invalid
        """
        selected_edges = []
        for idx, value in solution.items():
            if value == 1:
                selected_edges.append(self.idx_to_edge[idx])
        
        if not selected_edges:
            return None
        
        # Build graph from selected edges
        G = nx.DiGraph()
        G.add_edges_from(selected_edges)
        
        try:
            path = nx.shortest_path(G, self.source, self.target)
            return path
        except:
            return None
    
    def calculate_path_metrics(self, path: List[int]) -> Dict:
        """
        Calculate cost and resource usage for a path.
        
        Args:
            path: List of nodes
            
        Returns:
            Dictionary with cost and resource usage
        """
        if not path:
            return {'cost': float('inf'), 'resources': {}}
        
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
    
    def solve(self, num_reads: int = 1000, annealing_time: int = 20,
             use_hybrid: bool = False) -> Dict:
        """
        Solve the capacitated shortest path problem.
        
        Args:
            num_reads: Number of quantum annealing reads
            annealing_time: Annealing time per read
            use_hybrid: Use hybrid solver if available
            
        Returns:
            Dictionary with results
        """
        print("\n" + "="*60)
        print("SOLVING CAPACITATED SHORTEST PATH")
        print("="*60)
        
        start_time = time.time()
        
        # Build QUBO
        bqm = self.build_qubo(penalty_strength=20.0, resource_penalty=15.0)
        
        # Apply Grover mixer if enabled
        if USE_GROVER_MIXER:
            bqm = self.apply_grover_mixer(bqm, num_iterations=3)
        
        # Solve with appropriate sampler
        if use_hybrid:
            print("\nUsing hybrid solver...")
            try:
                sampler = LeapHybridSampler()
                response = sampler.sample(bqm, time_limit=min(30, annealing_time))
            except:
                print("Hybrid solver failed, falling back to simulated annealing...")
                sampler = SimulatedAnnealingSampler()
                response = sampler.sample(bqm, num_reads=num_reads, annealing_time=annealing_time)
        else:
            print(f"\nUsing simulated annealing with {num_reads} reads...")
            sampler = SimulatedAnnealingSampler()
            response = sampler.sample(bqm, num_reads=num_reads, annealing_time=annealing_time)
        
        solve_time = time.time() - start_time
        print(f"Solving completed in {solve_time:.2f} seconds")
        
        # Find best feasible solution
        best_solution = None
        best_energy = float('inf')
        feasible_count = 0
        all_feasible = []
        
        for sample, energy in response.data(['sample', 'energy']):
            if self.is_feasible(sample):
                feasible_count += 1
                all_feasible.append((sample, energy))
                if energy < best_energy:
                    best_energy = energy
                    best_solution = sample
        
        # Convert to path
        best_path = None
        if best_solution:
            best_path = self.solution_to_path(best_solution)
        
        return {
            'best_solution': best_solution,
            'best_energy': best_energy,
            'best_path': best_path,
            'feasible_count': feasible_count,
            'total_samples': len(response),
            'solve_time': solve_time,
            'all_feasible': all_feasible,
            'response': response
        }
    
    def plot_solution(self, path: List[int], title: str = "Solution", 
                     save_path: Optional[str] = None):
        """
        Plot the graph with highlighted solution path.
        
        Args:
            path: Solution path to highlight
            title: Plot title
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(14, 10))
        
        # Node colors and sizes
        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            if node == self.source:
                node_colors.append('green')
                node_sizes.append(1000)
            elif node == self.target:
                node_colors.append('red')
                node_sizes.append(1000)
            elif path and node in path:
                node_colors.append('orange')
                node_sizes.append(800)
            else:
                node_colors.append('lightblue')
                node_sizes.append(600)
        
        nx.draw_networkx_nodes(self.graph, self.pos, node_color=node_colors,
                              node_size=node_sizes, alpha=0.9)
        
        # Draw all edges
        all_edges = list(self.graph.edges())
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=all_edges,
                              edge_color='lightgray', width=2, alpha=0.4,
                              arrows=True, arrowstyle='-|>', arrowsize=15)
        
        # Highlight solution path
        if path:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(self.graph, self.pos, edgelist=path_edges,
                                  edge_color='red', width=4, alpha=0.9,
                                  arrows=True, arrowstyle='-|>', arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, self.pos, font_size=11, font_weight='bold')
        
        # Edge labels with weight and resources
        edge_labels = {}
        for (u, v) in self.graph.edges():
            weight = self.graph[u][v].get('weight', 1)
            label = f"w:{weight}"
            for resource in self.resource_constraints.keys():
                resource_val = self.graph[u][v].get(resource, 0)
                if resource_val > 0:
                    label += f"\n{resource[0]}:{resource_val}"
            edge_labels[(u, v)] = label
        
        nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels,
                                    font_size=8, font_color='darkblue')
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.9, label='Source'),
            Patch(facecolor='red', alpha=0.9, label='Target'),
            Patch(facecolor='orange', alpha=0.9, label='Path Node'),
            Patch(facecolor='white', edgecolor='red', linewidth=3, label='Selected Path'),
            Patch(facecolor='white', edgecolor='lightgray', linewidth=2, label='Other Edges')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def find_classical_solution(self) -> Dict:
        """
        Find classical solution using exhaustive search (for small graphs)
        or heuristic (for larger graphs).
        
        Returns:
            Dictionary with best classical solution found
        """
        print("\nFinding classical solution...")
        
        # Try to find all simple paths (works for small graphs)
        try:
            all_paths = list(nx.all_simple_paths(self.graph, self.source, self.target))
            
            feasible_paths = []
            for path in all_paths:
                metrics = self.calculate_path_metrics(path)
                feasible = True
                for resource, limit in self.resource_constraints.items():
                    if metrics['resources'][resource] > limit:
                        feasible = False
                        break
                
                if feasible:
                    feasible_paths.append((path, metrics['cost']))
            
            if feasible_paths:
                # Find minimum cost path
                best_path, best_cost = min(feasible_paths, key=lambda x: x[1])
                return {
                    'path': best_path,
                    'cost': best_cost,
                    'feasible_count': len(feasible_paths),
                    'method': 'exhaustive'
                }
        
        except:
            pass
        
        # For larger graphs, use a heuristic approach
        print("  Using heuristic approach (Dijkstra with resource filtering)...")
        
        # Try to find resource-constrained shortest path heuristically
        best_path = None
        best_cost = float('inf')
        
        # Try multiple resource thresholds
        for threshold_factor in [1.0, 1.2, 1.5, 2.0]:
            # Create filtered graph
            filtered_graph = self.graph.copy()
            edges_to_remove = []
            
            for u, v in filtered_graph.edges():
                total_resource = 0
                for resource, limit in self.resource_constraints.items():
                    resource_val = filtered_graph[u][v].get(resource, 0)
                    if resource_val > limit * threshold_factor:
                        edges_to_remove.append((u, v))
                        break
            
            filtered_graph.remove_edges_from(edges_to_remove)
            
            # Try to find path in filtered graph
            try:
                path = nx.shortest_path(filtered_graph, self.source, self.target, weight='weight')
                metrics = self.calculate_path_metrics(path)
                
                # Check if actually feasible
                feasible = True
                for resource, limit in self.resource_constraints.items():
                    if metrics['resources'][resource] > limit:
                        feasible = False
                        break
                
                if feasible and metrics['cost'] < best_cost:
                    best_path = path
                    best_cost = metrics['cost']
            except:
                continue
        
        if best_path:
            return {
                'path': best_path,
                'cost': best_cost,
                'feasible_count': 1,
                'method': 'heuristic'
            }
        
        return {
            'path': None,
            'cost': float('inf'),
            'feasible_count': 0,
            'method': 'failed'
        }


def create_capacitated_graph(num_nodes: int = 12, 
                           resource_names: List[str] = ['time']) -> nx.DiGraph:
    """
    Create a test graph with resource constraints.
    
    Args:
        num_nodes: Number of nodes
        resource_names: List of resource names
        
    Returns:
        Directed graph with weights and resources
    """
    print(f"Creating capacitated graph with {num_nodes} nodes...")
    
    # Create a random connected graph
    while True:
        G = nx.erdos_renyi_graph(num_nodes, 0.25, seed=42)
        if nx.is_connected(G):
            break
    
    # Convert to directed and add reverse edges
    G = G.to_directed()
    edges_to_add = []
    for u, v in list(G.edges()):
        if not G.has_edge(v, u):
            edges_to_add.append((v, u))
    
    for u, v in edges_to_add:
        G.add_edge(u, v)
    
    # Assign weights and resources
    random.seed(42)
    for u, v in G.edges():
        # Weight (cost)
        G[u][v]['weight'] = random.randint(1, 10)
        
        # Resources
        for resource in resource_names:
            # Some edges have high resource usage, some low
            if random.random() < 0.3:  # 30% of edges have high resource usage
                G[u][v][resource] = random.randint(3, 8)
            else:
                G[u][v][resource] = random.randint(1, 3)
    
    return G


def run_demonstration():
    """Run a complete demonstration of the capacitated shortest path solver."""
    print("="*70)
    print("CAPACITATED SHORTEST PATH WITH QUANTUM ANNEALING")
    print(f"Grover Mixer: {'ENABLED' if USE_GROVER_MIXER else 'DISABLED'}")
    print("="*70)
    
    # Create test graph
    G = create_capacitated_graph(num_nodes=10, resource_names=['time', 'cost'])
    
    # Set source and target
    source = 0
    target = 9
    
    # Set resource constraints
    resource_constraints = {
        'time': 15,    # Maximum total time
        'cost': 20     # Maximum total cost (in addition to weight objective)
    }
    
    print(f"\nProblem Setup:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Source: {source}")
    print(f"  Target: {target}")
    print(f"  Resource constraints: {resource_constraints}")
    
    # Create solver
    solver = CapacitatedShortestPathSolver(G, source, target, resource_constraints)
    
    # Plot original graph
    print("\nPlotting original graph...")
    resource_str = ', '.join([f"{k}:{v}" for k, v in resource_constraints.items()])
    solver.plot_solution(None, f"Original Graph\nResource constraints: {resource_str}")
    
    # Find classical solution
    classical_result = solver.find_classical_solution()
    
    if classical_result['path']:
        print(f"\nClassical Solution ({classical_result['method']}):")
        print(f"  Path: {classical_result['path']}")
        print(f"  Cost: {classical_result['cost']}")
        
        metrics = solver.calculate_path_metrics(classical_result['path'])
        print(f"  Resource usage:")
        for resource, usage in metrics['resources'].items():
            limit = resource_constraints[resource]
            status = "✓" if usage <= limit else "✗"
            print(f"    {resource}: {usage}/{limit} {status}")
        
        # Plot classical solution
        solver.plot_solution(classical_result['path'], 
                           f"Classical Solution\nCost: {classical_result['cost']}")
    else:
        print("\nNo feasible classical solution found!")
    
    # Solve with quantum annealing
    print("\n" + "="*60)
    print("QUANTUM ANNEALING SOLUTION")
    print("="*60)
    
    quantum_result = solver.solve(num_reads=500, annealing_time=15, use_hybrid=False)
    
    print(f"\nQuantum Results:")
    print(f"  Feasible solutions found: {quantum_result['feasible_count']}/{quantum_result['total_samples']}")
    print(f"  Best energy: {quantum_result['best_energy']:.2f}")
    
    if quantum_result['best_path']:
        metrics = solver.calculate_path_metrics(quantum_result['best_path'])
        print(f"\n  Best path found: {quantum_result['best_path']}")
        print(f"  Path cost: {metrics['cost']}")
        print(f"  Resource usage:")
        
        all_constraints_satisfied = True
        for resource, usage in metrics['resources'].items():
            limit = resource_constraints[resource]
            satisfied = usage <= limit
            if not satisfied:
                all_constraints_satisfied = False
            status = "✓" if satisfied else "✗"
            print(f"    {resource}: {usage}/{limit} {status}")
        
        if all_constraints_satisfied:
            print(f"  ✓ All constraints satisfied!")
        else:
            print(f"  ✗ Some constraints violated")
        
        # Plot quantum solution
        title = f"Quantum Solution"
        if all_constraints_satisfied:
            title += f"\nCost: {metrics['cost']}"
        else:
            title += f"\nCost: {metrics['cost']} (INFEASIBLE)"
        
        solver.plot_solution(quantum_result['best_path'], title)
        
        # Compare with classical if both exist
        if classical_result['path'] and all_constraints_satisfied:
            print(f"\nComparison:")
            print(f"  Classical cost: {classical_result['cost']}")
            print(f"  Quantum cost: {metrics['cost']}")
            
            if abs(metrics['cost'] - classical_result['cost']) < 0.01:
                print(f"  ✓ Quantum found optimal solution!")
            else:
                print(f"  ✗ Quantum solution is {metrics['cost'] - classical_result['cost']:.2f} from optimal")
    
    # Analyze feasible solutions found
    if quantum_result['all_feasible']:
        print(f"\nFeasible Solutions Analysis:")
        print(f"  Found {len(quantum_result['all_feasible'])} unique feasible solutions")
        
        # Group by path
        path_solutions = {}
        for sample, energy in quantum_result['all_feasible']:
            path = solver.solution_to_path(sample)
            if path:
                path_tuple = tuple(path)
                if path_tuple not in path_solutions:
                    path_solutions[path_tuple] = {
                        'count': 0,
                        'min_energy': energy,
                        'cost': solver.calculate_path_metrics(list(path_tuple))['cost']
                    }
                path_solutions[path_tuple]['count'] += 1
                path_solutions[path_tuple]['min_energy'] = min(
                    path_solutions[path_tuple]['min_energy'], energy
                )
        
        if path_solutions:
            print(f"  Unique feasible paths: {len(path_solutions)}")
            print(f"  Most frequent paths:")
            sorted_paths = sorted(path_solutions.items(), key=lambda x: x[1]['count'], reverse=True)
            for i, (path_tuple, info) in enumerate(sorted_paths[:3]):  # Top 3
                path = list(path_tuple)
                print(f"    {i+1}. Path: {path} (count: {info['count']}, cost: {info['cost']})")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    # You can toggle this flag at the top of the file
    print(f"Grover Mixer: {'ENABLED' if USE_GROVER_MIXER else 'DISABLED'}")
    run_demonstration()