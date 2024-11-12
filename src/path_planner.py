from collections import defaultdict

class Graph:
    def __init__(self, connections, directed=False):
        self._graph = defaultdict(dict)
        self._directed = directed
        self.add_connections(connections)
        
    def add_connections(self, connections):
        """Add connections (list of tuples) to the graph. Each tuple is (node1, node2, weight)."""
        for node1, node2, weight in connections:
            self.add(node1, node2, weight)
    
    def add(self, node1, node2, weight=1):
        """Add connection between node1 and node2 with an edge weight."""
        self._graph[node1][node2] = weight
        if not self._directed:
            self._graph[node2][node1] = weight
    
    def remove(self, node):
        """Remove all references to a node."""
        for n, cxns in self._graph.items():
            cxns.pop(node, None)
        self._graph.pop(node, None)
    
    def is_connected(self, node1, node2):
        """Check if node1 is directly connected to node2."""
        return node1 in self._graph and node2 in self._graph[node1]

    def get_weight(self, node1, node2):
        """Return the weight of the edge between node1 and node2, or None if no edge exists."""
        return self._graph[node1].get(node2)

    def find_path(self, node1, node2, path=[]):
        """Find any path between node1 and node2 (may not be the shortest)."""
        path = path + [node1]
        if node1 == node2:
            return path
        if node1 not in self._graph:
            return None
        for node in self._graph[node1]:
            if node not in path:
                new_path = self.find_path(node, node2, path)
                if new_path:
                    return new_path
        return None

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))
    
    def print_graph(self):
        """Print each node and its connected nodes with weights."""
        for node, edges in self._graph.items():
            connections = ', '.join(f"{neighbor} (weight: {weight})" for neighbor, weight in edges.items())
            print(f"{node} --> {connections}")


def path_planner():
    
    test_graph = Graph(((1, 2, 3),))
    
    test_graph.add(4, 7, 4)
    test_graph.add(9, 7, 5)
    
    test_graph.print_graph()
    
    # Open the dem as a raster
    
    
    # Make a graph of the raster with costs based on just the elevation values
    
    
    # Include costs for important landmarks. 
    # Make that as an option where the user can enter as many important landmarks and the graph gets updated
    
    # Include cost for dangerous sites that need to be avoided (like craters and the like)
    
    # Include cost for solar exposure (later)
    
    
    pass

if __name__ == "__main__":
    path_planner()