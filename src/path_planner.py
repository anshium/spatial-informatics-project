from collections import defaultdict
import heapq

from osgeo import gdal 

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

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

    def find_shortest_path(self, start, end):
        """Find the shortest path from start to end using Dijkstra's algorithm."""
        priority_queue = [(0, [start])]
        shortest_distances = {start: 0}

        while priority_queue:
            current_cost, path = heapq.heappop(priority_queue)
            current_node = path[-1]

            if current_node == end:
                return path, current_cost

            # explore neighbors
            for neighbor, weight in self._graph[current_node].items():
                new_cost = current_cost + weight
                
                # only consider new path if it is better
                if neighbor not in shortest_distances or new_cost < shortest_distances[neighbor]:
                    shortest_distances[neighbor] = new_cost
                    heapq.heappush(priority_queue, (new_cost, path + [neighbor]))

        return None, float('inf')

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))
    
    def print_graph(self):
        """Print each node and its connected nodes with weights."""
        for node, edges in self._graph.items():
            connections = ', '.join(f"{neighbor} (weight: {weight})" for neighbor, weight in edges.items())
            print(f"{node} --> {connections}")


def test_graph():
    sample_graph = Graph(((1, 2, 3),))
    
    sample_graph.add(4, 7, 4)
    sample_graph.add(9, 7, 5)
    
    sample_graph.print_graph()
    

def path_planner():
    
    # Open the dtm as a raster image
    DATASETS_PATH = "datasets"
    
    dtm_path_tiff = f"{DATASETS_PATH}/HiRISE/hello.tiff"

    dtm_image = gdal.Open(dtm_path_tiff)
    
    band1 = dtm_image.GetRasterBand(1) # Red channel 
    band2 = dtm_image.GetRasterBand(1) # Green channel 
    band3 = dtm_image.GetRasterBand(1) # Blue channel
    
    b1 = band1.ReadAsArray() 
    b2 = band2.ReadAsArray() 
    b3 = band3.ReadAsArray() 
    
    dtm_image_array = np.array(dtm_image)
    
    print(np.max(dtm_image_array))


    img = np.dstack((b1, b2, b3)) 

    min_val = -3961.39
    max_val = -3927.94
    
    img_clipped = np.clip(img, min_val, max_val)
    
    img_normalized = ((img_clipped - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    x_start, y_start = 1000, 1000
    x_end, y_end = 4000, 7000
    img_cropped = img_normalized[y_start:y_end, x_start:x_end]
    
    print(np.mean(img_normalized))
    
    plt.imshow(img_cropped)
    plt.savefig('Cropped_Tiff.png')
    plt.show()
    
    
    # Make a graph of the raster with costs based on just the elevation values
    
    
    # Include costs for important landmarks. 
    # Make that as an option where the user can enter as many important landmarks and the graph gets updated
    
    # Include cost for dangerous sites that need to be avoided (like craters and the like)
    
    # Include cost for solar exposure (later)
    
    
    pass

if __name__ == "__main__":
    path_planner()