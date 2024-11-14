from collections import defaultdict
import heapq

from osgeo import gdal 

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

import pickle

class newGraph:
    def __init__(self, matrix):
        self.matrix = matrix
        self.graph = self._create_graph_from_matrix()

    def _create_graph_from_matrix(self):
        n, m, _ = self.matrix.shape
        new_matrix = np.zeros((n, m, 5))

        # Original values (0th index)
        new_matrix[:, :, 0] = self.matrix[:, :, 0]

        # Left neighbor (1st index)
        left = np.pad(self.matrix[:, :-1, 0], ((0, 0), (1, 0)), mode='constant', constant_values=0)
        new_matrix[:, :, 1] = np.abs(self.matrix[:, :, 0] - left)  # Difference with left neighbor

        # Top neighbor (2nd index)
        top = np.pad(self.matrix[:-1, :, 0], ((1, 0), (0, 0)), mode='constant', constant_values=0)
        new_matrix[:, :, 2] = np.abs(self.matrix[:, :, 0] - top)  # Difference with top neighbor

        # Right neighbor (3rd index)
        right = np.pad(self.matrix[:, 1:, 0], ((0, 0), (0, 1)), mode='constant', constant_values=0)
        new_matrix[:, :, 3] = np.abs(self.matrix[:, :, 0] - right)  # Difference with right neighbor

        # Bottom neighbor (4th index)
        bottom = np.pad(self.matrix[1:, :, 0], ((0, 1), (0, 0)), mode='constant', constant_values=0)
        new_matrix[:, :, 4] = np.abs(self.matrix[:, :, 0] - bottom)  # Difference with bottom neighbor

        return new_matrix

    def print_graph(self):
        print(self.graph)

    def dijkstra(self, start, end):
        n, m, _ = self.graph.shape
        # distance matrix initialized to infinity
        distances = np.full((n, m), np.inf)
        distances[start] = 0  # set start node distance to 0
        
        # min-heap priority queue
        pq = [(0, start)]  # (distance, (i, j))

        # previous node matrix to reconstruct the path
        previous = np.full((n, m), None)

        # directions for neighbors (left, top, right, bottom)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (dy, dx)

        while pq:
            # fetch node with the smallest distance
            current_dist, (i, j) = heapq.heappop(pq)

            # if we reached the end, stop
            if (i, j) == end:
                return current_dist, self.reconstruct_path(previous, end)

            # explore neighbors
            for idx, (dy, dx) in enumerate(directions):
                ni, nj = i + dy, j + dx

                # check if the neighbor is within bounds
                if 0 <= ni < n and 0 <= nj < m:
                    # calculate new distance using the graph's cost to move to the neighbor
                    neighbor_cost = self.graph[i, j, idx + 1]
                    new_dist = current_dist + neighbor_cost

                    # if the new distance is shorter, update and add to the queue
                    if new_dist < distances[ni, nj]:
                        distances[ni, nj] = new_dist
                        previous[ni, nj] = (i, j)  # track the previous node
                        heapq.heappush(pq, (new_dist, (ni, nj)))

        return np.inf, []  # if there's no path to the end

    def reconstruct_path(self, previous, end):
        # Reconstruct the path by backtracking from the end to the start using the previous matrix
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()  # reverse the path to get from start to end
        return path

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
            
    def save_graph(graph, filename):
        with open(filename, 'wb') as f:
            pickle.dump(graph, f)
        print(f"Graph saved to {filename}")

def load_graph(filename) -> Graph:
    with open(filename, 'rb') as f:
        graph = pickle.load(f)
    print(f"Graph loaded from {filename}")
    return graph

def test_graph():
    sample_graph = Graph(((1, 2, 3),))
    
    sample_graph.add(4, 7, 4)
    sample_graph.add(9, 7, 5)
    
    sample_graph.print_graph()


def path_planner():
    
    #+-+-+-+-+-+-+-+-+-+-+-+- Open the dtm as a raster image +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
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
    
    x_start, y_start = 2200, 3600
    x_end, y_end = 2800, 4200
    img_cropped = img_normalized[y_start:y_end, x_start:x_end]
    
    print(np.mean(img_normalized))
    
    plt.imshow(img_cropped)
    # plt.savefig('Cropped_Tiff.png')
    plt.show()
    
    #+-+-+-+-+-+-+-+-+- Make a graph of the raster with costs based on just the elevation values +-+-+-+-+-+-+-+-+-
    # coordinate_2_index = [25, 25]
    # print(img_cropped[coordinate_2_index[0], coordinate_2_index[1]])
    
    
    rows, cols, _ = img_cropped.shape
    
    single_channel_img_cropped = np.zeros((rows, cols, 1))
    
    single_channel_img_cropped[:, :, 0] = img_cropped[:, :, 0]
    
    dtm_graph = newGraph(single_channel_img_cropped)

    print("Graph Created Successfully")
    
    
    start_coordinates = (0, 0)
    end_coordinates = (555, 555)
    
    cost, path = dtm_graph.dijkstra(start_coordinates, end_coordinates)
    
    print(f"Shortest Path from {start_coordinates} to {end_coordinates}: {path}, Total Cost: {cost}")
    
    path_x, path_y = zip(*path)
    
    plt.imshow(img_cropped)
    plt.plot(path_y, path_x, color='red', linewidth=2, marker='o', markersize=1)  # Draw the path
    plt.title("Shortest Path on the Original Image")
    plt.show()
    
    
    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

    # coords_1 = indices[:, None, :]
    # coords_2 = indices[None, :, :]

    # weights_1 = img_normalized[coords_1[:, :, 0], coords_1[:, :, 1]]
    # weights_2 = img_normalized[coords_2[:, :, 0], coords_2[:, :, 1]]

    # weight_diff = weights_2 - weights_1

    # connections = [(i, j, weight_diff[i, j]) for i in range(len(indices)) for j in range(len(indices))]
    
    # Include costs for important landmarks. 
    # Make that as an option where the user can enter as many important landmarks and the graph gets updated
    
    # Include cost for dangerous sites that need to be avoided (like craters and the like)
    
    # Include cost for solar exposure (later)
    
    
    pass

def path_planner2():
    
    conections = ((1, 2, 3),)
    test = Graph(conections)
    
    print(test._graph)

if __name__ == "__main__":
    # path_planner2()
    path_planner()