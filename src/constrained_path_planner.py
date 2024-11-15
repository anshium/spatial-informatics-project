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

    def dijkstra_with_near_avoid(self, start, end, avoid, buffer_radius):
        n, m, _ = self.graph.shape
        # Create a mask for restricted zones
        restricted = np.zeros((n, m), dtype=bool)

        # Mark points in the buffer radius around avoid as restricted
        ax, ay = avoid
        for i in range(n):
            for j in range(m):
                if np.sqrt((i - ax) ** 2 + (j - ay) ** 2) <= buffer_radius:
                    restricted[i, j] = True

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

                # check if the neighbor is within bounds and not restricted
                if 0 <= ni < n and 0 <= nj < m and not restricted[ni, nj]:
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
    
def find_path_via_near(graph, start, end, avoid, buffer_radius):
        # Step 1: Find a point near 'avoid' but outside the buffer zone
        near_point = None
        for i in range(avoid[0] - buffer_radius, avoid[0] + buffer_radius + 1):
            for j in range(avoid[1] - buffer_radius, avoid[1] + buffer_radius + 1):
                if 0 <= i < graph.graph.shape[0] and 0 <= j < graph.graph.shape[1]:
                    if np.sqrt((i - avoid[0]) ** 2 + (j - avoid[1]) ** 2) > buffer_radius:
                        near_point = (i, j)
                        break
            if near_point:
                break

        if not near_point:
            raise ValueError("No valid near point found outside buffer radius.")

        # Step 2: Find path from start to near_point
        cost1, path1 = graph.dijkstra(start, near_point)

        # Step 3: Find path from near_point to end
        cost2, path2 = graph.dijkstra(near_point, end)

        # Combine the paths
        full_path = path1 + path2[1:]  # Exclude the duplicate near_point in the combined path
        total_cost = cost1 + cost2

        return full_path, total_cost


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
    
    
    start_coordinates = (170, 60)
    end_coordinates = (555, 400)
    avoid_point = (267, 100)
    buffer_radius = 60

    # Find the path
    cost, path = dtm_graph.dijkstra_with_near_avoid(start_coordinates, end_coordinates, avoid_point, buffer_radius)

    # Visualize the path
    path_x, path_y = zip(*path)
    
    fig, ax = plt.subplots()
    ax.imshow(img_cropped)

    # Draw the buffer zone around the avoid point
    avoid_circle = plt.Circle((avoid_point[1], avoid_point[0]), buffer_radius, color='blue', fill=False, linewidth=2, linestyle='--')
    ax.add_patch(avoid_circle)

    # Plot the shortest path
    ax.plot(path_y, path_x, color='red', linewidth=2, marker='o', markersize=1)

    ax.set_title("Shortest Path with Buffer Zone")
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

if __name__ == "__main__":
    # path_planner2()
    path_planner()