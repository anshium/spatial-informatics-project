import numpy as np
import heapq

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

def main():
    test_matrix = np.array([[[1], [2], [3]],
                            [[4], [100], [6]],
                            [[7], [8], [9]]])

    graph = newGraph(test_matrix)

    graph.print_graph()

    start = (0, 0)
    end = (2, 2)
    shortest_path_cost, path_sequence = graph.dijkstra(start, end)
    
    print(f"The shortest path cost from {start} to {end} is: {shortest_path_cost}")
    print(f"The path taken (as a list of coordinates) is: {path_sequence}")


if __name__ == "__main__":
    main()