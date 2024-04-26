import numpy as np
import matplotlib.pyplot as plt
import argparse


class Node:
    def __init__(self, value, number, connections=None):

        self.index = number
        self.connections = connections
        self.value = value

class Network:

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def get_mean_degree(self):
        degree_list = []
        # finds the total number of degrees across all nodes
        for node in self.nodes:
            degree_list.append(sum(node.connections))
        degree_list = [sum(node.connections) for node in self.nodes]
        # divides total degrees by number of nodes for mean
        return sum(degree_list)/len(degree_list)

    def get_mean_clustering(self):
        # makes a list containing the clustering coefficient of each node
        coefficients_list = []
        for node in self.nodes:
            # checks neighbours of each connection
            neighbors = [self.nodes[i] for i, connection in enumerate(node.connections) if connection == 1]
            # to skip checking when there cannot be a connection
            if len(neighbors) < 2:
                coefficients_list.append(0)
                continue
            node_connections = 0
            # ckecks for neighbour connections for all nodes
            for var1, neighbor1 in enumerate(neighbors):
                for var2, neighbor2 in enumerate(neighbors):
                    # to avoid checking a pair twice
                    if var1 < var2 and neighbor1.connections[neighbor2.index] == 1:
                        # counts connections between neighbours
                        node_connections += 1
            possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
            # adds clustering coefficient of each node to the list
            coefficients_list.append(node_connections / possible_connections)
        # finds sum of the list and divides by number of nodes in list for mean
        return sum(coefficients_list) / len(coefficients_list)

    # function to find number of nodes connected to a node
    def get_num_connected_nodes(self, node):
        # list where 0 = unvisited and 1 = visited
        visited = [0] * len(self.nodes)
        visited_list = [node]
        # to mark as visited
        visited[node.index] = 1
        node_count = 0

        while visited_list:
            current_node = visited_list.pop(0)
            node_count += 1
            for i, connection in enumerate(current_node.connections):
                if connection == 1 and not visited[i]:
                    visited[i] = 1
                    visited_list.append(self.nodes[i])
        # checks if node has no connections
        if node_count == 1:
            return node_count
        # -1 to subtract original node
        return node_count-1

    def shortest_path_length(self, start_node):
        # list where 0 = unvisited and 1 = visited
        visited = [0] * len(self.nodes)
        # to set number of edges as distance
        distance = [0] * len(self.nodes)
        visited_list = [start_node]
        # to set as  visited
        visited[start_node.index] = 1

        while visited_list:
            current_node = visited_list.pop(0)
            # loop to check for all connections
            for i, connection in enumerate(current_node.connections):
                if connection == 1 and not visited[i]:
                    visited[i] = 1
                    # adds 1 to the distance
                    distance[i] = distance[current_node.index] + 1
                    # adds node to visited list
                    visited_list.append(self.nodes[i])
        return sum(distance)

    def get_mean_path_length(self):

        # list including distances between all nodes
        path_lenths_list=[]
        total_shortest_paths = 0
        for start_node in self.nodes:
            total_shortest_paths += self.shortest_path_length(start_node)
            # adds shortest distance divided by number of connections to the list
            path_lenths_list.append(self.shortest_path_length(start_node) / self.get_num_connected_nodes(start_node))
        # divides total distance by number of nodes for mean
        return sum(path_lenths_list)/ len(self.nodes)

    def make_random_network(self, N, connection_probability=0.5):

        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

#plotting
    def plot(self):
        for node in self.nodes:
            for x, connection in enumerate(node.connections):
                if connection == 1:
                    plt.plot([node.index, x], [node.value, self.nodes[x].value], color='black')
        plt.scatter([node.index for node in self.nodes], [node.value for node in self.nodes])
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-network", type=float)
    parser.add_argument("-test_network", action='store_true')

    args = parser.parse_args()
    network_size = args.network

    if args.network:
        network = Network()
        network.make_random_network(int(network_size))
        network.plot()
        mean_degree = network.get_mean_degree()
        mean_clustering = network.get_mean_clustering()
        mean_path_length = network.get_mean_path_length()

        print("Mean Degree:", mean_degree)
        print("Mean Clustering Coefficient:", mean_clustering)
        print("Mean Path Length:", mean_path_length)

    if args.test_network:
        nodes = []
        num_nodes = 10
        for node_number in range(num_nodes):
            connections = [1 for val in range(num_nodes)]
            connections[node_number] = 0
            new_node = Node(0, node_number, connections=connections)
            nodes.append(new_node)
        network = Network(nodes)

        assert (network.get_mean_degree() == num_nodes - 1), network.get_mean_degree()
        assert (network.get_mean_clustering() == 1), network.get_mean_clustering()
        assert (network.get_mean_path_length() == 1), network.get_mean_path_length()
        print("passed")

if __name__ == "__main__":
    main()
