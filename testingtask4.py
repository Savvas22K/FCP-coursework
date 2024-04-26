import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
#why for the other graphs is it plasma
class Node:
    def __init__(self, value, index, color=1, connections=None):
        if connections is None:
            connections = [0] * 100  #change this if needed
        self.value = value
        self.index = index
        self.color = color
        self.connections = connections

class Network:
    def __init__(self, nodes):
        self.nodes = nodes

    def make_ring_network(self, N, neighbour_range=1):
        colors = cm.hot(np.linspace(0, 1, N))  #for the colors
        self.nodes = []
        for node_number in range(N):
            connections = [0] * N
            for j in range(1, neighbour_range + 1):
                connections[(node_number - j) % N] = 1
                connections[(node_number + j) % N] = 1
            self.nodes.append(Node(0, node_number, colors[node_number], connections=connections))

    def make_small_world_network(self, N, re_wire_prob=0.2):
        colors = cm.hot(np.linspace(0, 1, N))  #for the colors
        self.make_ring_network(N, 1)
        for i, node in enumerate(self.nodes):
            node.color = colors[i]  #assigns the color
            targets_to_consider = list(enumerate(node.connections))
            for idx, connected in targets_to_consider:
                if connected and random.random() < re_wire_prob:
                    node.connections[idx] = 0
                    possible_new_targets = [i for i in range(N) if not node.connections[i] and i != node.index]
                    if possible_new_targets:
                        new_target = random.choice(possible_new_targets)
                        node.connections[new_target] = 1

    def get_mean_degree(self):
        degree_list = []
        # finds the total number of degrees across all nodes
        for node in self.nodes:
            degree_list.append(sum(node.connections))
        degree_list = [sum(node.connections) for node in self.nodes]
        # divides total degrees by number of nodes for mean
        return sum(degree_list)/len(degree_list)
    
    def get_clustering(self):
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
    
    def get_path_length(self):
        def shortest_path_length(start_node):
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
        # list including distances between all nodes
        path_lenths_list=[]
        total_shortest_paths = 0
        for start_node in self.nodes:
            total_shortest_paths += shortest_path_length(start_node)
            # adds shortest distance divided by number of connections to the list
            path_lenths_list.append(shortest_path_length(start_node) / self.get_num_connected_nodes(start_node))
        # divides total distance by number of nodes for mean
        return sum(path_lenths_list)/ len(self.nodes)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        node_radius = 0.2 * network_radius  #to make the nodes bigger
        ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
        ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

        for node in self.nodes: #not the same as starter code
            node_angle = node.index * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            '''circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=node.color)
            ax.add_patch(circle)'''
            circle = plt.Circle((node_x, node_y), node_radius, color=node.color)
            ax.add_patch(circle) #for making the nodes bigger

            for neighbour_index, connected in enumerate(node.connections):
                if connected:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)
                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

        plt.show()


def main(): #testing
    parser = argparse.ArgumentParser(description="Generate and plot network structures.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-ring_network", type=int, help="Create a ring network with a specified number of nodes.")
    group.add_argument("-small_world", type=int, help="Number of nodes in the small-world network.")
    parser.add_argument("-re_wire", "--rewiring_probability", type=float, default=0.2, help="Rewiring probability (default: 0.2 for small-world network)")
    parser.add_argument("-r", "--range", type=int, default=2, help="Connectivity range (default: 2 for small-world network)")

    args = parser.parse_args()

    if args.small_world is not None:
        n_nodes = args.small_world
        network = Network([])
        network.make_small_world_network(n_nodes)
        network.plot()
    elif args.ring_network is not None:
        n_nodes = args.ring_network
        network = Network([])
        network.make_ring_network(n_nodes)
        network.plot()

if __name__ == "__main__":
    main()
def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

	print("All tests passed")

test_networks()