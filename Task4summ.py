import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Network:
    def __init__(self, n, r):
        self.n = n
        self.r = r
        self.network = self.make_ring_network()

    def make_ring_network(self):
        network = {}
        for i in range(self.n):
            connected_nodes = set()
            for j in range(1, self.r + 1):
                connected_nodes.add((i - j) % self.n)
                connected_nodes.add((i + j) % self.n)
            network[i] = connected_nodes
        return network

    def make_small_world_network(self, p):
        self.network = self.make_ring_network()  
        for node in list(self.network.keys()):
            targets_to_consider = list(self.network[node])
            for target in targets_to_consider:
                if random.random() < p:
                    self.network[node].remove(target)
                    possible_new_targets = set(range(self.n)) - self.network[node] - {node}
                    if possible_new_targets:
                        new_target = random.choice(list(possible_new_targets))
                        self.network[node].add(new_target)
        return self.network

class NetworkPlotter:
    def __init__(self, network):
        self.network = network

    def plot(self, title):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.network)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

        '''for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')''' #check if i can keep this out

        for node, connections in self.network.items():
            node_angle = node * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            ax.plot(node_x, node_y, 'o', color='black')  

            for neighbor in connections:
                neighbor_angle = neighbor * 2 * np.pi / num_nodes
                neighbor_x = network_radius * np.cos(neighbor_angle)
                neighbor_y = network_radius * np.sin(neighbor_angle)

                ax.plot([node_x, neighbor_x], [node_y, neighbor_y], 'k-', lw=0.5)  

        plt.title(title)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Generate and plot network structures.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-ring_network", type=int, help="Create a ring network with a specified number of nodes.")
    group.add_argument("-small_world", type=int, help="Number of nodes in the small-world network.")
    parser.add_argument("-re_wire", "--rewiring_probability", type=float, default=0.2, help="Rewiring probability (default: 0.2 for small-world network)")
    parser.add_argument("-r", "--range", type=int, default=2, help="Connectivity range (default: 2 for small-world network)")

    args = parser.parse_args()

    if args.ring_network:
        n_nodes = args.ring_network
        connectivity_range = 1
        network = Network(n_nodes, connectivity_range).network
        NetworkPlotter(network).plot(f"Ring Network (N={n_nodes}, Range={connectivity_range})")
    elif args.small_world:
        n_nodes = args.small_world
        connectivity_range = args.range
        rewiring_probability = args.rewiring_probability
        network = Network(n_nodes, connectivity_range).make_small_world_network(rewiring_probability)
        NetworkPlotter(network).plot(f"Small-World Network (N={n_nodes}, Range={connectivity_range}, Rewiring Probability={rewiring_probability})")

if __name__ == "__main__":
    main()

#testing -do something about this
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





