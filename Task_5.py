import matplotlib.pyplot as plt
import numpy as np
import random
import math
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

def neighbours_opinions_network(node, network):
    opinions_list = [network.nodes[i].value for i, connected in enumerate(node.connections) if connected == 1]
    return opinions_list

def neighbours_opinions(population, i, j):
    k, m = population.shape
    opinions_list = []
    opinions_list.append(population[(i - 1) % m, j])
    opinions_list.append(population[(i + 1) % m, j])
    opinions_list.append(population[i, (j + 1) % k])
    opinions_list.append(population[i, (j - 1) % k])
    return opinions_list


def calculate_agreement(population, row, col, external=0.0):
    '''
    This function should return the *change* in agreement that would result if the cell at (row, col) was to flip it's value
    Inputs: population (numpy array)
            row (int)
            col (int)
            external (float)
    Returns:
            change_in_agreement (float)
    '''
    person = population[row, col]
    opinions = neighbours_opinions(population, row, col)
    agreement = 0
    '''print(opinions)
    for o in opinions:
        agreement += o * person'''
    agreement = sum(person * o for o in opinions)
    agreement += external * person
    return agreement

def ising_step_network(network, external=0.0, alpha=1.0):
    node = random.choice(network.nodes)
    opinions = neighbours_opinions_network(node, network)
    agreement = sum(node.value * o for o in opinions) + external * node.value

    if agreement < 0 or (alpha and random.random() < math.e ** (-agreement / alpha)):
        node.value *= -1
# def ising_step(population, external=0.0, alpha = 1.0):
#     '''
#     This function will perform a single update of the Ising model
#     Inputs: population (numpy array)
#             external (float) - optional - the magnitude of any external "pull" on opinion
#     '''
#
#     n_rows, n_cols = population.shape
#     row = np.random.randint(0, n_rows)
#     col = np.random.randint(0, n_cols)
#
#     agreement = calculate_agreement(population, row, col, external=0.0)
#
#     if agreement < 0:
#         population[row, col] *= -1
#     elif alpha:
#         random_prob = random.random()
#         if random_prob < math.e ** (-agreement/alpha):
#             population[row, col] *= -1


def plot_ising(im, population):
    '''
    This function will display a plot of the Ising model
    '''

    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.015)




def test_ising():
    '''
    This function will test the calculate_agreement function in the Ising model
    '''


print("Testing ising model calculations")
population = -np.ones((3, 3))
assert (calculate_agreement(population, 1, 1) == 4), "Test 1"

population[1, 1] = 1.
assert (calculate_agreement(population, 1, 1) == -4), "Test 2"

population[0, 1] = 1.
assert (calculate_agreement(population, 1, 1) == -2), "Test 3"

population[1, 0] = 1.
assert (calculate_agreement(population, 1, 1) == 0), "Test 4"

population[2, 1] = 1.
assert (calculate_agreement(population, 1, 1) == 2), "Test 5"

population[1, 2] = 1.
assert (calculate_agreement(population, 1, 1) == 4), "Test 6"

"Testing external pull"
population = -np.ones((3, 3))
assert (calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
assert (calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
assert(calculate_agreement(population,1,1,10)==-6), "Test 9"
assert(calculate_agreement(population,1,1, -10)==14), "Test 10"

print("Tests passed")


def ising_main_network(network, alpha=None, external=0.0):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_axis_off()
    # im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step_network(network, external, alpha)
            plot_network(network)

def plot_network(network):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    for node in network.nodes:
        for i, connected in enumerate(node.connections):
            if connected:
                # Draw edge
                plt.plot([node.index, i], [node.value, network.nodes[i].value], 'k-', lw=0.5)

    # Plot nodes with different colors based on their value
    colors = ['red' if node.value == 1 else 'blue' for node in network.nodes]
    positions = [(node.index, node.value) for node in network.nodes]  # Assuming positions are indexed
    positions = np.array(positions)

    plt.scatter(positions[:, 0], positions[:, 1], c=colors, s=100)  # Size of node

    plt.title("Network State")
    plt.xlabel("Node Index")
    plt.ylabel("Node Value")
    plt.grid(True)
    plt.show()
    plt.close('all')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ising_model", action='store_true')
    parser.add_argument("-alpha", type=float, default=1)
    parser.add_argument("-external", type=float, default=0)
    parser.add_argument("-test_ising", action='store_true')
    parser.add_argument("-use_network", type=int)
    args = parser.parse_args()
    alpha = args.alpha
    external = args.external
    #
    # if args.ising_model:
    #     population = np.random.choice([1, -1], size=(100,100))
    #     ising_main(population, alpha, external)
    if args.test_ising:
        test_ising()
    if args.use_network:
        network = Network()
        network.make_random_network(args.use_network)
        ising_main_network(network, alpha, external)

if __name__ == "__main__":
    main()