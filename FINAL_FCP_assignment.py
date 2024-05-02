import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import random
import argparse
from matplotlib.animation import FuncAnimation

class Individual:
    def __init__(self, opinion):
        self.opinion = opinion


# class a society
class Society:
    def __init__(self, size, beta, threshold):
        self.individuals = [Individual(random.uniform(0, 1)) for _ in range(size)]
        self.beta = beta
        self.threshold = threshold

    def update_opinions(self):  # create a list to store the opinions updated
        updated_opinions = [None] * len(self.individuals)
        # find the index of neighbour
        for _ in range(len(self.individuals)):
            random_index = np.random.randint(len(self.individuals))
            left_index = (random_index - 1) if random_index > 0 else None
            right_index = (random_index + 1) if random_index < len(self.individuals) - 1 else None
            neighbour_index = np.random.choice([left_index, right_index])
            # find the opinion now
            # new_opinion = self.individuals[np.random.choice(range(len(self.individuals)))].opinion
            new_opinion = self.individuals[random_index].opinion
            if neighbour_index is not None:  # interact with random neighbour
                neighbour_opinion = self.individuals[neighbour_index].opinion
                if abs(new_opinion - neighbour_opinion) < self.threshold:
                    updated_opinions[random_index] = new_opinion + self.beta * (neighbour_opinion - new_opinion)
                    updated_opinions[neighbour_index] = neighbour_opinion + self.beta * (
                                new_opinion - neighbour_opinion)

        # apply the new opinions to individuals
        for i, opinion in enumerate(updated_opinions):
            if opinion is not None:
                self.individuals[i].opinion = opinion

    def collect_opinions(self):
        return [ind.opinion for ind in self.individuals]


class Node: #Representing a node in nerwork

    def __init__(self, value, index, color=1, connections=None):
        if connections is None:
            connections = [0] * 100  #Adding in a default value for the nodes
        self.value = value
        self.index = index
        self.color = color
        self.connections = connections


class Network: #Managing the networks of the nodes and creating networks

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes #Initialising with a list of nodes

    def get_mean_degree(self):
        # Your code  for task 3 goes here
        degree_list = []
        # finds the total number of degrees across all nodes
        for node in self.nodes:
            degree_list.append(sum(node.connections))
        degree_list = [sum(node.connections) for node in self.nodes]
        # divides total degrees by number of nodes for mean
        return sum(degree_list) / len(degree_list)

    def get_mean_clustering(self):
        # Your code for task 3 goes here
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
        return node_count - 1

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
        # Your code for task 3 goes here
        # list including distances between all nodes
        path_lenths_list = []
        total_shortest_paths = 0
        for start_node in self.nodes:
            total_shortest_paths += self.shortest_path_length(start_node)
            # adds shortest distance divided by number of connections to the list
            path_lenths_list.append(self.shortest_path_length(start_node) / self.get_num_connected_nodes(start_node))
        # divides total distance by number of nodes for mean
        return round(sum(path_lenths_list) / len(self.nodes), ndigits=15)

    def make_random_network(self, N, connection_probability=0.5): 
        '''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

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

    def make_ring_network(self, N, neighbour_range=1):  #Creating a ring network while having each node connected to its neighbour
        colors = cm.hot(np.linspace(0, 1, N))  #Color map for the nodes
        self.nodes = []
        for node_number in range(N):
            connections = [0] * N #creating symmetrical connections in the neighbour range
            for j in range(1, neighbour_range + 1):
                connections[(node_number - j) % N] = 1
                connections[(node_number + j) % N] = 1
            self.nodes.append(Node(0, node_number, colors[node_number], connections=connections))

    def make_small_world_network(self, N, re_wire_prob=0.2): #Generating a small world network from a ring network using a rewiring probability of 0.2
        colors = cm.hot(np.linspace(0, 1, N))  #Color map for the nodes
        self.make_ring_network(N, 1)
        for i, node in enumerate(self.nodes):
            node.color = colors[i]  #Assigning colors
            targets_to_consider = list(enumerate(node.connections))
            for idx, connected in targets_to_consider:
                if connected and random.random() < re_wire_prob:
                    node.connections[idx] = 0
                    possible_new_targets = [i for i in range(N) if not node.connections[i] and i != node.index]
                    if possible_new_targets:
                        new_target = random.choice(possible_new_targets)
                        node.connections[new_target] = 1

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        node_radius = 0.2 * network_radius  #Re-sizing the nodes to make them bigger
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

        for node in self.nodes:  #Using a for loop for color and size of the nodes
            node_angle = node.index * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), node_radius, color=node.color)
            ax.add_patch(circle)  #For making the nodes bigger

            for neighbour_index, connected in enumerate(node.connections):
                if connected:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)
                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')
        plt.show()

    def plot_network(self):
        for node in self.nodes:
            for x, connection in enumerate(node.connections):
                if connection == 1:
                    plt.plot([node.index, x], [node.value, self.nodes[x].value], color='black')
        plt.scatter([node.index for node in self.nodes], [node.value for node in self.nodes])
        plt.show()


def test_networks():
    # Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number - 1) % num_nodes] = 1
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert (network.get_mean_degree() == 2), network.get_mean_degree()
    assert (network.get_mean_clustering() == 0), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 2.777777777777778), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert (network.get_mean_degree() == 1), network.get_mean_degree()
    assert (network.get_mean_clustering() == 0), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 5), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert (network.get_mean_degree() == num_nodes - 1), network.get_mean_degree()
    assert (network.get_mean_clustering() == 1), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 1), network.get_mean_path_length()

    print("All tests passed")


'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''


def neighbours_opinions(population, i, j):
    """This function finds the 4 neighbours of the selected grid point ensuring wrapping of the grid. The function
	initialises a list and then stores the value of the opinions of the neighbours in this list.
	 Inputs: population (numpy array)
	         i (integer)
	         j (integer)
	 Returns:
	        opinions_list (list)
	"""
    # Retrieving the shape of the grid. Where k is the number of rows and m is the number of columns.
    no_rows, no_cols = population.shape
    # Initialises an empty list and finds the value of each neighbour and stores them in the new list.
    opinions_list = []
    opinions_list.append(population[(i - 1) %  no_rows, j])
    opinions_list.append(population[(i + 1) %  no_rows, j])
    opinions_list.append(population[i, (j + 1) % no_cols])
    opinions_list.append(population[i, (j - 1) % no_cols])
    return opinions_list


def calculate_agreement(population, row, col, external=0.0):
    '''
	This function should return the extent to which a cell agrees with its neighbours.
	Inputs: population (numpy array)
			row (int)
			col (int)
			external (float)
	Returns:
			change_in_agreement (float)
	'''
    # Finds value for selected grid point.
    person = population[row, col]
    # Collects opinions of neighbours and stores in a list.
    opinions = neighbours_opinions(population, row, col)
    # Each neighbour's opinion is multiplied by the value of the opinion for the person selected and all summed together.
    agreement = sum(person * o for o in opinions)
    # Value of external influence is multiplied by the value of selected person's opinion and added to the value of the agreement.
    agreement += external * person
    return agreement


def ising_step(population, external=0.0, alpha=1.0):
    '''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
            alpha (float) - how tolerant society is of those who disagree with their neighbours

	'''

    n_rows, n_cols = population.shape
    # Randomly selects grid point
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    # Calculates agreement for selected point
    agreement = calculate_agreement(population, row, col, external)
    # Flips the person's opinion if negative agreement.
    # If a non-zero value of alpha is given, the probability a flip occurs is calculated.
    # If the probability of this is greater than a randomly generated float between 0.0 and 1.0 a flip of opinion occurs.
    if agreement <= 0:
        population[row, col] *= -1
    else:
	    if alpha != 0:
		    random_prob = random.random()
		    if random_prob < math.e ** (-agreement / alpha):
			    population[row, col] *= -1


def plot_ising(im, population):
    '''
	This function will display a plot of the Ising model
	'''

    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)


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
    assert (calculate_agreement(population, 1, 1, 10) == -6), "Test 9"
    assert (calculate_agreement(population, 1, 1, -10) == 14), "Test 10"

    print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external, alpha)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''


def defuant_main(beta=0.2, threshold=0.2):
    timestep = 350
    bins_count = 15
    scatter_size = 20
    print(f"Running defuant model with beta={beta} and threshold={threshold}")

    society = Society(size=100, beta=beta, threshold=threshold)

    # record the opinion in each timestep
    opinions_over_time = []

    # update model
    for t in range(timestep):  # the timestep is 350
        society.update_opinions()  # update the opinions
        opinions_over_time.append(society.collect_opinions())  # collection the opinions
    final_opinions = society.collect_opinions()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # plot the histogram
    ax1.hist(final_opinions, bins=np.linspace(0, 1, bins_count))
    ax1.set_title('Opinion Distribution')
    ax1.set_xlabel('Opinion')
    ax1.set_ylabel('Frequency')

    # plot the scatter diagram
    for step, opinions in enumerate(opinions_over_time):
        ax2.scatter([step] * len(opinions), opinions, color='red', s=scatter_size)
    ax2.set_title('Opinion Changes Over Time')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Opinion')

    plt.tight_layout()
    plt.savefig('figure.png')
    plt.show()


def test_defuant():
    time_step_1 = 100
    time_step_2 = 500
    # Test 1 when beta = 1, threshold = 0.5
    society_1 = Society(size=100, beta=1, threshold=0.5)

    # record the opinion in each timestep
    opinions_over_time = []

    # update model
    for t in range(time_step_1):  # the timestep is 100
        society_1.update_opinions()  # update the opinions
        opinions_over_time.append(society_1.collect_opinions())  # collection the opinions
    large_count = 0
    for opinions in opinions_over_time:
        for opinion in opinions:
            if opinion > 0.8:
                large_count += 1
    assert large_count > 3, 'Test 1 fail'
    print('Test 1 success')
    # Test 2 when beta = 0.1, threshold = 1
    society_2 = Society(size=100, beta=0.1, threshold=1)
    for t in range(time_step_2):
        society_2.update_opinions()  # update model
    opinions_at_500 = society_2.collect_opinions()
    # check if all the opinions at 500 are less than 0.1
    all_large = all(opinion < 1 for opinion in opinions_at_500)
    assert all_large, 'Test 2 fail'
    print('Test 2 success')
# Task 5
# get the opinions from the neighbour
def neighbours_opinions_network(node, network):
    opinions_list = [network.nodes[i].value for i, connected in enumerate(node.connections) if connected == 1]
    return opinions_list
# update the node value follow the rule
def ising_step_network(network, external=0.0, alpha=1.0):
    node = random.choice(network.nodes)
    opinions = neighbours_opinions_network(node, network)
    agreement = sum(node.value * o for o in opinions) + external * node.value

    if agreement < 0 or (alpha and random.random() < math.e ** (-agreement / alpha)):
        node.value *= -1
def plot_network(network):
    # Initialising graphs and axes
    magnitude_of_node = 100
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # exist some nodes
    positions = {node.index: (random.random(), random.random()) for node in network.nodes}
    # positions = {node.index: (np.cos(2 * np.pi * node.index / len(network.nodes)),
    #                           np.sin(2 * np.pi * node.index / len(network.nodes)))
    #              for node in network.nodes}
    for node in network.nodes:
        for i, connected in enumerate(node.connections):
            if connected:
                ax.plot([positions[node.index][0], positions[i][0]],
                        [positions[node.index][1], positions[i][1]], 'k-', lw=1)
    colors = ['red' if node.value < 0 else 'blue' for node in network.nodes]
    scatter = ax.scatter([pos[0] for pos in positions.values()], [pos[1] for pos in positions.values()], c=colors, s = magnitude_of_node)
# update the values
    def update(frame):
        ising_step_network(network)
        colors = ['red' if node.value < 0 else 'blue' for node in network.nodes]
        scatter.set_color(colors)
        return scatter,

    # create animation
    ani = FuncAnimation(fig, update, frames=100, interval=200, blit=True)

    plt.show()

'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''


def main(): #Arguments that help generate the desired networks 
    # You should write some code for handling flags here
    parser = argparse.ArgumentParser()
    parser.add_argument("-ising_model", action='store_true')
    parser.add_argument("-alpha", type=float, default=1)
    parser.add_argument("-external", type=float, default=0)
    parser.add_argument("-test_ising", action='store_true')
    parser.add_argument("-network", type=int)
    parser.add_argument("-test_networks", action='store_true')
    parser.add_argument("-ring_network", type=int, help="Create a ring network with a specified number of nodes.")
    parser.add_argument("-small_world", type=int, help="Number of nodes in the small-world network.")
    parser.add_argument("-re_wire", "--rewiring_probability", type=float, default=0.2,
                        help="Rewiring probability (default: 0.2 for small-world network)")
    parser.add_argument("-r", "--range", type=int, default=2,
                        help="Connectivity range (default: 2 for small-world network)")
    parser.add_argument('-defuant', action='store_true', help='Defuant model with default parameters')
    parser.add_argument('-beta', type=float, default=0.2, help='Beta , default is 0.2')
    parser.add_argument('-threshold', type=float, default=0.2, help='Threshold, default is 0.2')
    parser.add_argument('-test_defuant', action='store_true', help='Run the test functions')
    parser.add_argument('-use_network', type=int)
    args = parser.parse_args()
    alpha = args.alpha
    external = args.external
    network_size = args.network
    # task 1
    if args.ising_model and not args.use_network:
        rows = 100
        columns = 100
        population = np.random.choice([1, -1], size=(rows, columns))
        ising_main(population, alpha, external)
    if args.test_ising:
        test_ising()
    if args.test_networks:
        test_networks()

    # task 3
    if args.network:
        network = Network()
        network.make_random_network(network_size)

        mean_degree = network.get_mean_degree()
        mean_clustering = network.get_mean_clustering()
        mean_path_length = network.get_mean_path_length()

        print("Mean Degree:", mean_degree)
        print("Mean Clustering Coefficient:", mean_clustering)
        print("Mean Path Length:", mean_path_length)
        network.plot_network()
    # task 4
    if args.small_world: #Checking if a small world network or ring network is desired and plotting
        n_nodes = args.small_world #Number for a small world network
        network = Network([])
        network.make_small_world_network(n_nodes)
        network.plot()
    if args.ring_network:
        n_nodes = args.ring_network #Number of nodes for a ring network
        network = Network([])
        network.make_ring_network(n_nodes)
        network.plot()
    # task 2
    if args.test_defuant:
        test_defuant()
    if args.defuant:
        defuant_main(beta=args.beta, threshold=args.threshold)
    # task 5
    if args.use_network and args.ising_model:
        network = Network()
        network.make_random_network(args.use_network)
        plot_network(network)


if __name__ == "__main__":
    main() #Running the script in the main program

