import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
    path_lenths_list = []
    total_shortest_paths = 0
    for start_node in self.nodes:
        total_shortest_paths += self.shortest_path_length(start_node)
        # adds shortest distance divided by number of connections to the list
        path_lenths_list.append(self.shortest_path_length(start_node) / self.get_num_connected_nodes(start_node))
    # divides total distance by number of nodes for mean
    return sum(path_lenths_list) / len(self.nodes)
	def get_mean_clustering(self):
		#Your code for task 3 goes here

	def get_mean_path_length(self):
		#Your code for task 3 goes here

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
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

	def make_ring_network(self, N, neighbour_range=1):
		#Your code  for task 4 goes here

	def make_small_world_network(self, N, re_wire_prob=0.2):
		#Your code for task 4 goes here

	def plot(self):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
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

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

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
    k, m = population.shape
    # Initialises an empty list and finds the value of each neighbour and stores them in the new list.
    opinions_list = []
    opinions_list.append(population[(i - 1) % m, j])
    opinions_list.append(population[(i + 1) % m, j])
    opinions_list.append(population[i, (j + 1) % k])
    opinions_list.append(population[i, (j - 1) % k])
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
    agreement = 0
    agreement = sum(person * o for o in opinions)
    # Value of external influence is multiplied by the value of selected person's opinion and added to the value of the agreement.
    agreement += external * person
    return agreement


def ising_step(population, external=0.0, alpha=1.0):
    """
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
            external (float) - optional - the magnitude of any external "pull" on opinion
            alpha (float) - how tolerant society is of those who disagree with their neighbours
    """

    n_rows, n_cols = population.shape
    # Randomly selects grid point
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    # Calculates agreement for selected point
    agreement = calculate_agreement(population, row, col, external=0.0)
    # Flips the person's opinion if negative agreement.
    # If a non-zero value of alpha is given, the probability a flip occurs is calculated.
    # If the probability of this is greater than a randomly generated float between 0.0 and 1.0 a flip of opinion occurs.
    if agreement < 0:
        population[row, col] *= -1
    elif alpha:
        random_prob = random.random()
        if random_prob < math.e ** (-agreement/alpha):
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
    assert(calculate_agreement(population,1,1)==4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(population,1,1)==-4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(population,1,1)==-2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(population,1,1)==0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(population,1,1)==2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(population,1,1)==4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1,1)==3), "Test 7"
    assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
    assert(calculate_agreement(population,1,1,10)==-6), "Test 9"
    assert(calculate_agreement(population,1,1, -10)==14), "Test 10"

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
            ising_step(population, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

def defuant_main():
	#Your code for task 2 goes here

def test_defuant():
	#Your code for task 2 goes here


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
	#You should write some code for handling flags here
    parser = argparse.ArgumentParser()
    parser.add_argument("-ising_model", action='store_true')
    parser.add_argument("-alpha", type=float, default=1)
    parser.add_argument("-external", type=float, default=0)
    parser.add_argument("-test_ising", action='store_true')

    args = parser.parse_args()
    alpha = args.alpha
    external = args.external

    if args.ising_model:
        population = np.random.choice([1, -1], size=(100,100))
        ising_main(population, alpha, external)
    if args.test_ising:
        test_ising()
	    parser = argparse.ArgumentParser()


	parser.add_argument("-network", type=int)
	parser.add_argument("-test_network", action='store_true')
	
	args = parser.parse_args()
	network_size = args.network

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
	    print("test")
	
	if args.network:
	    network = Network()
	    network.make_random_network(network_size)
	
	    mean_degree = network.get_mean_degree()
	    mean_clustering = network.get_mean_clustering()
	    mean_path_length = network.get_mean_path_length()
	
	    print("Mean Degree:", mean_degree)
	    print("Mean Clustering Coefficient:", mean_clustering)
	    print("Mean Path Length:", mean_path_length)
	    network.plot()

if __name__=="__main__":
	main()
