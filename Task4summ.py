import argparse
import networkx as nx
import matplotlib.pyplot as plt
import random

def create_ring_network(n, r):
    network = {}
    for i in range(n):
        connected_nodes = set()
        for j in range(1, r+1):
            connected_nodes.add((i - j) % n)
            connected_nodes.add((i + j) % n)
        network[i] = connected_nodes
    return network

def create_small_world_network(n, r, p):
    network = create_ring_network(n, r)
    for node in network:
        for target in list(network[node]):
            if random.random() < p:
                network[node].remove(target)
                possible_new_targets = set(range(n)) - network[node] - {node}
                if possible_new_targets:
                    new_target = random.choice(list(possible_new_targets))
                    network[node].add(new_target)
    return network

def plot_network(network, title):
    G = nx.Graph()
    for node, edges in network.items():
        for edge in edges:
            G.add_edge(node, edge)
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue')
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
        network = create_ring_network(n_nodes, connectivity_range)
        plot_network(network, f"Ring Network (N={n_nodes}, Range={connectivity_range})")
    elif args.small_world:
        n_nodes = args.small_world
        connectivity_range = args.range  
        rewiring_probability = args.rewiring_probability  
        network = create_small_world_network(n_nodes, connectivity_range, rewiring_probability)
        plot_network(network, f"Small-World Network (N={n_nodes}, Range={connectivity_range}, Rewiring Probability={rewiring_probability})")

if __name__ == "__main__":
    main()






