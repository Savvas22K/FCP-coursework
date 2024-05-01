import matplotlib.pyplot as plt
import numpy as np
import random
import math
import argparse


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
    """
    This function should return the *change* in agreement that would result if the cell at (row, col) was to flip
     its value
    Inputs: population (numpy array)
            row (int)
            col (int)
            external (float)
    Returns:
            change_in_agreement (float)
    """
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
    agreement = calculate_agreement(population, row, col, external=None)
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
    """
    This function will display a plot of the Ising model
    """

    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)


def test_ising():
    """
    This function will test the calculate_agreement function in the Ising model
    """

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


def main():
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


if __name__ == "__main__":
    main()
