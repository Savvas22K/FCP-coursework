import matplotlib.pyplot as plt
import numpy as np
import random
import math
import argparse



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


def ising_step(population, external=0.0, alpha = 1.0):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
            external (float) - optional - the magnitude of any external "pull" on opinion
    '''

    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, external=0.0)

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