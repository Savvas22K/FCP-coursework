import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
# class an individual
class Individual:
    def __init__(self, opinion):
        self.opinion = opinion

# class a society
class Society:
    def __init__(self, size, beta, threshold):
        self.individuals = [Individual(random.uniform(0, 1)) for _ in range(size)]
        self.beta = beta
        self.threshold = threshold

    def update_opinions(self):  # creat a list to store the opinions updated
        updated_opinions = [None] * len(self.individuals)
        # find the index of neighbour
        for i in range(len(self.individuals)):
            left_index = (i - 1) if i > 0 else None
            right_index = (i + 1) if i < len(self.individuals) - 1 else None
            neighbour_index = np.random.choice([left_index,right_index])
            # find the opinion now
            new_opinion = self.individuals[i].opinion
            if neighbour_index is not None:  # interact with random neighbour
                neighbour_opinion = self.individuals[neighbour_index].opinion
                if abs(new_opinion - neighbour_opinion) < self.threshold:
                    updated_opinions[i] = new_opinion + self.beta * (neighbour_opinion - new_opinion)
                    updated_opinions[neighbour_index] = neighbour_opinion + self.beta * (new_opinion - neighbour_opinion)

        # apply the new opinions to individuals
        for i, opinion in enumerate(updated_opinions):
            if opinion is not None:
                self.individuals[i].opinion = opinion
    def collect_opinions(self):
        return [ind.opinion for ind in self.individuals]




def defuant_main(beta=0.2, threshold=0.2):
    print(f"Running defuant model with beta={beta} and threshold={threshold}")


    society = Society(size=100, beta=beta, threshold=threshold)

    # record the opinion in each timestep
    opinions_over_time = []

    # update model
    for t in range(350):  # the timestep is 350
          # for substep in range(5): # a substep is 5
            society.update_opinions()  # update the opinions
            opinions_over_time.append(society.collect_opinions())  # collection the opinions
    final_opinions = society.collect_opinions()


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # plot the histogram
    ax1.hist(final_opinions, bins=np.linspace(0, 1, 15))
    ax1.set_title('Opinion Distribution')
    ax1.set_xlabel('Opinion')
    ax1.set_ylabel('Frequency')

    # plot the scatter diagram
    for step, opinions in enumerate(opinions_over_time):
        ax2.scatter([step] * len(opinions), opinions, color='red', s=20)
    ax2.set_title('Opinion Changes Over Time')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Opinion')

    plt.tight_layout()
    plt.savefig('figure.png')
def test_defuant():
    # Test 1 when beta = 1, threshold = 0.5
    society_1 = Society(size=100, beta=1, threshold=0.5)

    # record the opinion in each timestep
    opinions_over_time = []

    # update model
    for t in range(100):  # the timestep is 100
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
    for t in range(500):
        society_2.update_opinions() # update model
    opinions_at_500 = society_2.collect_opinions()
    # check if all the opinions at 500 are less than 0.1
    all_large = all(opinion < 1 for opinion in opinions_at_500)
    assert all_large, 'Test 2 fail'
    print('Test 2 success')

def main():
    parser = argparse.ArgumentParser(description="Run the defuant model with various parameters.")
    parser.add_argument('-defuant', action='store_true', help='Defuant model with default parameters')
    parser.add_argument('-beta', type=float, default=0.2, help='Beta , default is 0.2')
    parser.add_argument('-threshold', type=float, default=0.2, help='Threshold, default is 0.2')
    parser.add_argument('-test_defuant', action='store_true', help='Run the test functions')

    args = parser.parse_args()

    if args.test_defuant:
         test_defuant()
    if args.defuant:
        defuant_main(beta=args.beta, threshold=args.threshold)
if __name__ == "__main__":
    main()