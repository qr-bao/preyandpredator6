import matplotlib.pyplot as plt
from simulator import Simulator  # Import the Simulator class from your simulator.py file

def main():
    # Create a Simulator instance
    simulator = Simulator()

    # Run the simulation for a certain number of steps
    for _ in range(100):  # Replace 100 with the number of steps you want
        simulator.numberOfcreature()  # Assume you have a step method to advance the simulation

    # Create a list of time steps
    time_steps = list(range(len(simulator.numberOfcreature.prey)))

    # Plot the counts over time
    plt.plot(time_steps, simulator.predator_counts, label='Predators')
    plt.plot(time_steps, simulator.prey_counts, label='Prey')
    plt.plot(time_steps, simulator.food_counts, label='Food')

    # Add labels and a legend
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()