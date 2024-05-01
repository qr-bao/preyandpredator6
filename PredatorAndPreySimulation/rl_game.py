import pygame, sys
import predator
import prey
import food
import random
import constants
from pygame.locals import *
import numpy as np
import pandas as pd
class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Predator:
    def __init__(self, position):
        self.position = position
        self.detectionOfPrey = None  
        self.health = self.health
class Prey:
    def __init__(self, position):
        self.position = position
        self.health = self.health
class Simulator:
    def __init__(self, initialPredatorPopulation, initialPreyPopulation):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.surface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
        self.predators = []
        self.prey = []
        self.food = []
        for i in range(initialPredatorPopulation):
            p = predator.Predator(
                (
                    predator.MAX_HEALTH, 
                    predator.VIEW_RADIUS,
                    predator.MAX_VELOCITY,
                    (random.randint(0, WIDTH), random.randint(0, HEIGHT)),
                    (random.uniform(-1, 1) * INIT_VELOCITY, random.uniform(-1, 1) * INIT_VELOCITY),
                    predator.COLOR,
                    predator.SIZE,
                ),
                predator.MAX_DETECTION_OF_PREY  * random.uniform(-1, 1),
                predator.MAX_ATTRACTION_TO_PREY * random.uniform(-1, 1),    
            )
            self.predators.append(p)

        for i in range(initialPreyPopulation):
            p = prey.Prey(
                (
                    prey.MAX_HEALTH, 
                    prey.VIEW_RADIUS,
                    prey.MAX_VELOCITY,
                    (random.randint(0, WIDTH), random.randint(0, HEIGHT)),
                    (random.uniform(-1, 1) * INIT_VELOCITY, random.uniform(-1, 1) * INIT_VELOCITY),
                    prey.COLOR,
                    prey.SIZE,
                ),
                prey.MAX_DETECTION_OF_FOOD       * random.uniform(-1, 1),
                prey.MAX_ATTRACTION_TO_FOOD      * random.uniform(-1, 1),
                prey.MAX_DETECTION_OF_PREDATOR   * random.uniform(-1, 1),
                prey.MAX_REPULSION_FROM_PREDATOR * random.uniform(-1, 1),
            )
            self.prey.append(p)
        def get_prey_and_predator(self):
            return self.prey, self.predator

class Environment:
    def __init__(self, size, predator, prey):
        simulator = Simulator()
        prey, predator = simulator.get_prey_and_predator()
        self.size = size
        self.predator = predator
        self.prey = prey

    def step(self, predator_action, prey_action):
        self.predator.position.x += predator_action[0]
        self.predator.position.y += predator_action[1]
        self.prey.position.x += prey_action[0]
        self.prey.position.y += prey_action[1]

        self.predator.position.x = max(0, min(self.size - 1, self.predator.position.x))
        self.predator.position.y = max(0, min(self.size - 1, self.predator.position.y))
        self.prey.position.x = max(0, min(self.size - 1, self.prey.position.x))
        self.prey.position.y = max(0, min(self.size - 1, self.prey.position.y))

        reward = self.calculate_reward()

        done = self.is_done()

        return reward, done   
    def step(self, predator_action, prey_action):
        self.predator.position.x += predator_action[0]
        self.predator.position.y += predator_action[1]
        self.prey.position.x += prey_action[0]
        self.prey.position.y += prey_action[1]

        self.predator.position.x = max(0, min(self.size - 1, self.predator.position.x))
        self.predator.position.y = max(0, min(self.size - 1, self.predator.position.y))
        self.prey.position.x = max(0, min(self.size - 1, self.prey.position.x))
        self.prey.position.y = max(0, min(self.size - 1, self.prey.position.y))

        reward = self.calculate_reward()

        done = self.is_done()

        return reward, done

    def calculate_reward(self):
        if self.predator.position.x == self.prey.position.x and self.predator.position.y == self.prey.position.y:
            self.predator.health += 100  # Increase predator's health
            self.prey.health -= 100 
            return 100  # Predator caught the prey
        else:
            return -1  # Predator did not catch the prey


    def is_done(self):
        if self.prey.health <= 0 :
            return True  # Prey's health is zero or below, game is done
        return False  # Game is not done

class RLAgent:
    def __init__(self, environment):
        self.environment = environment
        self.q_table = self.initialize_q_table()

    def initialize_q_table(self):
        # Initialize Q-table with zeros
        self.q_table = {}
        for predator_x in range(self.size):
            for predator_y in range(self.size):
                for prey_x in range(self.size):
                    for prey_y in range(self.size):
                        for action in self.predator_actions:
                            self.q_table[((predator_x, predator_y), (prey_x, prey_y), action)] = 0.0
        return 0

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.actions, key=lambda action: self.q_table[(state, action)])
        
    def update_q_table(self, state, action, reward, new_state):
        # Get the maximum Q-value for the new state
        max_new_state_q_value = max(self.q_table[(new_state, a)] for a in self.actions)

        # Update the Q-value for the current state and action
        self.q_table[(state, action)] = (1 - self.alpha) * self.q_table[(state, action)] + self.alpha * (reward + self.gamma * max_new_state_q_value)

    def train(self, episodes):
        for episode in range(episodes):
            state = self.environment.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                reward, done = self.environment.step(action)
                next_state = self.environment.get_state()
                self.update_q_table(state, action, reward, next_state)
                state = next_state

def main():

    environment = Environment()  
    agent = RLAgent(environment)
    agent.train(1000)  

if __name__ == "__main__":
    main()                

