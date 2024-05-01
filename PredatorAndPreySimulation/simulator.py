import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
from matplotlib.animation import FuncAnimation
from pygame.locals import *

import constants
import food
import predator
import prey

WIDTH  = constants.WIDTH
HEIGHT = constants.HEIGHT
MAX_FOOD_SUPPLY = constants.MAX_FOOD_SUPPLY
INIT_VELOCITY = constants.INIT_VELOCITY
FPS = constants.FPS
CHANCE_OF_ESCAPE = constants.CHANCE_OF_ESCAPE
initialPredatorPopulation = constants.initialPredatorPopulation
initialPreyPopulation = constants.initialPreyPopulation

PREY_REPRODUCTION_CHANCE = constants.PREY_REPRODUCTION_CHANCE
PREDATOR_REPRODUCTION_CHANCE = constants.PREDATOR_REPRODUCTION_CHANCE
MUTATION_CHANCE = constants.MUTATION_CHANCE

Populations = []
lengthofprey  = []
lengthofpredator = []
lengthoffood = []
class Environment:
    def __init__(self, initialPredatorPopulation, initialPreyPopulation):
            # pygame.init()

        self.predator = []
        self.prey = []
        self.food = []
        self.surface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
        self.obstacles =[]
        #first predator is import from predator.py,then second is class Predator
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
            self.predator.append(p)

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
        for i in range(MAX_FOOD_SUPPLY):
            f = food.Food(
                (random.randint(0, WIDTH), random.randint(0, HEIGHT)),
                food.COLOR,
                food.SIZE
            )
            self.food.append(f)
        pygame.display.set_caption('SIMULATOR')
    def env_map(self):
        num_obstacles = 20
        min_obstacle_size, max_obstacle_size = 20, 50

        for _ in range(num_obstacles):
            # 随机生成障碍物的位置和大小
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            width = random.randint(min_obstacle_size, max_obstacle_size)
            height = random.randint(min_obstacle_size, max_obstacle_size)  
            obstacle =(x, y, width, height)
            self.obstacles.append(obstacle)
    def envrionment_state(self):
        #store the information of prey,food and predator location
        #store the turrians information 
        self.turrians = np.ones((WIDTH, HEIGHT))
        self.envrionment_prey = np.zeros((WIDTH, HEIGHT))
        self.envrionment_predator = np.zeros((WIDTH, HEIGHT))
        self.envrionment_food = np.zeros((WIDTH, HEIGHT))
        for i in range(len(self.prey)):
            self.envrionment_prey[self.prey[i].position[0],self.prey[i].position[1]] = 1
        for i in range(len(self.predators)):
            self.envrionment_predator[self.predators[i].position[0],self.predators[i].position[1]] = 1
        for i in range(len(self.food)):
            self.envrionment_food[self.food[i].position[0],self.food[i].position[1]] = 1
    def envrionment_reward(self):
        pass

class Simulator:
    def __init__(self,env_prey,env_predator,env_food,env_surface,env_obstacles):
        pygame.init()

        # cap = cv2.VideoCapture('/home/wicky/Desktop/bcpandascute/code_project/test/test4/PredatorAndPreySimulation/video.mp4')
        # success, img = cap.read()
        # shape = [1024, 768]
        # wn = pygame.display.set_mode(shape)
        # clock = pygame.time.Clock()

        

        # self.clock = pygame.time.Clock()
        # self.surface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
        #color = (255, 255, 255)
        #self.surface.fill(color)
        self.predators = env_predator
        self.prey = env_prey
        self.food = env_food
        self.clock = pygame.time.Clock()
        self.surface = env_surface
        self.obstacles = env_obstacles
        pygame.display.set_caption('SIMULATOR')
    def drawModels(self):
        color = (255, 255, 255)
        self.surface.fill(color)
        for i in range(len(self.obstacles)):
            pygame.draw.rect(self.surface, (0, 0, 0), self.obstacles[i])
        for p in self.predators:
            # The code `draw` is not a valid Python command or function. It seems like there is a
            # comment `# Python` followed by `draw` and `
            p.draw(self.surface)
        for p in self.prey:
            p.draw(self.surface)
        for p in self.food:
            p.draw(self.surface)



    def moveModels(self):
        self.surface.fill((0, 0, 0))
        for i in range(len(self.predators)):
            self.predators[i].move(WIDTH, HEIGHT, self.prey,self.predators, self.food)
        for i in range(len(self.prey)):
            self.prey[i].move(WIDTH, HEIGHT, self.predators, self.food,self.prey)
    def update(self):
        pygame.display.update()
        self.clock.tick(FPS)

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.kill()
            
            # add keys for config settings here
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.kill()

    def preyHunt(self):
        for j in range(len(self.prey)):
            p = self.prey[j]
            for i in range(len(self.food)-1, -1, -1):
                foodRect = pygame.Rect(0, 0, 2*self.food[i].size, 2*self.food[i].size)
                foodRect.centerx = self.food[i].x
                foodRect.centery = self.food[i].y
                if p.rect.colliderect(foodRect):
                    del self.food[i]
                    self.prey[j].health += prey.HEALTH_GAIN
                    self.prey[j].health = min(self.prey[j].health, self.prey[j].maxHealth)
    
    def predatorHunt(self):
        for j in range(len(self.predators)):
            p = self.predators[j]
            for i in range(len(self.prey)-1, -1, -1):
                if p.rect.colliderect(self.prey[i].rect):
                    """chance = random.random()
                    if chance > CHANCE_OF_ESCAPE:
                        self.prey[i].dead()
                        del self.prey[i]
                        self.predators[j].health += predator.HEALTH_GAIN
                        self.predators[j].health = min(self.predators[j].health, self.predators[j].maxHealth)
                    else:
                        self.prey[i].health -= prey.HEALTH_LOSS / 2
                        self.predators[j].health += predator.HEALTH_GAIN / 2
                        self.predators[j].health = min(self.predators[j].health, self.predators[j].maxHealth)"""
                    self.prey[i].dead()
                    del self.prey[i]
                    self.predators[j].health += predator.HEALTH_GAIN
                    self.predators[j].health = min(self.predators[j].health, self.predators[j].maxHealth)
    
    def decreaseHealth(self):
        for i in range(len(self.predators)):
            self.predators[i].health -= predator.HEALTH_LOSS
            if self.predators[i].health <= 0:
                self.predators[i].dead()
        for i in range(len(self.prey)):
            self.prey[i].health -= prey.HEALTH_LOSS
            if self.prey[i].health <= 0:
                self.prey[i].dead()
    
    def removeDead(self):
        for i in range(len(self.predators)-1, -1, -1):
            if (self.predators[i].alive == False):
                del self.predators[i]
        for i in range(len(self.prey)-1, -1, -1):
            if (self.prey[i].alive == False):
                del self.prey[i]

    def addFood(self):
        while len(self.food) < MAX_FOOD_SUPPLY:
            f = food.Food(
                (random.randint(0, WIDTH), random.randint(0, HEIGHT)),
                food.COLOR,
                food.SIZE
            )
            self.food.append(f)

    def breedPrey(self):
        preyHealth = []
        for p in self.prey:
            preyHealth.append(p.health)
        if len(self.prey) >1:
            parent1, parent2 = random.choices(self.prey, weights = preyHealth, k = 2)
            child = parent1.crossbreed(parent2)
            if(random.random() < MUTATION_CHANCE):
                child.mutate()
            self.prey.append(child)
    
    def breedPredator(self):
        predHealth = []
        if self.predators != []:
            for p in self.predators:
                predHealth.append(p.health)
            if len(self.predators) >1:
                parent1, parent2 = random.choices(self.predators, weights = predHealth, k = 2)
                child = parent1.crossbreed(parent2)
                if(random.random() < MUTATION_CHANCE):
                    child.mutate()
                self.predators.append(child)

    def applyGeneticAlgorithm(self):
        if random.random() < PREY_REPRODUCTION_CHANCE:
            self.breedPrey()
        if random.random() < PREDATOR_REPRODUCTION_CHANCE:
            self.breedPredator()

    def kill(self):
        pygame.quit()
        sys.exit()
env = Environment(initialPredatorPopulation, initialPreyPopulation)
env.env_map()
env_prey = env.prey
env_predator = env.predator
env_food = env.food
env_surface = env.surface
env_obstacles = env.obstacles
sim = Simulator(env_prey,env_predator,env_food,env_surface,env_obstacles)
lengthofpredator = []
lengthofprey = []
lengthoffood = []
def draw_dynamic_chart(prey_counts, predator_counts):
    plt.ion()  # Turn on interactive mode

    for i in range(len(prey_counts)):
        plt.clf()  # Clear the current figure
        plt.plot(prey_counts[:i+1], label='Prey')
        plt.plot(predator_counts[:i+1], label='Predator')
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.legend()  # Add a legend
        plt.pause(0.1)  # Pause for a short period of time

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the final plot
def draw_chart(prey_counts, predator_counts):
    plt.ion()  # Turn on interactive mode
    plt.clf()  # Clear the current figure
    plt.plot(prey_counts, label='Prey')
    plt.plot(predator_counts, label='Predator')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.legend()  # Add a legend
    plt.pause(10000)  # Pause for a short period of time    



for i in range(1000000):
    try:
        #updateData(Populations)
        sim.checkEvents()
        sim.removeDead()
        sim.applyGeneticAlgorithm()
        sim.addFood()
        sim.moveModels()
        sim.drawModels()
        sim.preyHunt()
        sim.predatorHunt()
        sim.decreaseHealth()
        lengthofprey.append(len(sim.prey))
        lengthofpredator.append(len(sim.predators))
        lengthoffood.append(len(sim.food))
        #sim.draw_dynamic_chart()
        #sim.graph()
        #sim.print()
        sim.update()
    except UnboundLocalError:
        if True:
            lengthofprey.append(len(sim.prey))
            lengthofpredator.append(len(sim.predators))
            lengthoffood.append(len(sim.food))
            break
#draw_dynamic_chart(lengthofprey, lengthofpredator)
draw_chart(lengthofprey, lengthofpredator)
