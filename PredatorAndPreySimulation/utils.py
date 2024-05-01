import numpy as np
import pygame
import math
import predator
import creature
import constants
# Position               => a tuple of 2D coordinates
# ListOfCounterCreatures => The list which is to be used to get creatures in the field of view
# Upperbound             => the field of view


def PreyFilterUsingEuclideanDistances1(Vector,Position, ListOfprey,ListOfPredators,ListOfFoods, Upperbound):
    response = []
    predatorDistancePrey = []
    predatorDistanceFood = []
    predatorDistancePredator = []

    
    #if isinstance(ListOfprey, list):
    for animal in ListOfprey:
        x,y = animal.rect.centerx, animal.rect.centery
        verctor1 = (x-Position[0],y-Position[1])
        verctor2 = Vector
        distance = ((x-Position[0])**2 + (y-Position[1])**2)**(0.5)
        predatorDistancePrey.append((distance,(abs(x-Position[0]),abs(y-Position[1]))))
        theta_rad, theta_deg,position = calculate_angle_between_vectors(verctor1, verctor2)
        if distance < Upperbound and theta_deg <= constants.PredatorsVIEW_RANGE:
            if distance != 0:
                response.append((animal,(1/distance)))
            else:
                response.append((animal, float('inf')))
    for predator in ListOfPredators:
        x,y = predator.rect.centerx, predator.rect.centery
        distance = ((x-Position[0])**2 + (y-Position[1])**2)**(0.5)
        if distance < Upperbound and theta_deg <= constants.PredatorsVIEW_RANGE:
            if distance != 0:
                predatorDistancePredator.append((distance,(abs(x-Position[0]),abs(y-Position[1]))))
            else:
                predatorDistancePredator.append((distance,(float('inf'),float('inf'))))
        #predatorDistancePredator.append((distance,(abs(x-Position[0]),abs(y-Position[1]))))
    for food in ListOfFoods:
        x,y = food.x, food.y
        distance = ((x-Position[0])**2 + (y-Position[1])**2)**(0.5)
        if distance < Upperbound and theta_deg <= constants.PredatorsVIEW_RANGE:
            if distance != 0:
                predatorDistanceFood.append((distance,(abs(x-Position[0]),abs(y-Position[1]))))
            else:
                predatorDistanceFood.append((distance,(float('inf'),float('inf'))))
        #predatorDistanceFood.append((distance,(abs(x-Position[0]),abs(y-Position[1]))))

    return response,predatorDistancePrey,predatorDistancePredator,predatorDistanceFood

def PredictPredatorDirection(Position, CreaturesAround, behaviourRate,HearningL,HearningR):
    if len(CreaturesAround)==0:
        return pygame.math.Vector2(0, 0)

    desiredVelocitylist = []

    for details in CreaturesAround:
        animal, factor = details
        desiredVelocitylist.append([behaviourRate*factor*(animal.rect.x - Position[0]) , behaviourRate*factor*(animal.rect.y - Position[1])])
    for unknowns in HearningL:
        distance,distanceX,distanceY = unknowns
        if distance != 0:
            factor = 1/(distance+0.001)
            desiredVelocitylist.append([behaviourRate*factor*(distanceX) , behaviourRate*factor*(distanceY)])
    for unknowns in HearningR:
        distance,distanceX,distanceY = unknowns
        if distance != 0:
            factor =  1/(distance+0.001)
            desiredVelocitylist.append([behaviourRate*factor*(distanceX) , behaviourRate*factor*(distanceY)])
    desiredVelocities = np.array(desiredVelocitylist)
    resultant = np.sum(desiredVelocities, axis = 0)
    resultantVector2 = pygame.math.Vector2(resultant[0], resultant[1])
    
    target = CreaturesAround[0][0]
    maxFactor = CreaturesAround[0][1]
    for vel, details in zip(desiredVelocities, CreaturesAround):
        animal, factor = details
        currVector = pygame.math.Vector2(vel[0], vel[1])
        if abs(resultantVector2.angle_to(currVector)) <= 45 and maxFactor < factor:
            maxFactor = factor
            target = animal   

    desired = (pygame.math.Vector2(target.rect.x, target.rect.y) - pygame.math.Vector2(Position))*behaviourRate
    return desired
def Hearning(Position, ListOfCounterCreatures, ListOfFood, preys,Upperbound):
    HearningL = []
    HearningR = []
    earL_postions = [Position[0]-1,Position[1]]
    earR_postions = [Position[0]+1,Position[1]]

    for creature in ListOfCounterCreatures:
        x,y = creature.rect.centerx, creature.rect.centery
        distanceEarL = ((x-earL_postions[0])**2 + (y-earL_postions[1])**2)**(0.5)
        creatureDistanceOfEarL = [abs(x-earL_postions[0]),abs(y-earL_postions[1])]
        if distanceEarL < Upperbound :
            HearningL.append((distanceEarL,creatureDistanceOfEarL[0],creatureDistanceOfEarL[1]))
        distanceEarR = ((x-earR_postions[0])**2 + (y-earR_postions[1])**2)**(0.5)
        creatureDistanceOfEarR = (abs(x-earR_postions[0]),abs(y-earR_postions[1]))
        if distanceEarR < Upperbound :
            HearningR.append((distanceEarR,creatureDistanceOfEarR[0],creatureDistanceOfEarR[1]))
    for prey in preys:
        x,y = prey.rect.centerx, prey.rect.centery
        distanceEarL = ((x-earL_postions[0])**2 + (y-earL_postions[1])**2)**(0.5)
        preyDistanceOfEarL = (abs(x-earL_postions[0]),abs(y-earL_postions[1]))
        if distanceEarL < Upperbound :
            HearningL.append((distanceEarL,preyDistanceOfEarL[0],preyDistanceOfEarL[1]))
        distanceEarR = ((x-earR_postions[0])**2 + (y-earR_postions[1])**2)**(0.5)
        preyDistanceOfEarR = (abs(x-earR_postions[0]),abs(y-earR_postions[1]))
        if distanceEarR < Upperbound :
            HearningR.append((distanceEarR,preyDistanceOfEarR[0],preyDistanceOfEarR[1]))
    # for food in ListOfFood:
    #     x, y = food.x, food.y
    #     distanceEarL = ((x-earL_postions[0])**2 + (y-earL_postions[1])**2)**(0.5)
    #     foodDistanceOfEarL = (abs(x-earL_postions[0]),abs(y-earL_postions[1]))
    #     if distanceEarL < Upperbound :
    #         HearningL.append((distanceEarL,foodDistanceOfEarL[0],foodDistanceOfEarL[1]))
    #     distanceEarR = ((x-earR_postions[0])**2 + (y-earR_postions[1])**2)**(0.5)
    #     foodDistanceOfEarR = (abs(x-earR_postions[0]),abs(y-earR_postions[1]))
    #     if distanceEarL < Upperbound :
    #         HearningR.append((distanceEarR,foodDistanceOfEarR[0],foodDistanceOfEarR[1]))
    HearningL_sorted = sorted(HearningL, key=lambda x: x[0])
    HearningR_sorted = sorted(HearningR, key=lambda x: x[0])
    return HearningL_sorted[:5],HearningR_sorted[:5]
    # HearningL =[distance ,distanceX,distanceY]
    # HearningR =[distance ,distanceX,distanceY]
def FoodAndPredatorFilterUsingEuclideanDistances(vector,Position, ListOfCounterCreatures, ListOfFood, ListOfPrey,Upperbound):
    preds = []
    foods = []
    predatorDistance = []
    preyDistance = []
    FoodDistance = []
    for prey in ListOfPrey:
        x,y = prey.rect.centerx, prey.rect.centery
        distance = ((x-Position[0])**2 + (y-Position[1])**2)**(0.5)
        theta_rad, theta_deg,position = calculate_angle_between_vectors((x-Position[0],y-Position[1]), vector)
        if distance < Upperbound and theta_deg <= constants.PreysVIEW_RANGE:
            if distance != 0:
                preyDistance.append((distance,abs(x-Position[0]),abs(x-Position[0])))
            else:
                preyDistance.append((distance,float('inf'),float('inf')))
        #preyDistance.append((distance,abs(x-Position[0]),abs(x-Position[0])))
    for creature in ListOfCounterCreatures:
        x,y = creature.rect.centerx, creature.rect.centery
        distance = ((x-Position[0])**2 + (y-Position[1])**2)**(0.5)
        predatorDistance.append((distance,abs(x-Position[0]),abs(x-Position[0])))
        verctorOfPreyAndPredator = (x-Position[0],y-Position[1])
        velocityDerection = vector
        #position = 0 means the prey is on the left back of the predator
        #position = 1 means the prey is on the right back of the predator
        theta_rad, theta_deg,position = calculate_angle_between_vectors(verctorOfPreyAndPredator, velocityDerection)
        if distance < Upperbound and theta_deg <= constants.PreysVIEW_RANGE:
            if distance != 0 :
                preds.append((creature, (1/distance),x,y))
            else:
                preds.append((creature, float('inf'),float('inf'),float('inf')))
    
    for food in ListOfFood:
        x, y = food.x, food.y
        distance = ((x-Position[0])**2 + (y-Position[1])**2)**(0.5)
        verctorOfPreyAndFood = (x-Position[0],y-Position[1])
        FoodDistance.append((distance,abs(x-Position[0]),abs(x-Position[0])))
        velocityDerection = vector
        theta_rad, theta_deg,position = calculate_angle_between_vectors(verctorOfPreyAndPredator, velocityDerection)
        if distance < Upperbound and theta_deg <= constants.PreysVIEW_RANGE:
            if distance != 0:
                foods.append((food,(1/distance),x,y))
            else:
                foods.append((food, float('inf'),float('inf'),float('inf')))
    return foods, preds,predatorDistance,preyDistance,FoodDistance

def PredictPreyDirection(Position, CreaturesAround, FoodAround, foodBehaviour, creatureBehaviour,HearningL,HearningR):
    if len(CreaturesAround) == 0 and len(FoodAround) == 0 :
        return pygame.math.Vector2(0, 0)

    desiredVelocityCreaturelist = [[0,0]]
    for details in CreaturesAround:
        animal, factor,animalDistanceOfX,animalDistanceOfY = details

        desiredVelocityCreaturelist.append([creatureBehaviour*factor*(animalDistanceOfX) , creatureBehaviour*factor*(animalDistanceOfY)])
    for unknowns in HearningL:
        distance,distanceX,distanceY = unknowns
        if distance != 0:
            factor = 1/(distance+0.001)
            desiredVelocityCreaturelist.append([-1*creatureBehaviour*factor*(distanceX) , -1*creatureBehaviour*factor*(distanceY)])
    for unknowns in HearningR:
        distance,distanceX,distanceY = unknowns
        if distance != 0:
            factor = 1/(distance+0.001)
            desiredVelocityCreaturelist.append([-1*creatureBehaviour*factor*(distanceX) , -1*creatureBehaviour*factor*(distanceY)])
    desiredVelocityFoodlist = [[0, 0]]
    for details in FoodAround:
        food, factor,foodDistanceOfX,foodDistanceOfY= details
        desiredVelocityFoodlist.append([foodBehaviour*factor*(foodDistanceOfX), foodBehaviour*factor*(foodDistanceOfY)])

    desiredVelocitiesCreature = np.array(desiredVelocityCreaturelist)
    desiredVelocitiesFood = np.array(desiredVelocityFoodlist)
    
    resultant1 = np.sum(desiredVelocitiesCreature, axis = 0)
    resultant2 = np.sum(desiredVelocitiesFood, axis = 0)

    resultantVector2 = pygame.math.Vector2(resultant1[0] + resultant2[0], resultant1[1] + resultant2[1])

    if(len(CreaturesAround) != 0):
        target = CreaturesAround[0][0]
        maxFactor = CreaturesAround[0][1]
    else:
        target = FoodAround[0][0]
        maxFactor = FoodAround[0][1]

    for vel, details in zip(desiredVelocitiesCreature, CreaturesAround):
        animal, factor,distanceOfX,distanceOfY = details
        currVector = pygame.math.Vector2(vel[0], vel[1])
        if abs(resultantVector2.angle_to(currVector)) <= 45 and maxFactor < factor:
            maxFactor = factor
            target = animal
    
    for vel, details in zip(desiredVelocitiesFood, FoodAround):
        food, factor,distanceOfX,distanceOfY = details
        currVector = pygame.math.Vector2(vel[0], vel[1])
        if abs(resultantVector2.angle_to(currVector)) <= 45 and maxFactor < factor:
            maxFactor = factor
            target = food

    if type(target) == predator.Predator:
        return (pygame.math.Vector2(target.rect.x, target.rect.y) - pygame.math.Vector2(Position))*creatureBehaviour
    
    else:
        return (pygame.math.Vector2(target.x, target.y) - pygame.math.Vector2(Position))*foodBehaviour
"""def calculate_angle_between_vectors(vector1, vector2):

    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    cosine_theta = dot_product / (magnitude1 * magnitude2)
    cosine_theta = max(-1, min(1, cosine_theta))
    theta_rad = math.acos(cosine_theta)
    theta_deg = math.degrees(theta_rad)
    

    return theta_rad, theta_deg"""
import pygame
import math

def calculate_angle_between_vectors(vector1, vector2):
    # Convert lists to Vector2 objects
    v1 = pygame.math.Vector2(vector1)
    v2 = pygame.math.Vector2(vector2)

    # Calculate the dot product
    dot_product = v1.dot(v2)

    # Calculate the magnitudes of the vectors
    magnitude_v1 = v1.length()
    magnitude_v2 = v2.length()
    if magnitude_v1 * magnitude_v2 == 0:
        cos_theta = 1
    else:
        cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    # Calculate the cosine of the angle

    # Clamp the value to the range [-1, 1] to avoid any floating point errors
    cos_theta = max(-1, min(1, cos_theta))

    # Use the arccos function to find the angle in radians
    theta_rad = math.acos(cos_theta)

    # Convert the angle to degrees
    theta_deg = math.degrees(theta_rad)

    # Calculate the cross product
    cross_product = v1.x * v2.y - v1.y * v2.x

    # Determine which vector is on the left
    if cross_product > 0:
        position = 0
    elif cross_product < 0:
        position = 1
    else:
        position = 0

    return theta_rad, theta_deg,position
