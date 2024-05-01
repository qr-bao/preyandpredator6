import constants
import utils
from creature import *

MAX_HEALTH = constants.PredatorMAX_HEALTH
VIEW_RADIUS = constants.PredatorVIEW_RADIUS
MAX_VELOCITY = constants.PredatorMAX_VELOCITY

SIZE = constants.PredatorSIZE
COLOR = constants.PredatorCOLOR

HEALTH_GAIN = constants.PredatorHEALTH_GAIN
HEALTH_LOSS = constants.PredatorHEALTH_LOSS

MAX_DETECTION_OF_PREY = constants.MAX_DETECTION_OF_PREY
MAX_ATTRACTION_TO_PREY = constants.MAX_ATTRACTION_TO_PREY

MUTATION_AMOUNT = constants.PredatorsMUTATION_AMOUNT


class Predator(Creature):
    def __init__(self, creatureFields, detection, attraction):
        super(Predator, self).__init__(*creatureFields)
        self.detectionOfPrey = detection
        self.attractionOfPrey = attraction
        self.velocity.scale_to_length(self.maxVelocity)
        self.velocitydirection = pygame.math.Vector2(self.velocity)

    def distanceinformation(self, preys, predators, foods):
        (
            FilteredPrey,
            predatorDistancePrey,
            predatorDistancePredator,
            predatorDistanceFood,
        ) = utils.PreyFilterUsingEuclideanDistances1(
            self.velocitydirection,
            (self.rect.centerx, self.rect.centery),
            preys,
            predators,
            foods,
            self.fieldRadius,
        )
        HearningL, HearningR = utils.Hearning(
            (self.rect.centerx, self.rect.centery),
            predators,
            foods,
            preys,
            self.fieldRadius,
        )
        return (
            predatorDistancePrey,
            predatorDistancePredator,
            predatorDistanceFood,
            HearningL,
            HearningR,
        )

    def getTarget(self, preys, predators, foods):
        if len(preys) == 0:
            print("NO Prey")
        (
            FilteredPrey,
            predatorDistancePrey,
            predatorDistancePredator,
            predatorDistanceFood,
        ) = utils.PreyFilterUsingEuclideanDistances1(
            self.velocitydirection,
            (self.rect.centerx, self.rect.centery),
            preys,
            predators,
            foods,
            self.fieldRadius,
        )
        HearningL, HearningR = utils.Hearning(
            (self.rect.centerx, self.rect.centery),
            predators,
            foods,
            preys,
            self.fieldRadius,
        )

        return utils.PredictPredatorDirection(
            (self.rect.centerx, self.rect.centery),
            FilteredPrey,
            self.attractionOfPrey,
            HearningL,
            HearningR,
        )

    def move(self, width, height, prey, predator, food):
        targetVelocity = self.getTarget(prey, predator, food)

        if targetVelocity.magnitude() != 0:
            targetVelocity.scale_to_length(self.maxVelocity)
            steer = targetVelocity - self.velocity
            if steer.magnitude() > constants.maxForce:
                steer.scale_to_length(constants.maxForce)

            self.velocity = self.velocity + steer
            self.health -= 1.8 * steer.magnitude()

        super().move(width, height)

    def details(self):
        if self.alive:
            super().details("Predator")
            print(f"The Predator's detection capability is {self.detectionOfPrey}")
            print(f"The Predator's attraction capability is {self.attractionOfPrey}")
        else:
            print("The Predator is dead. Sorry :(")

    def crossbreed(self, other):
        childDetectionOfPrey = (self.detectionOfPrey + other.detectionOfPrey) / 2
        childAttractionOfPrey = (self.attractionOfPrey + other.attractionOfPrey) / 2
        return Predator(
            (
                MAX_HEALTH,
                VIEW_RADIUS,
                MAX_VELOCITY,
                (
                    random.randint(0, constants.WIDTH),
                    random.randint(0, constants.HEIGHT),
                ),
                (
                    random.uniform(-1, 1) * constants.INIT_VELOCITY,
                    random.uniform(-1, 1) * constants.INIT_VELOCITY,
                ),
                COLOR,
                SIZE,
            ),
            childDetectionOfPrey,
            childAttractionOfPrey,
        )

    def mutate(self):
        self.detectionOfPrey += (
            random.uniform(-1, 1) * MUTATION_AMOUNT * MAX_DETECTION_OF_PREY
        )
        self.attractionOfPrey += (
            random.uniform(-1, 1) * MUTATION_AMOUNT * MAX_ATTRACTION_TO_PREY
        )
