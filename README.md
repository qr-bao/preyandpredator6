this is a homework about predator and prey, I refer to “PredatorAndPreySimulation Public” project‘s work the original code is here: https://github.com/unknownblueguy6/PredatorAndPreySimulation

I've added some features:
The vision of predators and prey is limited, and they can only see creatures within a certain angle and within a certain range.

predator and prey can hear sounds. They can know whether there are creatures nearby, but they don't know whether they are predators or predators, and their specific locations.

The predator's health not only decreases with the passage of time, he also decreases with the size of the movement acceleration.

It falls as the predator moves, and the faster the predator moves, the faster it falls. In order to better simulate energy consumption and ecological balance. The original health consumption parameters have been modified.

In order to make the model simple, the function that allows the prey to escape when the predator catches the prey is deleted.

New features I'm trying to add
Standardize the code to create a matrix of surrounding biological information for each creature at each moment. (FINISHED)

Use reinforcement learning to play a game and plot the number and energy of creatures over time for Prey and Predator.(PART FINISHED)

Change the map, add obstacles to the map, and continue playing the game using reinforcement learning (HAVE NOT STARTED)
Add obstacles to the environment（DOING）
Changing food production strategies（DOING）
