# Q-Learner-Learner

This is a little pet project to visualize how a simple Q-learner works, specifically in the context of a path-finding algorithm. I learned about Q-learning in an online course, but I was pretty disappointing in the visualization provided to study / test Q-learning. So, I decided to re-write a Q-learner from scratch and create a visual environment around it to help visualize and understand the development of Q-tables as an entity explores its environment.

This project implements basic Q-learning to solve randomly generated mazes, visually demonstrating the development of the Q-table is it progresses.

## How to use

The only non-native dependencies for the project are numpy and pygame.
```
pip install numpy pygame
```

Then, you can run the visualization by simply calling the main program.
```
python3 main.py
```

Once the visualization has loaded, follow these steps to run it:
1. Click `Generate Walls` to randomly generate walls until you have a maze you like
    - Note: if the entity is boxed in or the reward is not reachable, the model will just keep trying forever without much success.

2. **Optional:** Click `Weights: Off` to see the weights in real time (this is the core functionality of this visualization!), and click `Set to 2x speed` if it's moving too slowly for you.

3. Click `Start Training` to begin training the model.
    - The model will explore the environment and build a Q-table that learns the shortest possible path to solve the maze.

4. Click `Stop Training` to stop training the model once it's figured out the shortest possible path and can reliably take it (look for a trail of positive Q-table weights)

5. Click `Start Model` to watch the model run without updating its Q-table anymore.

## About the Environment

The entity is the green square in the visualization, and it's goal is to reach the red square.
- The map on the left side of the visualization shows the environment's set weights
    - Walls have a highly negative reward of `-100`.
    - The floor has a slightly negative reward of `-1`.
    - The goal has a large reward of `+100`.
- The map on the right side of the visualization shows a selection from the model's Q-Table
    - Each time the entity enters the state, the reward value from the Q-learner previously used to determine the action to enter that space is recorded and written on the square
    - Additionally, the direction of the state from which that value was derived is also recorded to aid visual understanding.
    - As the model trains, you can see a "reverse snake" of positive weights form from the goal moving back to the starting point. 