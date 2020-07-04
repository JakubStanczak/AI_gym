# AI_gym

In this project I explored machine learning using Open AI gym environments

1. Cart_pole - At first the environment was controlled randomly to collect data. Data was then auto labeled (good/bad moves), so a neural network could be taught on it. The outcome was so good that random inputs were added to make the environment harder to control.
2. Mountain_car - All possible environment states are mapped (in bins to save memory) and all possible moves in those situation are given a value that show how efficient they will lid to the correct outcome. Values are random at first but after the algorithm stumbles upon the right destination the moves that led there start to be rewarded and are more likely to be chosen in the future.