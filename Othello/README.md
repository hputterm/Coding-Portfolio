# Othello bot

The Othello bot was the result of a group project for a competition at the end of a class. Me and my partner implemented parts of the board representation, a heuristic to represent the state of the game, and the recursive decision methods used to find the most optimal move. The bot initially used a plain minimax decision tree to choose the best move. By adding alpha beta pruning we were able to increase the recursive depth achieved by the bot in the limited time available to it. This bot was written in c++.

The recursive decision trees are in the player.cpp file.  
