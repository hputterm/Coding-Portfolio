This is a repo with some of the projects I have done.  Hopefully you find them interesting.

### Biology Simulations and Cell Computer Vision Analysis

The Gene Expression and Genetic Drift Simulations were both projects done in a biology class I took.  They required extensive use of python to do scientific analysis and simulations.  For instance in the Gene Expression project I used computer vision to identify cells on microscope images and then analyze the fluorescence content to determine the degree of gene expression.  In the Genetic Drift project I used Monte Carlo simulations to model the behavior of populations over time in the presence of external factors that affected their genetic viability.  This is just a small sample of the scientific simulations done during the class.

### IOS Applications

The Graphing Calculator and Twitter Application were both loosely a part of a Stanford course on IOS development.   In the Graphing Calculator project I made a split view application for iPhone/iPad that has a custom view to display a graph for the user.  The application also parsed calculator input to represent the current state of the user input.  The Twitter Application which is for android is far more complex because of the significantly larger amount of MVCs.  This application also has a lot of data which persists between different sessions of the application.  The app uses the core data framework to store and analyze large amounts of information.  Multithreading is used in this application to prevent deadlock when fetching files from the internet.  Both of these applications were programmed in swift.  In the future I plan on reworking the model for the calculator to make it use a priority system when parsing to string.  Currently there are still some bugs left in this part of the project.  
Here are the links to the repos with the applications

https://github.com/hputterm/Twitter-Application

https://github.com/hputterm/Graphing-Calculator

### Othello Bot

The Othello bot was the result of a group project for a competition at the end of a class.  Me and my partner implemented the board representation, a heuristic to represent the state of the game, and the recursive decision methods used to find the most optimal move.  The bot initially used a plain minimax decision tree to choose the best move.  By adding alpha beta pruning we were able to increase the recursive depth achieved by the bot in the limited time available to it.  This bot was written in c++.  
