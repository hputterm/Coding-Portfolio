#ifndef __PLAYER_H__
#define __PLAYER_H__
#define MINIMAX_DEPTH 7

#include <iostream>
#include "common.hpp"
#include "board.hpp"
using namespace std;

class Player {

public:
    Player(Side side);
    ~Player();

    Move *doMove(Move *opponentsMove, int msLeft);  
    
    int two_ply_minimax(vector<Move*> possible_moves);
    int recursive_minimax(int depth, Side side, vector<Move*> possible_moves,
                          Board *new_board, Move *&best_move);
    int alpha_beta_prune(int depth, Side side, vector<Move*> possible_moves,
                             Board *new_board, Move *&best_move, int alpha, 
                             int beta);   
    // Flag to tell if the player is running within the test_minimax context
    bool testingMinimax;
    Side playerSide;
    Side opponentSide;
    Board *othelloBoard;
};

#endif
