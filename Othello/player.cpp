#include "player.hpp"
#include <vector>
#include <iostream>
#include <climits>
using namespace std;

/*
 * Constructor for the player; initialize everything here. The side your AI is
 * on (BLACK or WHITE) is passed in as "side". The constructor must finish
 * within 30 seconds
 */
Player::Player(Side side) {
    // Will be set to true in test_minimax.cpp.
    testingMinimax = false;

    /*
     * TODO: Do any initialization you need to do here (setting up the board,
     * precalculating things, etc.) However, remember that you will only have
     * 30 seconds.
     */
     
    othelloBoard = new Board;
    playerSide = side;
    if (playerSide == WHITE)
    {
        opponentSide = BLACK;
    }
    else
    {
        opponentSide = WHITE;
    }   
}

/*
 * Destructor for the player.
 */
Player::~Player() {
}


/*
 * Compute the next move given the opponent's last move. Your AI is
 * expected to keep track of the board on its own. If this is the first move,
 * or if the opponent passed on the last move, then opponentsMove will be
 * nullptr.
 *
 * msLeft represents the time your AI has left for the total game, in
 * milliseconds. doMove() must take no longer than msLeft, or your AI will
 * be disqualified! An msLeft value of -1 indicates no time limit.
 *
 * The move returned must be legal; if there are no valid moves for your side,
 * return nullptr.
 */
Move *Player::doMove(Move *opponentsMove, int msLeft) {
    /*
     * TODO: Implement how moves your AI should play here. You should first
     * process the opponent's opponents move before calculating your own move
     */
    

    if (opponentsMove != nullptr)
    {
        othelloBoard->doMove(opponentsMove, opponentSide);
    }
        
    if (othelloBoard->hasMoves(playerSide))
    {
         vector<Move*> possible_moves = 
                 othelloBoard->possibleMoves(playerSide);
       
        
       // int bestX;
        Move *bestMove = possible_moves[0];
        
        // 2-ply minimax
       /* bestX = two_ply_minimax(possible_moves);
        bestMove = possible_moves[bestX];*/
        
        // recursive minimax
       /* recursive_minimax(MINIMAX_DEPTH - 1, playerSide, possible_moves,
                                 othelloBoard, bestMove);*/
                                  
        // alpha-beta pruning
        int alpha = INT_MIN;
        int beta = INT_MAX;       
        
        alpha_beta_prune(MINIMAX_DEPTH - 1, playerSide, possible_moves,
                         othelloBoard, bestMove, alpha, beta);
        
        othelloBoard->doMove(bestMove, playerSide);
        return bestMove;
    }
    
    else
    {
        return nullptr;
    }
}


// return the index of the best move in the possible moves for two ply
int Player::two_ply_minimax(vector<Move*> possible_moves)
{
    int maximum_min = INT_MIN;
    int bestX = 0;
    
    for (unsigned int x = 0; x < possible_moves.size(); x++)
    {
        Board *newOthelloCopy = othelloBoard->copy();
        newOthelloCopy->doMove(possible_moves[x], playerSide);
        int localMin = INT_MAX;
        vector<Move*> opponent_possibles = newOthelloCopy->
                                           possibleMoves(opponentSide);
        for (unsigned int y = 0; y < opponent_possibles.size(); y++)
        {
            Board *newNewOthelloCopy = newOthelloCopy->copy();
            newNewOthelloCopy->doMove(opponent_possibles[y], opponentSide);
           
            // better heuristic
            if (testingMinimax == false)
            {
                if (newNewOthelloCopy->calculateScore(playerSide) < localMin)
                {
                    localMin = newNewOthelloCopy->calculateScore(playerSide);
                }
            }
            
            // simple difference heuristic               
            else
            {
                if (newNewOthelloCopy->getDiffScore(playerSide) < localMin)
                {
                    localMin = newNewOthelloCopy->getDiffScore(playerSide);
                }
            }          
            delete newNewOthelloCopy;
        }
        
        if (localMin > maximum_min)
        {
            maximum_min = localMin;
            bestX = x;
        }
        delete newOthelloCopy;
    }
    
    return bestX;
}    

int Player::recursive_minimax(int depth, Side side, vector<Move*> possible_moves,
                              Board *new_board, Move *&best_move)
{    
    int depth_counter = MINIMAX_DEPTH - depth;
    int best_score;
    int local_score;
    
    if (side == opponentSide)
    {
        best_score = INT_MAX;
    }
    else
    {
        best_score = INT_MIN;
    }

    for (unsigned int x = 0; x < possible_moves.size(); x++)
    {
        Board *newOthelloCopy = new_board->copy();
        newOthelloCopy->doMove(possible_moves[x], side);
        if (depth_counter < MINIMAX_DEPTH)
        { 
            vector<Move*> next_possibles;
            Side new_side;
            if (side == playerSide)
            {
                next_possibles = newOthelloCopy->possibleMoves(opponentSide);
                new_side = opponentSide;
            }
            else
            {
                next_possibles = newOthelloCopy->possibleMoves(playerSide);
                new_side = playerSide;
            }
            
            local_score = recursive_minimax(depth - 1, new_side, next_possibles,
                                            newOthelloCopy, best_move);
            for (unsigned int i = 0; i < next_possibles.size(); i++)
            {
                delete next_possibles[i];
            }
        }
        
        else
        {
            if (testingMinimax == false)
            {
                local_score = newOthelloCopy->calculateScore(playerSide);
            }
            else
            {
                local_score = newOthelloCopy->getDiffScore(playerSide);
            }                
        }
        
        if (side == opponentSide)
        {
            if (local_score < best_score)
            {
                best_score = local_score;
                if (depth_counter == 1)
                {
                    best_move = possible_moves[x];
                }
            }
        }
        else
        {
            if (local_score > best_score)
            {
                best_score = local_score;
                if (depth_counter == 1)
                {
                    best_move = possible_moves[x];
                }
            }
        }

        delete newOthelloCopy;
    }
    
    return best_score;
}

int Player::alpha_beta_prune(int depth, Side side, vector<Move*> possible_moves,
                             Board *new_board, Move *&best_move, int Alpha, 
                             int Beta)
{    
    int depth_counter = MINIMAX_DEPTH - depth;
    int local_score;
    int alpha = Alpha;
    int beta = Beta;
    

    for (unsigned int x = 0; x < possible_moves.size(); x++)
    {
        Board *newOthelloCopy = new_board->copy();
        newOthelloCopy->doMove(possible_moves[x], side);
        if (depth_counter < MINIMAX_DEPTH)
        { 
            vector<Move*> next_possibles;
            Side new_side;
            if (side == playerSide)
            {
                next_possibles = newOthelloCopy->possibleMoves(opponentSide);
                new_side = opponentSide;
            }
            else
            {
                next_possibles = newOthelloCopy->possibleMoves(playerSide);
                new_side = playerSide;
            }
            
            local_score = alpha_beta_prune(depth - 1, new_side, next_possibles,
                                        newOthelloCopy, best_move, alpha, beta);
            for (unsigned int i = 0; i < next_possibles.size(); i++)
            {
                delete next_possibles[i];
            }
        }
        
        else
        {
            if (testingMinimax == false)
            {
                local_score = newOthelloCopy->calculateScore(playerSide);
            }
            else
            {
                local_score = newOthelloCopy->getDiffScore(playerSide);
            }                
        }
        
        if (side == opponentSide)
        {
            if (local_score < beta)
            {
                beta = local_score;
                if (depth_counter == 1)
                {
                    best_move = possible_moves[x];
                }
            }
            if (beta <= alpha)
            {
                break;
            }
        }
        else
        {
            if (local_score > alpha)
            {
                alpha = local_score;
                if (depth_counter == 1)
                {
                    best_move = possible_moves[x];
                }
            }
            if (alpha >= beta)
            {
                break;
            }
        }

        delete newOthelloCopy;
    }
    
    
    if (side == opponentSide)
    {
        return beta;
    }
    else
    {
        return alpha;
    }
}    
