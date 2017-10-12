#ifndef __BOARD_H__
#define __BOARD_H__
#define CORN 4
#define CORN_EMPTY_ADJ 2
#define CORN_EMPTY_DIAG 4
#define EDGE 2

#include <bitset>
#include "common.hpp"
#include <vector>

using namespace std;

class Board {

private:
    bitset<64> black;
    bitset<64> taken;

public:
    Board();
    ~Board();
    Board *copy();
    
    bool occupied(int x, int y);
    bool get(Side side, int x, int y);
    void set(Side side, int x, int y);
    bool onBoard(int x, int y);


    bool isDone();
    bool hasMoves(Side side);
    bool checkMove(Move *m, Side side);
    void doMove(Move *m, Side side);
    int count(Side side);
    int countBlack();
    int countWhite();
    int calculateScore(Side playerSide);
    vector<Move*> possibleMoves(Side playerSide);
    int getDiffScore(Side playerSide);

    void setBoard(char data[]);
};

#endif
