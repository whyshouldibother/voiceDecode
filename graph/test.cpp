#include <raylib.h>
#include "graph.h"

#include <vector>

int main(){
    InitWindow(800, 600, "Pie Chart Test");
    SetTargetFPS(60);

    std::vector<double> testValues = {100, 123, 123, 123, 4123, 123};

    while(!WindowShouldClose()){
    BeginDrawing();
    ClearBackground(WHITE);
    graphPieByCount(testValues, {0, 0, 500, 500});
    EndDrawing();
    }
    CloseWindow();
    return 0;
}
