#pragma once

#include <raylib.h>
#include <vector>
#include <map>
#include <string>
void graphPieByCount(const std::vector<double>&,const Rectangle&, const double& range=0.0, std::vector<Color> colors = {});
void graphPieByValue(const std::vector<double>&, const Rectangle&, std::vector<Color> colors={});
void graphPieByValue(const std::vector<std::pair<std::string, double>>&, const Rectangle&, std::vector<Color> colors = {});
