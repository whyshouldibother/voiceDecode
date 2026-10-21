#include "graph.h"
#include <random>
#include <raylib.h>
#include <vector>
#include <map>
#include <string>
Color RandomColor(size_t index = 0){
    std::mt19937 rng(index);
    std::uniform_int_distribution<int> r(index%3*90, 255);
    std::uniform_int_distribution<int> g((index+1)%3*90, 255);
    std::uniform_int_distribution<int> b((index%+2)%3*90, 255);
    return {(unsigned char)r(rng), (unsigned char)g(rng), (unsigned char)b(rng), 255};
}
void ColorExtender(std::vector<Color> &colors, const size_t& maxSize){
    while(colors.size()<maxSize){
        colors.push_back(RandomColor(colors.size()));
    }
}
std::vector<std::pair<std::string, double>> countValues(const std::vector<double>& data, const double &range){
    std::map<std::string,double> counts;
    std::string key;
    double group;
    for(double value: data){
        if(range ==0){
            key = std::to_string(value);
        }
        else{
            group = floor(value/range);
            key = std::to_string(group * range) + ":"+ std::to_string((group+1)*range);
        }
        counts[key]++;
    }
    std::vector<std::pair<std::string, double>> countsPair;
    for(auto count:counts){
        countsPair.push_back(count);
    }
    return countsPair;
}
void graphPieByValue(const std::vector<std::pair<std::string, double>>& values,const Rectangle& bounds, std::vector<Color> colors){
    if(values.empty() || bounds.width <=0 || bounds.height <=0) return;
    double total = 0.0f;
    for(auto value: values){
        if(!std::isfinite(value.second) || value.second<0.0f)return;
        total += value.second;
    }
    if(total <= 0.0f || !std::isfinite(total)) return ;
    if(colors.size() < values.size()) ColorExtender(colors, values.size());
    Vector2 center = {bounds.x + bounds.width/2.0f, bounds.y + bounds.height/2.0f};
    float radius = fminf(bounds.width, bounds.height) / 2.5f;
    float angle = 0.0f;
    Vector2 label;
    float sliceAngle, midAngle;
    for(int i=0; i<values.size(); i++){
        auto entry = std::next(values.begin(), i);
        if(entry->second==0.0f) continue;
        sliceAngle = entry->second / total * 360.0f;
        midAngle = (angle + sliceAngle / 2.0f) * DEG2RAD;
        DrawCircleSector(center, radius, angle, angle+sliceAngle, 0, colors[i]);
        DrawCircleSectorLines(center, radius, angle, angle+sliceAngle,0,BLACK);
        angle += sliceAngle;
        label = {center.x+ cosf(midAngle) * (radius + 20), center.y + sinf(midAngle) * (radius+20)};
        DrawText(entry->first.c_str(), label.x, label.y, 16, BLACK);
    }
}
void graphPieByValue(const std::vector<double>& values,const Rectangle& bounds, std::vector<Color> colors){
    std::vector<std::pair<std::string, double>> counts;
    for(double value: values){
        counts.push_back({std::to_string(value), value});
    }
        graphPieByValue(counts, bounds, colors);
}
void graphPieByCount(const std::vector<double>& values, const Rectangle& bounds, const double& range, std::vector<Color> colors){
    std::vector<std::pair<std::string, double>> counts = countValues(values, range);
    graphPieByValue(counts, bounds, colors);
}
