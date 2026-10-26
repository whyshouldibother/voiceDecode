#include "preprocessor.hpp"
#include <vector>
#include <algorithm>
std::vector<double> generateInputNodes(const std::vector<double> &audio, const long &sample_rate){

//grouping into input vector
    size_t groupSize = sample_rate/100, count;
    std::vector<double> input;
    double sum = 0;
    for(size_t i=0; i<audio.size(); i+=groupSize){
        sum =0 ;
        count = std::min(groupSize, audio.size() - i);

        for(size_t j=0; j<count; j++){
            sum += audio[i+j];
        }
        input.push_back(sum/count);
    }
    if(input.size()<400){
        input.resize(400, 0.0);
    }else if(input.size()>400){
        size_t start = (input.size()-400)/2;
        input = std::vector<double>(input.begin() + start, input.begin() + start + 400);
    }
    return input;
}
