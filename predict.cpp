#include "classifier.hpp"
#include "decode.hpp"
#include "preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]){
    if(argc!=2){
        std::cerr<<"Usage :"<<argv[0]<<"<audio_path>\n";
        return 1;
    }

    Classifier classifier(400,2);

    if(!classifier.load("model.dat")){
        std::cerr<<"Failed to load model.dat\n";
        return 1;
    }

    auto [audio, sampleRate] = decode(argv[1]);
    std::vector<double> inputs = generateInputNodes(audio, sampleRate);

    std::vector<double> logits = classifier.predict(inputs);

    auto prediction = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));

    double maxLogit = *std::max_element(logits.begin(), logits.end());
    
    double sum = 0.0;

    for(double logit: logits){
        sum+= std::exp(logit -maxLogit);
    }

    double confidence = std::exp(logits[prediction] - maxLogit) /sum;
    std::cout<<"Predicted command: "<<prediction<<'\n';
    std::cout<<"Confidence: " <<confidence * 100.0 <<"%\n";
}
