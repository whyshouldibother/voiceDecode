#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <cstdint>
#include <vector>
#include <string>
class Classifier {
    private: 
        std::vector<int> layers;
        std::vector<std::vector<std::vector<float>>> weights;
        std::vector<std::vector<float>>biases;
        void init();
    public:
        uint32_t epoch=0;
        std::vector<std::vector<float>> forward(const std::vector<float>& inputs);
        Classifier(const int&, const int&);
        std::vector<float> predict (const std::vector<float>&);
        float calculateLoss(const std::vector<float>&, const std::size_t) const;
        void trainSample(const std::vector<float>& inputs, const std::size_t label, float learningRate);
        bool save(const std::string&) const; 
        bool load(const std::string&);
};
struct AudioSample {
    std::vector<float> inputs;
    std::size_t label;
};
#endif
