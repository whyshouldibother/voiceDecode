#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <vector>
#include <string>
class Classifier {
    private: 
        std::vector<int> layers;
        std::vector<std::vector<std::vector<double>>> weights;
        std::vector<std::vector<double>>biases;
    public:
        std::vector<std::vector<double>> forward(const std::vector<double>& inputs);
        Classifier(const int&, const int&);
        std::vector<double> predict (const std::vector<double>&);
        double calculateLoss(const std::vector<double>&, std::size_t) const;
        void trainSample(const std::vector<double>& inputs, std::size_t label, double learningRate);
        bool save(const std::string&) const; 
        bool load(const std::string&);
};
struct AudioSample {
    std::vector<double> inputs;
    std::size_t label;
};
#endif
