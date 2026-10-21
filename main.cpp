#include "classifier.hpp"
#include "dataset.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

namespace fs = std::filesystem;
int main() {
  auto dataset = loadDataset("./samples");
  if (dataset.empty()) {
    std::cerr << "Dataset is empty" << std::endl;
    return 1;
  }
  const std::size_t inputSize = dataset[0].inputs.size();
  std::size_t outputSize = 0;
  for (const AudioSample &sample : dataset) {
    if (sample.inputs.size() != inputSize) {
      std::cerr << "Inconsistent inputs" << std::endl;
      return 1;
    }
    outputSize = std::max(outputSize, sample.label + 1);
  }
  Classifier classifier(inputSize, outputSize);
  std::mt19937 rng(std::random_device{}());

  if (fs::exists("model.dat")) {
    if (!classifier.load("model.dat")) {
      std::cerr << "Failed to load model.dat\n";
      return 1;
    }
    std::cout << "Loaded existing model\n";

  } else {
    std::cout << "Created new model\n";
  }
  int epochs = 100;
  std::cout<<"Additional epochs to train: ";
  std::cin>>epochs;

  if(!std::cin||epochs<=0){
      std::cerr<<"Invalid epoch count\n";
      return 1;
  }


  const double learningRate = 0.001;

  for (int epoch = 0; epoch < epochs; epoch++) {
    std::shuffle(dataset.begin(), dataset.end(), rng);

    for (const auto &sample : dataset) {
      classifier.trainSample(sample.inputs, sample.label, learningRate);
    }

    double totalLoss = 0.0;
    std::size_t correct = 0;

    for (const auto &sample : dataset) {
      std::vector<double> logits = classifier.predict(sample.inputs);
      totalLoss += classifier.calculateLoss(logits, sample.label);

      auto prediction = std::distance(
          logits.begin(), std::max_element(logits.begin(), logits.end()));
      if (static_cast<std::size_t>(prediction) == sample.label) {
        correct++;
      }
    }

    double averageLoss = totalLoss / dataset.size();
    double accuracy = 100.0 * correct / dataset.size();

    std::cout << "Epoch" << epoch + 1 << "/" << " | Loss: " << averageLoss
              << " | Accuracy: " << accuracy << "%\n";
    if(epoch%20==0 && accuracy==100){
        std::cout<<"Stop(y/N)";
        std::string stop = "false";
        std::cin>>stop;
        if(stop[0]=='y' || stop[0]=='Y') break;
    }
  }

  if(classifier.save("model.dat")){
      std::cout<<"Model saved successfully\n";
  }else{
      std::cerr<<"Failed to save model\n";
      return 1;
  }
  return 0;
}
