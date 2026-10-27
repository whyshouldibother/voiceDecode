#include "classifier.hpp"
#include "dataset.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <random>
#include <cstring>
#include <vector>
#ifndef MODEL
#define MODEL "commands"
#endif
#ifndef DATA_DIR
#define DATA_DIR "."
#endif
namespace fs = std::filesystem;
int main(const int argc, const char *argv[]) {
  const fs::path datasetPath = fs::path(DATA_DIR) / "data" / MODEL / "samples";
  auto dataset = loadDataset(datasetPath.c_str());
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
    outputSize = (std::size_t)std::max(outputSize, sample.label + 1);
  }
  const std::vector<std::size_t> layers = (std::strcmp(MODEL,"command")==0)?(std::vector<std::size_t>({inputSize, 128, 64, outputSize})):(std::vector<std::size_t>({inputSize, 32,outputSize}));
  Classifier classifier(layers);
  std::mt19937 rng(std::random_device{}());
  fs::path modelFile = fs::path(DATA_DIR) / "models" / MODEL ".bin";
  const float learningRate = 0.001;
  size_t silent = 0;
  if (argc > 1 && std::isdigit(static_cast<unsigned char>(argv[1][0]))) {
    silent = argv[1][0] - '0';
  }
  if (argc > 2) {
    modelFile = argv[2];
  }

  if (fs::exists(modelFile)) {
    if (!classifier.load(modelFile)) {
      std::cerr << "Failed to load " << modelFile << std::endl;
      return 1;
    }
    std::cout << "Loaded existing model" << std::endl
              << "Epoch:" << classifier.epoch << std::endl;

  } else {
    std::cout << "Created new model\n";
  }
  int epochs;
  std::cout << "Additional epochs to train: ";
  std::cin >> epochs;

  if (!std::cin || epochs <= 0) {
    std::cerr << "Invalid epoch count\n";
    return 1;
  }

  const int checkGap = epochs / 5;
  int epoch = 0;
  for (; epoch < epochs; epoch++) {

    std::shuffle(dataset.begin(), dataset.end(), rng);

    for (const auto &sample : dataset) {
      classifier.trainSample(sample.inputs, sample.label, learningRate);
    }

    float totalLoss = 0.0;
    std::size_t correct = 0;

    if (epoch % checkGap == 0) {

      for (const auto &sample : dataset) {
        std::vector<float> logits = classifier.predict(sample.inputs);
        totalLoss += classifier.calculateLoss(logits, sample.label);

        auto prediction = std::distance(
            logits.begin(), std::max_element(logits.begin(), logits.end()));
        if (static_cast<std::size_t>(prediction) == sample.label) {
          correct++;
        }
      }
      float averageLoss = totalLoss / dataset.size();
      float accuracy = 100.0 * correct / dataset.size();

      if (silent < 2) {
        std::cout << "Epoch" << epoch + 1 << "/" << " | Loss: " << averageLoss
                  << " | Accuracy: " << accuracy << "%\n";
      }
      if (averageLoss <= 1e-5 && silent == 0) {
        std::cout << "Stop(y/N)";
        std::string stop = "false";
        std::cin >> stop;
        if (stop[0] == 'y' || stop[0] == 'Y')
          break;
      }
    }
  }
  classifier.epoch += epoch;
  if (classifier.save(modelFile)) {
    std::cout << "Model saved successfully\n";
  } else {
    std::cerr << "Failed to save model\n";
    return 1;
  }
  return 0;
}
