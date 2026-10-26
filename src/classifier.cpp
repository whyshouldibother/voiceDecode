#include "classifier.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/types.h>

void Classifier::init() {
    weights.clear();
    biases.clear();
  std::mt19937 rng(std::random_device{}());
  for (size_t l = 1; l < layers.size(); l++) {
    int inputs = layers[l - 1];
    int outputs = layers[l];

    weights.emplace_back(outputs, std::vector<double>(inputs));
    biases.emplace_back(outputs, 0.0);

    double scale = std::sqrt(2.0 / inputs);
    std::normal_distribution<double> dist(0.0, scale);

    for (std::vector<double> &neuron : weights.back()) {
      for (double &weight : neuron) {
        weight = dist(rng);
      }
    }
  }
}
Classifier::Classifier(const int &inputSize, const int &outputSize) {
  layers = {inputSize, 128, 64, outputSize};
    init();
}

std::vector<double> Classifier::predict(const std::vector<double> &inputs) {
  std::vector<double> activations = inputs;
  for (size_t l = 0; l < weights.size(); l++) {
    std::vector<double> next(layers[l + 1], 0.0);
    for (size_t i = 0; i < next.size(); i++) {
      next[i] = biases[l][i];
      double sum = next[i];
      std::vector<double> layerWeight = weights[l][i];
      for (size_t j = 0; j < activations.size(); j++) {
        sum += layerWeight[j] * activations[j];
      }
      next[i] = sum;
      if (l < weights.size() - 1) {
        next[i] = std::max(0.0, next[i]);
      }
    }
    activations = std::move(next);
  }
  return activations;
}

double Classifier::calculateLoss(const std::vector<double> &outputs,
                                 const std::size_t label) const {
  if (outputs.empty() || label >= outputs.size()) {
    throw std::invalid_argument("Invalid outputs or label");
  }
  double maxValue = *std::max_element(outputs.begin(), outputs.end()),
         sum = 0.0;
  for (double output : outputs) {
    sum += std::exp(output - maxValue);
  }
  return maxValue + std::log(sum) - outputs[label];
}

std::vector<std::vector<double>>
Classifier::forward(const std::vector<double> &inputs) {
  if (inputs.size() != (size_t)layers[0]) {
    throw std::invalid_argument("Incorrect input size");
  }
  std::vector<std::vector<double>> activations;
  activations.push_back(inputs);
  for (std::size_t l = 0; l < weights.size(); l++) {
    const std::vector<double> &previous = activations.back();
    std::vector<double> next(layers[l + 1], 0.0);
    for (size_t i = 0; i < next.size(); i++) {
      next[i] = biases[l][i];

      for (size_t j = 0; j < previous.size(); j++) {
        next[i] += weights[l][i][j] * previous[j];
      }

      if (l < weights.size() - 1) {
        next[i] = std::max(0.0, next[i]);
      }
    }
    activations.push_back(std::move(next));
  }
  return activations;
}

void Classifier::trainSample(const std::vector<double> &inputs,
                             const std::size_t label,const double learningRate) {
  std::vector<std::vector<double>> activations = forward(inputs);
  const std::vector<double> logits = activations.back();
  if (label >= logits.size()) {
    throw std::invalid_argument("Invalid label");
  }
  double maxLogit = *std::max_element(logits.begin(), logits.end());
  std::vector<double> probabilities(logits.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < logits.size(); i++) {
    probabilities[i] = std::exp(logits[i] - maxLogit);
    sum += probabilities[i];
  }
  for (double &p : probabilities) {
    p /= sum;
  }
  const std::size_t layerCount = weights.size();
  // Gradient
  std::vector<std::vector<double>> deltas(layerCount);
  deltas.back() = probabilities;
  deltas.back()[label] -= 1.0;

  for (std::size_t l = layerCount - 1; l > 0; l--) {
    std::vector<double> previousDelta(layers[l], 0.0);
    for (std::size_t j = 0; j < previousDelta.size(); j++) {
      for (std::size_t i = 0; i < deltas[l].size(); i++) {
        previousDelta[j] += weights[l][i][j] * deltas[l][i];
      }

      if (activations[l][j] <= 0.0) {
        previousDelta[j] = 0.0;
      }
    }
    deltas[l - 1] = std::move(previousDelta);
  }

  // Weight update
  for (std::size_t l = 0; l < layerCount; l++) {
    for (std::size_t i = 0; i < weights[l].size(); i++) {
      for (std::size_t j = 0; j < weights[l][i].size(); j++) {
        double weightGradient = deltas[l][i] * activations[l][j];
        weights[l][i][j] -= learningRate * weightGradient;
      }
      biases[l][i] -= learningRate * deltas[l][i];
    }
  }
}

bool Classifier::save(const std::string &filename) const {
  std::ofstream file(filename, std::ios::binary);

  if (!file)
    return false;

  // Save Metadataclang++ main.cpp decode.cpp preprocessor.cpp classifier.cpp dataset.cpp \
    -lmpg123 -O3 -march=native  \
    -o train
  // Epoch
  file.write(reinterpret_cast<const char *>(&epoch), sizeof(epoch));

  // Layer Info
  uint32_t layerCount = layers.size();
  file.write(reinterpret_cast<const char *>(&layerCount), sizeof(layerCount));
  for (const int &layer : layers) {
    file.write(reinterpret_cast<const char *>(&layer), sizeof(layer));
  }

  // Weights
  for (const std::vector<std::vector<double>> &layer : weights) {
    for (const std::vector<double> &neuron : layer) {
      file.write(reinterpret_cast<const char *>(neuron.data()),
                 neuron.size() * sizeof(double));
    }
  }

  // Bias
  for (const std::vector<double> &layer : biases) {
    file.write(reinterpret_cast<const char *>(layer.data()),
               layer.size() * sizeof(double));
  }

  return file.good();
}

bool Classifier::load(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    return false;
  }
  file.read(reinterpret_cast<char *>(&epoch), sizeof(epoch));
  uint32_t layerCount;
  file.read(reinterpret_cast<char *>(&layerCount), sizeof(layerCount));
  if (!file || layerCount < 2)
    return false;
  layers.resize(layerCount);
  for (int &layer : layers) {
    file.read(reinterpret_cast<char *>((&layer)), sizeof(layer));
    if (!file || layer <= 0)
      return false;
  }
  init();

  //Load Weights
  for (std::vector<std::vector<double>> &layer : weights) {
    for (std::vector<double> &neuron : layer) {
      file.read(reinterpret_cast<char *>(neuron.data()),
                neuron.size() * sizeof(double));
      if (!file)
        return false;
    }
  }

  //Load Biases
  for(std::vector<double>& layer: biases){
      file.read(reinterpret_cast<char*>(layer.data()), layer.size()*sizeof(double));
      if(!file) return false;
  }
  return file.peek() == std::char_traits<char>::eof();
}
