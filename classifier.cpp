#include "classifier.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <stdexcept>
Classifier::Classifier(const int &inputSize, const int &outputSize) {
  layers = {inputSize, 128, 64, outputSize};
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

std::vector<double> Classifier::predict(const std::vector<double> &inputs) {
  std::vector<double> activations = inputs;
  for (size_t l = 0; l < weights.size(); l++) {
    std::vector<double> next(layers[l + 1], 0.0);
    for (size_t i = 0; i < next.size(); i++) {
      next[i] = biases[l][i];

      for (size_t j = 0; j < activations.size(); j++) {
        next[i] += weights[l][i][j] * activations[j];
      }

      if (l < weights.size() - 1) {
        next[i] = std::max(0.0, next[i]);
      }
    }
    activations = std::move(next);
  }
  return activations;
}

double Classifier::calculateLoss(const std::vector<double> &outputs,
                                 std::size_t label) const {
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
                             std::size_t label, double learningRate) {
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
  std::ofstream file(filename);

  if (!file) {
    return false;
  }

  file << std::setprecision(std::numeric_limits<double>::max_digits10);
  file << layers.size() << std::endl;

  for (int size : layers) {
    file << size << '\n';
  }

  file << '\n';

  for (std::size_t l = 0; l < weights.size(); l++) {
    for (const std::vector<double> neuron : weights[l]) {
      for (double weight : neuron) {
        file << weight << ' ';
      }
      file << '\n';
    }

    for (double bias : biases[l]) {
      file << bias << ' ';
    }
    file << '\n';
  }
  return static_cast<bool>(file);
}

bool Classifier::load(const std::string &filename) {
  std::ifstream file(filename);
  if (!file) {
    return false;
  }
  std::size_t count;

  if (!(file >> count) || count < 2 || count > 100) {
    return false;
  }
  std::vector<int> newLayers(count);

  for (int &size : newLayers) {
    if (!(file >> size) || size <= 0) {
      return false;
    }
  }
  std::vector<std::vector<std::vector<double>>> newWeights;
  std::vector<std::vector<double>> newBiases;
  for (std::size_t l = 1; l < newLayers.size(); l++) {
    std::size_t inputCount = newLayers[l - 1];
    std::size_t outputCount = newLayers[l];

    newWeights.emplace_back(outputCount, std::vector<double>(inputCount));

    newBiases.emplace_back(outputCount);

    for (auto &neuron : newWeights.back()) {
      for (double &weight : neuron) {
        if (!(file >> weight)) {
          return false;
        }
      }
    }

    for (double &bias : newBiases.back()) {
      if (!(file >> bias)) {
        return false;
      }
    }
  }

  layers = std::move(newLayers);
  weights = std::move(newWeights);
  biases = std::move(newBiases);

  return true;
}
