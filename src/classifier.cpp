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
  weightOffsets.clear();
  biases.clear();
  std::mt19937 rng(std::random_device{}());
  for (size_t l = 1; l < layers.size(); l++) {
    int inputs = layers[l - 1];
    int outputs = layers[l];

    weightOffsets.push_back(weights.size());
    weights.resize(weights.size() + (inputs * outputs));

    biases.emplace_back(outputs, 0.0f);
    float scale = std::sqrt(2.0f / inputs);
    std::normal_distribution<float> dist(0.0f, scale);
    for (std::size_t i = weightOffsets.back(); i < weights.size(); i++) {
      weights[i] = dist(rng);
    }
  }
}
Classifier::Classifier(const std::vector<std::size_t> &layers)
    : layers(layers) {
  init();
}

std::vector<float> Classifier::predict(const std::vector<float> &inputs) {
  std::vector<float> activations = inputs;
  for (size_t l = 0; l < weightOffsets.size(); l++) {
    std::vector<float> next(layers[l + 1], 0.0f);
//#pragma omp parallel for
    for (size_t i = 0; i < next.size(); i++) {
      float sum = biases[l][i];
      const std::size_t offset = weightOffsets[l] + i * layers[l];
      for (size_t j = 0; j < activations.size(); j++) {
        sum += weights[offset + j] * activations[j];
      }
      next[i] = sum;
      if (l < weightOffsets.size() - 1) {
        next[i] = std::max(0.0f, next[i]);
      }
    }
    activations = std::move(next);
  }
  return activations;
}

float Classifier::calculateLoss(const std::vector<float> &outputs,
                                const std::size_t label) const {
  if (outputs.empty() || label >= outputs.size()) {
    throw std::invalid_argument("Invalid outputs or label");
  }
  float maxValue = *std::max_element(outputs.begin(), outputs.end()),
        sum = 0.0f;
  for (float output : outputs) {
    sum += std::exp(output - maxValue);
  }
  return maxValue + std::log(sum) - outputs[label];
}

std::vector<std::vector<float>>
Classifier::forward(const std::vector<float> &inputs) {
  if (inputs.size() != (size_t)layers[0]) {
    throw std::invalid_argument("Incorrect input size");
  }
  std::vector<std::vector<float>> activations;
  activations.push_back(inputs);
  for (std::size_t l = 0; l < weightOffsets.size(); l++) {
    const std::vector<float> &previous = activations.back();
    std::vector<float> next(layers[l + 1], 0.0f);
//#pragma omp parallel for
    for (size_t i = 0; i < next.size(); i++) {
      float sum = biases[l][i];
      const std::size_t offset = weightOffsets[l] + i * layers[l];
      for (size_t j = 0; j < previous.size(); j++) {
        sum += weights[offset + j] * previous[j];
      }
      next[i] = sum;
      if (l < weightOffsets.size() - 1) {
        next[i] = std::max(0.0f, next[i]);
      }
    }
    activations.push_back(std::move(next));
  }
  return activations;
}

void Classifier::trainSample(const std::vector<float> &inputs,
                             const std::size_t label,
                             const float learningRate) {
  std::vector<std::vector<float>> activations = forward(inputs);
  const std::vector<float> logits = activations.back();
  if (label >= logits.size()) {
    throw std::invalid_argument("Invalid label");
  }
  float maxLogit = *std::max_element(logits.begin(), logits.end());
  std::vector<float> probabilities(logits.size());
  float sum = 0.0f;
  for (std::size_t i = 0; i < logits.size(); i++) {
    probabilities[i] = std::exp(logits[i] - maxLogit);
    sum += probabilities[i];
  }
  for (float &p : probabilities) {
    p /= sum;
  }
  const std::size_t layerCount = layers.size();
  // Gradient
  std::vector<std::vector<float>> deltas(layerCount);
  deltas.back() = probabilities;
  deltas.back()[label] -= 1.0f;

  for (std::size_t l = layerCount - 1; l > 1; l--) {
    deltas[l - 1].assign(layers[l - 1], 0.0f);
    for (std::size_t j = 0; j < layers[l - 1]; j++) {
      for (std::size_t i = 0; i < layers[l]; i++) {
        deltas[l - 1][j] +=
            weights[weightOffsets[l - 1] + i * layers[l - 1] + j] *
            deltas[l][i];
      }
      if (activations[l - 1][j] <= 0.0f) {
        deltas[l - 1][j] = 0.0f;
      }
    }
  }

  // Weight update
  for (std::size_t l = 0; l < weightOffsets.size(); l++) {
    const std::size_t inputSize = layers[l];
    const std::size_t outputSize = layers[l + 1];
    for (std::size_t i = 0; i < outputSize; i++) {
      const std::size_t offset = weightOffsets[l] + i * inputSize;
      const float gradient = learningRate * deltas[l + 1][i];
      for (std::size_t j = 0; j < inputSize; j++) {
        weights[offset + j] -= gradient * activations[l][j];
      }
      biases[l][i] -= gradient;
    }
  }
}

bool Classifier::save(const std::string &filename) const {
  std::ofstream file(filename, std::ios::binary);

  if (!file)
    return false;

  // Save Metadata
  // Epoch
  file.write(reinterpret_cast<const char *>(&epoch), sizeof(epoch));

  // Layer Info
  uint32_t layerCount = layers.size();
  file.write(reinterpret_cast<const char *>(&layerCount), sizeof(layerCount));
  for (const std::size_t &layer : layers) {
    if (layer > UINT32_MAX)
      return false;
    const uint32_t size = static_cast<uint32_t>(layer);
    file.write(reinterpret_cast<const char *>(&size), sizeof(size));
  }

  // Weights
  file.write(reinterpret_cast<const char *>(weights.data()),
             weights.size() * sizeof(float));
  // Bias
  for (const std::vector<float> &layer : biases) {
    file.write(reinterpret_cast<const char *>(layer.data()),
               layer.size() * sizeof(float));
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
  for (std::size_t &layer : layers) {
    uint32_t size;
    file.read(reinterpret_cast<char *>((&size)), sizeof(size));
    if (!file || size <= 0)
      return false;
    layer = size;
  }
  init();
  // Load Weights
  file.read(reinterpret_cast<char *>(weights.data()),
            weights.size() * sizeof(float));
  // Load Biases
  for (std::vector<float> &layer : biases) {
    file.read(reinterpret_cast<char *>(layer.data()),
              layer.size() * sizeof(float));
    if (!file)
      return false;
  }
  return file.peek() == std::char_traits<char>::eof();
}
