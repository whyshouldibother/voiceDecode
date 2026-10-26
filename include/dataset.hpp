#ifndef DATASET_HPP
#define DATASET_HPP

#include "classifier.hpp"
#include <vector>

std::vector<AudioSample> loadDataset(const char* directory);
#endif
