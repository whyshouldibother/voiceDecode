#include "dataset.hpp"
#include "classifier.hpp"
#include "decode.hpp"
#include "preprocessor.hpp"

#include<filesystem>
#include <iostream>
#include <algorithm>


namespace fs = std::filesystem;

std::vector<AudioSample> loadDataset(const char* directory){
    std::vector <AudioSample> dataset;
    std::vector<fs::path> folders;
    for(const auto& entry: fs::directory_iterator(directory)){
        if(entry.is_directory())
            folders.push_back(entry.path());
    }
    std::sort(folders.begin(), folders.end());
    for(const auto& folder: folders){
        std::size_t label = std::stoul(folder.filename().string());
        for(const auto& file : fs::directory_iterator(folder)){
            if(!file.is_regular_file())
                continue;
            if(file.path().extension() !=".mp3")
                continue;
            auto[audio,sample_rate] = decode(file.path().c_str());
            auto inputs = generateInputNodes(audio, sample_rate);
            dataset.push_back({
                    std::move(inputs), label
                    });
        }
    }
    std::cout<<"Total samples: "<<dataset.size()<<std::endl;
    return dataset;
}

