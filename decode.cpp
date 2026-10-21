#include <iostream>
#include<mpg123.h>
#include<vector>
#include "decode.hpp"
#include <utility>
#include "preprocessor.hpp"
std::pair<std::vector<double>, long> decode(const char* file){
    std::vector<unsigned char> audio_data;
    //initialize the mpg123 lib
    if(mpg123_init() !=MPG123_OK){
        std::cerr<<"Failed to initialize mpg123"<<std::endl;
        exit(1);
    }

    int error = MPG123_OK;
    mpg123_handle* decoder = mpg123_new(nullptr, &error);
    if(decoder == nullptr){
        std::cerr << "Failed to create decoder"<<std::endl;
        mpg123_exit();
        exit(1);
    }
    if(mpg123_open(decoder, file) != MPG123_OK){
        std::cerr<<"Failed to open MP3: " <<mpg123_strerror(decoder)<< std::endl;
        mpg123_delete(decoder);
        mpg123_exit();
        exit(1);
    }

    int channels, encoding; 
    long sample_rate;
    if(mpg123_getformat(decoder, &sample_rate, &channels, &encoding) != MPG123_OK){
        std::cerr<<"Failed to extract audio format"<<std::endl;
        mpg123_close(decoder);
        mpg123_delete(decoder);
        mpg123_exit();
        exit(1);
    }
    
    std::cout << "Sample rate: " <<sample_rate<<" Hz"<<std::endl;
    std::cout<<"Channels: "<<channels<<std::endl;
    std::cout<<"Encoding: "<<encoding<<std::endl;

    unsigned char buffer[8192];
    size_t bytes_read = 0, total_bytes = 0;

    while(true){
        int status = mpg123_read(decoder, buffer, sizeof(buffer), &bytes_read);

        total_bytes += bytes_read;
        audio_data.insert(audio_data.end(), buffer, buffer+bytes_read);
        if(status == MPG123_NEW_FORMAT){
            if(mpg123_getformat(decoder, &sample_rate, &channels, &encoding) != MPG123_OK){
                std::cerr<<"Failed to read updated audio format"<<std::endl;
                break;
            }
            std::cout<<"Audio format changed during decoding"<<std::endl;
        }else if(status == MPG123_DONE){
            break;
        }else if(status != MPG123_OK){
            std::cerr<<"Decoding error: "<<mpg123_strerror(decoder)<<std::endl; 
        break;
        }
    }
    std::cout<<"Decoded bytes: "<<total_bytes<<std::endl;
    mpg123_close(decoder);
    mpg123_delete(decoder);
    mpg123_exit();


    std::vector<short>samples;
    for(size_t i=0; i+1<audio_data.size(); i+=2){
        short sample = audio_data[i] | (audio_data[i+1] << 8);
        samples.push_back(sample);
    }

    std::vector<double> time, audio;

    for(size_t i=0; i+1<samples.size(); i+=2){
        time.push_back((i/2.0)/sample_rate);
        audio.push_back((samples[i]+samples[i+1])/2.0);
    }
    return {audio, sample_rate};
}
#ifdef STANDALONE
int main(int argc, char* argv[]){
    //check if the file is actually being properly passed or not
    if(argc!=2){
        std::cerr<<"Usage: "<<argv[0]<<" <file.mp3>"<<std::endl;
        return 1;
    }
    auto [audio, sample_rate]= decode(argv[1]);
    std::vector<double> inputNodes = generateInputNodes(audio, sample_rate);
    for(double &input: inputNodes){
        std::cout<<input<<std::endl;
    }
    return 0;
}
#endif
