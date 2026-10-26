cmake_minimum_required(VERSION 4.4.2)

project(VoiceDecode LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3 -march=native -flto)

find_library(MPG123_LIBRARY mpg123 REQUIRED)

add_library(classifier_lib STATIC
    src/classifier.cpp
    src/dataset.cpp
    src/decode.cpp
    src/preprocessor.cpp
)

target_link_libraries(classifier_lib PUBLIC ${MPG123_LIBRARY})
target_include_directories(classifier_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(trainCommands src/main.cpp)
target_compile_definitions(trainCommands PRIVATE MODEL="commands" DATA_DIR="${DATA_DIR}")
target_link_libraries(trainCommands PRIVATE classifier_lib)

add_executable(trainWakeup src/main.cpp)
target_compile_definitions(trainWakeup PRIVATE MODEL="wakeup" DATA_DIR="${DATA_DIR}")
target_link_libraries(trainWakeup PRIVATE classifier_lib)

add_executable(predict src/predict.cpp)
target_compile_definitions(predict PRIVATE DATA_DIR=${DATA_DIR})
target_link_libraries(predict PRIVATE classifier_lib)
