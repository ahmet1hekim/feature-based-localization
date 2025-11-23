cmake_minimum_required(VERSION 3.28)
project(featurebasedlocalization LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.0.2
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM)
FetchContent_MakeAvailable(SFML)

set(OpenCV_DIR /usr/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

# You don't need Boost libraries — just include headers
add_executable(main src/main.cpp)
target_compile_features(main PRIVATE cxx_std_17)
target_compile_definitions(main PRIVATE ASSETS_DIR=\"${CMAKE_SOURCE_DIR}/assets\")

target_include_directories(main PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    /usr/include        # boost headers are here
)

target_link_libraries(main PRIVATE
    SFML::Graphics
    ${OpenCV_LIBS}
)
