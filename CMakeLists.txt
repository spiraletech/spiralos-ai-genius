cmake_minimum_required(VERSION 3.25)
project(SpiralOS_AI_Genius VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(spiralos_core STATIC
    src/barcode.cpp
)

target_include_directories(spiralos_core PUBLIC include)
target_compile_features(spiralos_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(spiralos_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(spiralos_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_executable(spiralos_barcode_test tests/barcode.cpp)
target_link_libraries(spiralos_barcode_test PRIVATE spiralos_core)
add_test(NAME spiralos.barcode COMMAND spiralos_barcode_test)
