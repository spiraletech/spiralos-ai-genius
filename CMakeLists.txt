cmake_minimum_required(VERSION 3.24)
project(spiralos_ai_genius VERSION 0.0.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(spiral_core STATIC
    src/tensor.cpp
    src/tokenizer.cpp
    src/random.cpp
    src/nn.cpp
    src/tensor_ops.cpp
    src/attention.cpp
)

target_include_directories(spiral_core PUBLIC include)
target_compile_features(spiral_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(spiral_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(spiral_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(spiral_genius apps/genius/main.cpp)
target_link_libraries(spiral_genius PRIVATE spiral_core)

enable_testing()

add_executable(spiral_core_tests tests/tensor_tests.cpp)
target_link_libraries(spiral_core_tests PRIVATE spiral_core)
add_test(NAME spiral_core_tests COMMAND spiral_core_tests)

add_executable(spiral_nn_tests tests/nn_tests.cpp)
target_link_libraries(spiral_nn_tests PRIVATE spiral_core)
add_test(NAME spiral_nn_tests COMMAND spiral_nn_tests)

add_executable(spiral_attention_tests tests/attention_tests.cpp)
target_link_libraries(spiral_attention_tests PRIVATE spiral_core)
add_test(NAME spiral_attention_tests COMMAND spiral_attention_tests)
