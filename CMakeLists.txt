cmake_minimum_required(VERSION 3.24)
project(spiralos_ai_genius VERSION 0.0.9 LANGUAGES CXX)

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
    src/model.cpp
    src/train_ops.cpp
    src/train_attention.cpp
    src/train_model_backward.cpp
    src/train.cpp
    src/generate.cpp
    src/runtime.cpp
    src/memory.cpp
    src/tools.cpp
    src/agent.cpp
    src/vision.cpp
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

add_executable(spiral_model_tests tests/model_tests.cpp)
target_link_libraries(spiral_model_tests PRIVATE spiral_core)
add_test(NAME spiral_model_tests COMMAND spiral_model_tests)

add_executable(spiral_train_tests tests/train_tests.cpp)
target_link_libraries(spiral_train_tests PRIVATE spiral_core)
add_test(NAME spiral_train_tests COMMAND spiral_train_tests)

add_executable(spiral_generate_tests tests/generate_tests.cpp)
target_link_libraries(spiral_generate_tests PRIVATE spiral_core)
add_test(NAME spiral_generate_tests COMMAND spiral_generate_tests)

add_executable(spiral_runtime_tests tests/runtime_tests.cpp)
target_link_libraries(spiral_runtime_tests PRIVATE spiral_core)
add_test(NAME spiral_runtime_tests COMMAND spiral_runtime_tests)

add_executable(spiral_agent_tests tests/agent_tests.cpp)
target_link_libraries(spiral_agent_tests PRIVATE spiral_core)
add_test(NAME spiral_agent_tests COMMAND spiral_agent_tests)

add_executable(spiral_vision_tests tests/vision_tests.cpp)
target_link_libraries(spiral_vision_tests PRIVATE spiral_core)
add_test(NAME spiral_vision_tests COMMAND spiral_vision_tests)
