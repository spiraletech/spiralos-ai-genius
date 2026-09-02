cmake_minimum_required(VERSION 3.24)
project(spiralos_ai_genius VERSION 0.0.33 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_library(spiral_core STATIC
    src/gguf.cpp
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
    src/multimodal.cpp
    src/flow.cpp
    src/latent_transformer.cpp
    src/scale.cpp
    src/precision.cpp
    src/data.cpp
    src/audio.cpp
    src/dsp.cpp
    src/temporal.cpp
    src/temporal_generation.cpp
    src/media_generation.cpp
    src/media_flow.cpp
    src/compute.cpp
    src/units.cpp
    src/device.cpp
    src/renderer.cpp
    src/raster.cpp
    src/visual_self_test.cpp
    src/host_bridge.cpp
    src/gpu_compute.cpp
    src/native_host.cpp
    src/openai_backend.cpp
    src/genius_kernel.cpp
    src/organic_ai.cpp
    src/genius_shell.cpp
    src/xenon_os.cpp
    src/ether_ai.cpp
)

target_include_directories(spiral_core PUBLIC include)
target_compile_features(spiral_core PUBLIC cxx_std_20)
target_link_libraries(spiral_core PUBLIC Threads::Threads)

if(WIN32)
    target_link_libraries(spiral_core PUBLIC d3d11 dxgi d3dcompiler winhttp)
endif()

if(MSVC)
    target_compile_options(spiral_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(spiral_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(spiral_genius apps/genius/main.cpp)
target_link_libraries(spiral_genius PRIVATE spiral_core)

add_executable(spiral_benchmark apps/benchmark/main.cpp)
target_link_libraries(spiral_benchmark PRIVATE spiral_core)

add_executable(SpiralGGUFInspect apps/gguf_inspect/main.cpp)
target_link_libraries(SpiralGGUFInspect PRIVATE spiral_core)

add_executable(SpiralAIShell apps/ai_shell/main.cpp)
target_link_libraries(SpiralAIShell PRIVATE spiral_core)

# HAKUI L9 process boundary. This executable hosts the real
# spiral::ether_ai::Runtime with HostKind::Hakui and exposes only a loopback,
# read-context/request-response protocol. HAKUI remains a separate authority
# and never links the model/runtime implementation into gameplay state.
add_executable(SpiralHakuiCortex apps/hakui_cortex/main.cpp)
target_link_libraries(SpiralHakuiCortex PRIVATE spiral_core)
if(WIN32)
    target_link_libraries(SpiralHakuiCortex PRIVATE ws2_32)
endif()

if(WIN32)
    add_executable(spiral_ether_ai WIN32
        apps/ether_ai/main.cpp
    )
    target_link_libraries(spiral_ether_ai PRIVATE spiral_core d2d1 dwrite dwmapi user32 gdi32 shell32 ole32)
    target_compile_definitions(spiral_ether_ai PRIVATE UNICODE _UNICODE WIN32_LEAN_AND_MEAN)
    if(MSVC)
        target_compile_options(spiral_ether_ai PRIVATE /W4 /permissive- /utf-8)
    endif()
endif()

enable_testing()

add_executable(spiral_gguf_tests tests/gguf_tests.cpp)
target_link_libraries(spiral_gguf_tests PRIVATE spiral_core)
add_test(NAME spiral_gguf_tests COMMAND spiral_gguf_tests)

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

add_executable(spiral_multimodal_tests tests/multimodal_tests.cpp)
target_link_libraries(spiral_multimodal_tests PRIVATE spiral_core)
add_test(NAME spiral_multimodal_tests COMMAND spiral_multimodal_tests)

add_executable(spiral_flow_tests tests/flow_tests.cpp)
target_link_libraries(spiral_flow_tests PRIVATE spiral_core)
add_test(NAME spiral_flow_tests COMMAND spiral_flow_tests)

add_executable(spiral_latent_transformer_tests tests/latent_transformer_tests.cpp)
target_link_libraries(spiral_latent_transformer_tests PRIVATE spiral_core)
add_test(NAME spiral_latent_transformer_tests COMMAND spiral_latent_transformer_tests)

add_executable(spiral_scale_tests tests/scale_tests.cpp)
target_link_libraries(spiral_scale_tests PRIVATE spiral_core)
add_test(NAME spiral_scale_tests COMMAND spiral_scale_tests)

add_executable(spiral_data_audio_tests tests/data_audio_tests.cpp)
target_link_libraries(spiral_data_audio_tests PRIVATE spiral_core)
add_test(NAME spiral_data_audio_tests COMMAND spiral_data_audio_tests)

add_executable(spiral_temporal_tests tests/temporal_tests.cpp)
target_link_libraries(spiral_temporal_tests PRIVATE spiral_core)
add_test(NAME spiral_temporal_tests COMMAND spiral_temporal_tests)

add_executable(spiral_temporal_generation_tests tests/temporal_generation_tests.cpp)
target_link_libraries(spiral_temporal_generation_tests PRIVATE spiral_core)
add_test(NAME spiral_temporal_generation_tests COMMAND spiral_temporal_generation_tests)

add_executable(spiral_media_generation_tests tests/media_generation_tests.cpp)
target_link_libraries(spiral_media_generation_tests PRIVATE spiral_core)
add_test(NAME spiral_media_generation_tests COMMAND spiral_media_generation_tests)

add_executable(spiral_media_flow_tests tests/media_flow_tests.cpp)
target_link_libraries(spiral_media_flow_tests PRIVATE spiral_core)
add_test(NAME spiral_media_flow_tests COMMAND spiral_media_flow_tests)

add_executable(spiral_compute_tests tests/compute_tests.cpp)
target_link_libraries(spiral_compute_tests PRIVATE spiral_core)
add_test(NAME spiral_compute_tests COMMAND spiral_compute_tests)

add_executable(spiral_units_tests tests/units_tests.cpp)
target_link_libraries(spiral_units_tests PRIVATE spiral_core)
add_test(NAME spiral_units_tests COMMAND spiral_units_tests)

add_executable(spiral_renderer_device_tests tests/renderer_device_tests.cpp)
target_link_libraries(spiral_renderer_device_tests PRIVATE spiral_core)
add_test(NAME spiral_renderer_device_tests COMMAND spiral_renderer_device_tests)

add_executable(spiral_visual_self_test_tests tests/visual_self_test_tests.cpp)
target_link_libraries(spiral_visual_self_test_tests PRIVATE spiral_core)
add_test(NAME spiral_visual_self_test_tests COMMAND spiral_visual_self_test_tests)

add_executable(spiral_native_gpu_host_tests tests/native_gpu_host_tests.cpp)
target_link_libraries(spiral_native_gpu_host_tests PRIVATE spiral_core)
add_test(NAME spiral_native_gpu_host_tests COMMAND spiral_native_gpu_host_tests)

add_executable(spiral_gpu_compute_tests tests/gpu_compute_tests.cpp)
target_link_libraries(spiral_gpu_compute_tests PRIVATE spiral_core)
add_test(NAME spiral_gpu_compute_tests COMMAND spiral_gpu_compute_tests)

add_executable(spiral_genius_shell_tests tests/genius_shell_tests.cpp)
target_link_libraries(spiral_genius_shell_tests PRIVATE spiral_core)
add_test(NAME spiral_genius_shell_tests COMMAND spiral_genius_shell_tests)

add_executable(spiral_performance_proof_tests tests/performance_proof_tests.cpp)
target_link_libraries(spiral_performance_proof_tests PRIVATE spiral_core)
add_test(NAME spiral_performance_proof_tests COMMAND spiral_performance_proof_tests)

add_executable(spiral_ether_ai_tests tests/ether_ai_tests.cpp)
target_link_libraries(spiral_ether_ai_tests PRIVATE spiral_core)
add_test(NAME spiral_ether_ai_tests COMMAND spiral_ether_ai_tests)

add_executable(spiral_organic_ai_tests tests/organic_ai_tests.cpp)
target_link_libraries(spiral_organic_ai_tests PRIVATE spiral_core)
add_test(NAME spiral_organic_ai_tests COMMAND spiral_organic_ai_tests)

add_executable(spiral_genius_kernel_tests tests/genius_kernel_tests.cpp)
target_link_libraries(spiral_genius_kernel_tests PRIVATE spiral_core)
add_test(NAME spiral_genius_kernel_tests COMMAND spiral_genius_kernel_tests)

add_executable(spiral_xenon_os_tests tests/xenon_os_tests.cpp)
target_link_libraries(spiral_xenon_os_tests PRIVATE spiral_core)
add_test(NAME spiral_xenon_os_tests COMMAND spiral_xenon_os_tests)
