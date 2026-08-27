#include "spiral/gpu.hpp"
#include "spiral/gpu_compute.hpp"
#include "spiral/tensor.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool close(float lhs, float rhs, float tolerance = 2.0e-4F) {
    return std::abs(lhs - rhs) <= tolerance;
}

void require_close(const spiral::Tensor& actual, const spiral::Tensor& expected, float tolerance = 2.0e-4F) {
    assert(actual.shape() == expected.shape());
    assert(actual.data().size() == expected.data().size());
    for (std::size_t i = 0; i < actual.data().size(); ++i) {
        if (!close(actual.data()[i], expected.data()[i], tolerance)) {
            std::cerr << "tensor mismatch at " << i << ": actual=" << actual.data()[i]
                      << " expected=" << expected.data()[i] << '\n';
            assert(false);
        }
    }
}

spiral::Tensor silu_reference(const spiral::Tensor& input) {
    std::vector<float> values;
    values.reserve(input.numel());
    for (const float value : input.data()) {
        values.push_back(value / (1.0F + std::exp(-value)));
    }
    return spiral::Tensor(input.shape(), std::move(values));
}

spiral::Tensor layer_norm_rows_reference(const spiral::Tensor& input, float epsilon) {
    assert(input.rank() >= 1);
    const std::size_t cols = input.shape().back();
    const std::size_t rows = input.numel() / cols;
    std::vector<float> result(input.numel(), 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * cols;
        float mean = 0.0F;
        for (std::size_t col = 0; col < cols; ++col) mean += input.data()[base + col];
        mean /= static_cast<float>(cols);
        float variance = 0.0F;
        for (std::size_t col = 0; col < cols; ++col) {
            const float delta = input.data()[base + col] - mean;
            variance += delta * delta;
        }
        variance /= static_cast<float>(cols);
        const float inverse = 1.0F / std::sqrt(variance + epsilon);
        for (std::size_t col = 0; col < cols; ++col) {
            result[base + col] = (input.data()[base + col] - mean) * inverse;
        }
    }
    return spiral::Tensor(input.shape(), std::move(result));
}

spiral::Tensor softmax_rows_reference(const spiral::Tensor& input) {
    assert(input.rank() >= 1);
    const std::size_t cols = input.shape().back();
    const std::size_t rows = input.numel() / cols;
    std::vector<float> result(input.numel(), 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * cols;
        float maximum = input.data()[base];
        for (std::size_t col = 1; col < cols; ++col) maximum = std::max(maximum, input.data()[base + col]);
        float denominator = 0.0F;
        for (std::size_t col = 0; col < cols; ++col) {
            result[base + col] = std::exp(input.data()[base + col] - maximum);
            denominator += result[base + col];
        }
        for (std::size_t col = 0; col < cols; ++col) result[base + col] /= denominator;
    }
    return spiral::Tensor(input.shape(), std::move(result));
}

} // namespace

int main() {
    using namespace spiral;
    using namespace spiral::gpu;

    assert(gpu_activation_name(GpuActivation::Relu) == "relu");
    assert(gpu_activation_name(GpuActivation::Silu) == "silu");
    assert(execution_backend_name(ExecutionBackend::Cpu) == "cpu");
    assert(execution_backend_name(ExecutionBackend::Gpu) == "gpu");

#ifdef _WIN32
    assert(D3D11ComputeEngine::platform_supported());
    std::string error;
    auto device = D3D11GpuDevice::try_create(&error);
    if (!device) {
        std::cerr << "D3D11 device creation failed: " << error << '\n';
        assert(false);
    }

    D3D11ComputeEngine engine(*device);
    assert(engine.available());

    const Tensor lhs({2, 3}, {
        1.0F, -2.0F, 3.0F,
        4.0F,  0.5F, -1.0F,
    });
    const Tensor rhs({3, 2}, {
        2.0F, -1.0F,
        0.0F,  3.0F,
        -2.0F, 4.0F,
    });
    const Tensor cpu_matmul = lhs.matmul(rhs);

    // Residency contract: intermediates remain on the GPU between matmul and activation.
    auto gpu_lhs = engine.upload(lhs);
    auto gpu_rhs = engine.upload(rhs);
    assert(gpu_lhs.valid() && gpu_rhs.valid());
    assert(gpu_lhs.shape() == lhs.shape());
    auto gpu_product = engine.matmul(gpu_lhs, gpu_rhs);
    auto gpu_relu = engine.activation(gpu_product, GpuActivation::Relu);
    const Tensor resident_result = engine.download(gpu_relu);
    require_close(resident_result, cpu_matmul.relu());

    const auto resident_stats = engine.stats();
    assert(resident_stats.dispatches >= 2);
    assert(resident_stats.uploaded_bytes == (lhs.numel() + rhs.numel()) * sizeof(float));
    assert(resident_stats.downloaded_bytes == resident_result.numel() * sizeof(float));

    // Direct matmul convenience path must match the CPU reference.
    const Tensor gpu_matmul = engine.matmul(lhs, rhs);
    require_close(gpu_matmul, cpu_matmul);

    const Tensor activation_input({2, 4}, {
        -5.0F, -1.0F, 0.0F, 2.0F,
        4.0F, -3.0F, 1.0F, 0.5F,
    });
    require_close(engine.activation(activation_input, GpuActivation::Relu), activation_input.relu());
    require_close(engine.activation(activation_input, GpuActivation::Silu), silu_reference(activation_input), 4.0e-4F);

    const Tensor norm_input({3, 4}, {
        1.0F, 2.0F, 3.0F, 4.0F,
        -2.0F, -1.0F, 2.0F, 5.0F,
        10.0F, 10.0F, 10.0F, 10.0F,
    });
    constexpr float epsilon = 1.0e-5F;
    require_close(
        engine.layer_norm_rows(norm_input, epsilon),
        layer_norm_rows_reference(norm_input, epsilon),
        8.0e-4F);

    const Tensor softmax_input({3, 5}, {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
        100.0F, 101.0F, 99.0F, 98.0F, 102.0F,
        -3.0F, -3.0F, -3.0F, -3.0F, -3.0F,
    });
    const Tensor gpu_softmax = engine.softmax_rows(softmax_input);
    require_close(gpu_softmax, softmax_rows_reference(softmax_input), 5.0e-4F);
    for (std::size_t row = 0; row < 3; ++row) {
        float sum = 0.0F;
        for (std::size_t col = 0; col < 5; ++col) sum += gpu_softmax.data()[row * 5 + col];
        assert(close(sum, 1.0F, 5.0e-4F));
    }

    HybridComputeConfig hybrid_config;
    hybrid_config.cpu.workers = 2;
    hybrid_config.gpu_matmul_threshold_ops = 1;
    HybridComputeBackend hybrid(device.get(), hybrid_config);
    assert(hybrid.gpu_available());
    const Tensor hybrid_product = hybrid.matmul(lhs, rhs);
    require_close(hybrid_product, cpu_matmul);
    assert(hybrid.last_execution_backend() == ExecutionBackend::Gpu);

    // Raise the threshold and prove deterministic CPU fallback through the same backend.
    HybridComputeConfig cpu_pref_config;
    cpu_pref_config.cpu.workers = 2;
    cpu_pref_config.gpu_matmul_threshold_ops = 1'000'000;
    HybridComputeBackend cpu_pref(device.get(), cpu_pref_config);
    require_close(cpu_pref.matmul(lhs, rhs), cpu_matmul);
    assert(cpu_pref.last_execution_backend() == ExecutionBackend::Cpu);

    const auto final_stats = engine.stats();
    assert(final_stats.dispatches >= 7);
    std::cout << "L23 GPU adapter: " << device->capabilities().adapter_name << '\n';
    std::cout << "L23 hardware accelerated: " << (device->capabilities().hardware_accelerated ? "yes" : "no/WARP") << '\n';
    std::cout << "L23 dispatches: " << final_stats.dispatches << '\n';
#else
    assert(!D3D11ComputeEngine::platform_supported());
    assert(!D3D11GpuDevice::platform_supported());

    bool rejected = false;
    try {
        D3D11GpuDevice unsupported;
        (void)unsupported;
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);

    HybridComputeConfig config;
    config.cpu.workers = 2;
    config.gpu_matmul_threshold_ops = 1;
    HybridComputeBackend hybrid(nullptr, config);
    assert(!hybrid.gpu_available());
    const Tensor lhs({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    const Tensor rhs({2, 2}, {2.0F, 0.0F, 1.0F, 2.0F});
    require_close(hybrid.matmul(lhs, rhs), lhs.matmul(rhs));
    assert(hybrid.last_execution_backend() == ExecutionBackend::Cpu);
#endif

    std::cout << "L23 GPU compute regression passed\n";
    return 0;
}
