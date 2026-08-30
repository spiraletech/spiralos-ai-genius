#include "spiral/compute.hpp"
#include "spiral/gpu.hpp"
#include "spiral/gpu_compute.hpp"
#include "spiral/tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace {

spiral::Tensor make_matrix(std::size_t rows, std::size_t cols, float phase) {
    std::vector<float> values(rows * cols);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const float x = static_cast<float>((i * 37U) % 503U) * 0.021F + phase;
        values[i] = 0.6F * std::sin(x) + 0.25F * std::cos(0.43F * x);
    }
    return spiral::Tensor({rows, cols}, std::move(values));
}

spiral::Tensor silu_cpu(const spiral::Tensor& input) {
    std::vector<float> values(input.data().size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const float v = input.data()[i];
        values[i] = v / (1.0F + std::exp(-v));
    }
    return spiral::Tensor(input.shape(), std::move(values));
}

spiral::Tensor layer_norm_cpu(const spiral::Tensor& input) {
    const std::size_t cols = input.shape().back();
    const std::size_t rows = input.numel() / cols;
    std::vector<float> values(input.data().size());
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * cols;
        float mean = 0.0F;
        for (std::size_t col = 0; col < cols; ++col) mean += input.data()[base + col];
        mean /= static_cast<float>(cols);
        float variance = 0.0F;
        for (std::size_t col = 0; col < cols; ++col) {
            const float d = input.data()[base + col] - mean;
            variance += d * d;
        }
        variance /= static_cast<float>(cols);
        const float inv = 1.0F / std::sqrt(variance + 1.0e-5F);
        for (std::size_t col = 0; col < cols; ++col) values[base + col] = (input.data()[base + col] - mean) * inv;
    }
    return spiral::Tensor(input.shape(), std::move(values));
}

float max_abs_error(const spiral::Tensor& lhs, const spiral::Tensor& rhs) {
    assert(lhs.shape() == rhs.shape());
    float result = 0.0F;
    for (std::size_t i = 0; i < lhs.data().size(); ++i) result = std::max(result, std::abs(lhs.data()[i] - rhs.data()[i]));
    return result;
}

} // namespace

int main() {
    using namespace spiral;

#ifndef _WIN32
    std::string error;
    auto device = gpu::D3D11GpuDevice::try_create(&error);
    assert(device == nullptr);
    assert(!gpu::D3D11ComputeEngine::platform_supported());
    std::cout << "L24 performance proof: explicit non-Windows fallback passed\n";
    return 0;
#else
    std::string error;
    auto device = gpu::D3D11GpuDevice::try_create(&error);
    assert(device != nullptr);
    gpu::D3D11ComputeEngine engine(*device);
    compute::CpuBackend cpu;

    const auto lhs = make_matrix(64, 64, 0.2F);
    const auto rhs = make_matrix(64, 64, 0.7F);
    const auto cpu_mm = cpu.matmul(lhs, rhs);
    const auto gpu_mm = engine.matmul(lhs, rhs);
    assert(max_abs_error(cpu_mm, gpu_mm) < 2.0e-4F);

    auto gpu_lhs = engine.upload(lhs);
    auto gpu_rhs = engine.upload(rhs);
    const auto before = engine.stats();
    auto mm = engine.matmul(gpu_lhs, gpu_rhs);
    auto activated = engine.activation(mm, gpu::GpuActivation::Silu);
    auto normalized = engine.layer_norm_rows(activated);
    const auto gpu_chain = engine.download(normalized);
    const auto after = engine.stats();

    const auto cpu_chain = layer_norm_cpu(silu_cpu(cpu_mm));
    assert(max_abs_error(cpu_chain, gpu_chain) < 5.0e-4F);
    assert(after.dispatches - before.dispatches == 3);
    assert(after.uploaded_bytes == before.uploaded_bytes); // no hidden upload between resident GPU ops
    assert(after.downloaded_bytes - before.downloaded_bytes == gpu_chain.numel() * sizeof(float));

    const auto caps = device->capabilities();
    assert(caps.available);
    std::cout << "L24 GPU adapter: " << caps.adapter_name << '\n';
    std::cout << "L24 hardware accelerated: " << (caps.hardware_accelerated ? "yes" : "no/WARP") << '\n';
    std::cout << "L24 resident chain parity passed\n";
    return 0;
#endif
}
