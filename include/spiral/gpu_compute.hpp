#pragma once

#include "spiral/compute.hpp"
#include "spiral/gpu.hpp"
#include "spiral/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::gpu {

enum class GpuActivation {
    Relu,
    Silu,
};

class GpuTensor final {
public:
    GpuTensor() = default;
    ~GpuTensor();

    GpuTensor(const GpuTensor&) = delete;
    GpuTensor& operator=(const GpuTensor&) = delete;
    GpuTensor(GpuTensor&& other) noexcept;
    GpuTensor& operator=(GpuTensor&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return device_ != nullptr && handle_ != 0; }
    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] std::size_t rank() const noexcept { return shape_.size(); }
    [[nodiscard]] std::size_t numel() const noexcept;
    [[nodiscard]] device::BufferHandle handle() const noexcept { return handle_; }

private:
    GpuTensor(D3D11GpuDevice& device, device::BufferHandle handle, std::vector<std::size_t> shape);
    void reset() noexcept;

    D3D11GpuDevice* device_ = nullptr;
    device::BufferHandle handle_ = 0;
    std::vector<std::size_t> shape_;

    friend class D3D11ComputeEngine;
};

struct GpuComputeStats {
    std::uint64_t dispatches = 0;
    std::uint64_t uploaded_bytes = 0;
    std::uint64_t downloaded_bytes = 0;
};

class D3D11ComputeEngine final {
public:
    explicit D3D11ComputeEngine(D3D11GpuDevice& device);
    ~D3D11ComputeEngine();

    D3D11ComputeEngine(const D3D11ComputeEngine&) = delete;
    D3D11ComputeEngine& operator=(const D3D11ComputeEngine&) = delete;

    [[nodiscard]] static bool platform_supported() noexcept;
    [[nodiscard]] bool available() const noexcept;
    [[nodiscard]] std::string_view backend_name() const noexcept { return "d3d11-compute"; }

    [[nodiscard]] GpuTensor upload(const Tensor& tensor);
    [[nodiscard]] Tensor download(const GpuTensor& tensor) const;
    [[nodiscard]] GpuTensor zeros(std::vector<std::size_t> shape);

    [[nodiscard]] GpuTensor matmul(const GpuTensor& lhs, const GpuTensor& rhs);
    [[nodiscard]] GpuTensor activation(const GpuTensor& input, GpuActivation activation);
    [[nodiscard]] GpuTensor layer_norm_rows(const GpuTensor& input, float epsilon = 1.0e-5F);
    [[nodiscard]] GpuTensor softmax_rows(const GpuTensor& input);

    [[nodiscard]] Tensor matmul(const Tensor& lhs, const Tensor& rhs);
    [[nodiscard]] Tensor activation(const Tensor& input, GpuActivation activation_kind);
    [[nodiscard]] Tensor layer_norm_rows(const Tensor& input, float epsilon = 1.0e-5F);
    [[nodiscard]] Tensor softmax_rows(const Tensor& input);

    [[nodiscard]] GpuComputeStats stats() const noexcept;

private:
    struct Impl;
    D3D11GpuDevice& device_;
    std::unique_ptr<Impl> impl_;
};

enum class ExecutionBackend {
    Cpu,
    Gpu,
};

struct HybridComputeConfig {
    compute::ComputeConfig cpu{};
    std::size_t gpu_matmul_threshold_ops = 64 * 1024;
};

class HybridComputeBackend final : public compute::ComputeBackend {
public:
    HybridComputeBackend(D3D11GpuDevice* gpu_device, HybridComputeConfig config = {});

    [[nodiscard]] std::string_view name() const noexcept override { return name_; }
    [[nodiscard]] std::size_t worker_count() const noexcept override { return cpu_.worker_count(); }
    [[nodiscard]] Tensor matmul(const Tensor& lhs, const Tensor& rhs) override;
    [[nodiscard]] Tensor matmul_int8(
        const precision::QuantizedTensor& lhs,
        const precision::QuantizedTensor& rhs) override;

    [[nodiscard]] bool gpu_available() const noexcept { return gpu_ != nullptr; }
    [[nodiscard]] ExecutionBackend last_execution_backend() const noexcept { return last_backend_; }
    [[nodiscard]] D3D11ComputeEngine* gpu_engine() noexcept { return gpu_.get(); }

private:
    [[nodiscard]] bool should_use_gpu_matmul(const Tensor& lhs, const Tensor& rhs) const noexcept;

    HybridComputeConfig config_;
    compute::CpuBackend cpu_;
    std::unique_ptr<D3D11ComputeEngine> gpu_;
    ExecutionBackend last_backend_ = ExecutionBackend::Cpu;
    std::string name_;
};

[[nodiscard]] std::string_view execution_backend_name(ExecutionBackend backend) noexcept;
[[nodiscard]] std::string_view gpu_activation_name(GpuActivation activation) noexcept;

} // namespace spiral::gpu
