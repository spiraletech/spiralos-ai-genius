#include "spiral/compute.hpp"
#include "spiral/gpu.hpp"
#include "spiral/gpu_compute.hpp"
#include "spiral/tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

spiral::Tensor make_matrix(std::size_t rows, std::size_t cols, float phase) {
    std::vector<float> values(rows * cols);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const float x = static_cast<float>((row * 131U + col * 17U) % 997U) * 0.013F + phase;
            values[row * cols + col] = 0.55F * std::sin(x) + 0.35F * std::cos(x * 0.37F);
        }
    }
    return spiral::Tensor({rows, cols}, std::move(values));
}

spiral::Tensor silu_cpu(const spiral::Tensor& input) {
    std::vector<float> values(input.data().size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const float value = input.data()[i];
        values[i] = value / (1.0F + std::exp(-value));
    }
    return spiral::Tensor(input.shape(), std::move(values));
}

spiral::Tensor layer_norm_cpu(const spiral::Tensor& input, float epsilon = 1.0e-5F) {
    if (input.rank() == 0 || input.shape().back() == 0) throw std::invalid_argument("layer norm input must be non-empty");
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
            const float delta = input.data()[base + col] - mean;
            variance += delta * delta;
        }
        variance /= static_cast<float>(cols);
        const float inv = 1.0F / std::sqrt(variance + epsilon);
        for (std::size_t col = 0; col < cols; ++col) values[base + col] = (input.data()[base + col] - mean) * inv;
    }
    return spiral::Tensor(input.shape(), std::move(values));
}

float max_abs_error(const spiral::Tensor& lhs, const spiral::Tensor& rhs) {
    if (lhs.shape() != rhs.shape()) return std::numeric_limits<float>::infinity();
    float maximum = 0.0F;
    for (std::size_t i = 0; i < lhs.data().size(); ++i) {
        maximum = std::max(maximum, std::abs(lhs.data()[i] - rhs.data()[i]));
    }
    return maximum;
}

template <typename Fn>
double time_ms(std::size_t iterations, Fn&& fn) {
    const auto begin = Clock::now();
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) fn();
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count() / static_cast<double>(iterations);
}

std::string escape_json(std::string_view value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

struct Result {
    std::size_t size = 0;
    double cpu_matmul_ms = 0.0;
    double gpu_cold_matmul_ms = 0.0;
    double cpu_chain_ms = 0.0;
    double gpu_resident_chain_ms = 0.0;
    double matmul_ratio_cpu_over_gpu = 0.0;
    double chain_ratio_cpu_over_gpu = 0.0;
    float matmul_max_abs_error = 0.0F;
    float chain_max_abs_error = 0.0F;
    std::uint64_t resident_dispatches = 0;
    std::uint64_t resident_uploaded_bytes = 0;
    std::uint64_t resident_downloaded_bytes = 0;
};

Result run_case(
    spiral::compute::CpuBackend& cpu,
    spiral::gpu::D3D11ComputeEngine& gpu,
    std::size_t size,
    std::size_t iterations) {
    const auto lhs = make_matrix(size, size, 0.13F);
    const auto rhs = make_matrix(size, size, 0.71F);

    const auto cpu_reference = cpu.matmul(lhs, rhs);
    const auto gpu_reference = gpu.matmul(lhs, rhs);

    Result result;
    result.size = size;
    result.matmul_max_abs_error = max_abs_error(cpu_reference, gpu_reference);

    result.cpu_matmul_ms = time_ms(iterations, [&] {
        auto output = cpu.matmul(lhs, rhs);
        volatile float sink = output.data().front();
        (void)sink;
    });

    result.gpu_cold_matmul_ms = time_ms(iterations, [&] {
        auto output = gpu.matmul(lhs, rhs);
        volatile float sink = output.data().front();
        (void)sink;
    });

    const auto cpu_chain_reference = layer_norm_cpu(silu_cpu(cpu_reference));
    result.cpu_chain_ms = time_ms(iterations, [&] {
        auto mm = cpu.matmul(lhs, rhs);
        auto activated = silu_cpu(mm);
        auto normalized = layer_norm_cpu(activated);
        volatile float sink = normalized.data().front();
        (void)sink;
    });

    auto gpu_lhs = gpu.upload(lhs);
    auto gpu_rhs = gpu.upload(rhs);
    const auto stats_before = gpu.stats();
    spiral::Tensor gpu_chain_reference;
    result.gpu_resident_chain_ms = time_ms(iterations, [&] {
        auto mm = gpu.matmul(gpu_lhs, gpu_rhs);
        auto activated = gpu.activation(mm, spiral::gpu::GpuActivation::Silu);
        auto normalized = gpu.layer_norm_rows(activated);
        gpu_chain_reference = gpu.download(normalized);
    });
    const auto stats_after = gpu.stats();
    result.chain_max_abs_error = max_abs_error(cpu_chain_reference, gpu_chain_reference);
    result.resident_dispatches = stats_after.dispatches - stats_before.dispatches;
    result.resident_uploaded_bytes = stats_after.uploaded_bytes - stats_before.uploaded_bytes;
    result.resident_downloaded_bytes = stats_after.downloaded_bytes - stats_before.downloaded_bytes;

    if (result.gpu_cold_matmul_ms > 0.0) result.matmul_ratio_cpu_over_gpu = result.cpu_matmul_ms / result.gpu_cold_matmul_ms;
    if (result.gpu_resident_chain_ms > 0.0) result.chain_ratio_cpu_over_gpu = result.cpu_chain_ms / result.gpu_resident_chain_ms;
    return result;
}

std::string render_json(
    bool supported,
    const spiral::gpu::NativeGpuCapabilities* capabilities,
    std::size_t cpu_workers,
    const std::vector<Result>& results,
    std::string_view error) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"schema\": \"spiral-performance-proof-v1\",\n";
    out << "  \"gpu_supported\": " << (supported ? "true" : "false") << ",\n";
    if (capabilities != nullptr) {
        out << "  \"hardware_accelerated\": " << (capabilities->hardware_accelerated ? "true" : "false") << ",\n";
        out << "  \"adapter\": \"" << escape_json(capabilities->adapter_name) << "\",\n";
        out << "  \"feature_level\": \"" << escape_json(capabilities->feature_level) << "\",\n";
    } else {
        out << "  \"hardware_accelerated\": false,\n";
        out << "  \"adapter\": \"\",\n";
        out << "  \"feature_level\": \"\",\n";
    }
    out << "  \"cpu_workers\": " << cpu_workers << ",\n";
    out << "  \"error\": \"" << escape_json(error) << "\",\n";
    out << "  \"claim_policy\": \"speedup is claimable only when hardware_accelerated=true\",\n";
    out << "  \"cases\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\n";
        out << "      \"matrix_size\": " << r.size << ",\n";
        out << "      \"cpu_matmul_ms\": " << r.cpu_matmul_ms << ",\n";
        out << "      \"gpu_cold_matmul_ms\": " << r.gpu_cold_matmul_ms << ",\n";
        out << "      \"cpu_chain_ms\": " << r.cpu_chain_ms << ",\n";
        out << "      \"gpu_resident_chain_ms\": " << r.gpu_resident_chain_ms << ",\n";
        out << "      \"cpu_over_gpu_matmul_ratio\": " << r.matmul_ratio_cpu_over_gpu << ",\n";
        out << "      \"cpu_over_gpu_chain_ratio\": " << r.chain_ratio_cpu_over_gpu << ",\n";
        out << "      \"matmul_max_abs_error\": " << r.matmul_max_abs_error << ",\n";
        out << "      \"chain_max_abs_error\": " << r.chain_max_abs_error << ",\n";
        out << "      \"resident_dispatches\": " << r.resident_dispatches << ",\n";
        out << "      \"resident_uploaded_bytes\": " << r.resident_uploaded_bytes << ",\n";
        out << "      \"resident_downloaded_bytes\": " << r.resident_downloaded_bytes << "\n";
        out << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    std::string output_path;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string_view(argv[i]) == "--output") output_path = argv[++i];
    }

    spiral::compute::CpuBackend cpu;
    std::string gpu_error;
    auto device = spiral::gpu::D3D11GpuDevice::try_create(&gpu_error);
    if (device == nullptr) {
        const std::string json = render_json(false, nullptr, cpu.worker_count(), {}, gpu_error);
        std::cout << json;
        if (!output_path.empty()) {
            std::ofstream file(output_path, std::ios::trunc);
            file << json;
        }
        return 0;
    }

    const auto capabilities = device->capabilities();
    spiral::gpu::D3D11ComputeEngine gpu(*device);
    std::vector<Result> results;
    for (const std::size_t size : {64U, 128U, 192U}) {
        results.push_back(run_case(cpu, gpu, size, 3));
    }

    const std::string json = render_json(true, &capabilities, cpu.worker_count(), results, {});
    std::cout << json;
    if (!output_path.empty()) {
        std::ofstream file(output_path, std::ios::trunc);
        file << json;
        if (!file) {
            std::cerr << "failed to write benchmark output: " << output_path << '\n';
            return 2;
        }
    }
    return 0;
}
