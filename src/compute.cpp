#include "spiral/compute.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define SPIRAL_COMPUTE_SSE2 1
#include <immintrin.h>
#else
#define SPIRAL_COMPUTE_SSE2 0
#endif

namespace spiral::compute {
namespace {

std::size_t resolved_workers(std::size_t requested) {
    if (requested != 0) return requested;
    const auto detected = static_cast<std::size_t>(std::thread::hardware_concurrency());
    return std::max<std::size_t>(1, detected);
}

bool is_power_of_two(std::size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

void require_matmul_shapes(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.rank() != 2 || rhs.rank() != 2) throw std::invalid_argument("compute matmul requires rank-2 tensors");
    if (lhs.shape()[1] != rhs.shape()[0]) throw std::invalid_argument("compute matmul inner dimension mismatch");
}

void require_quantized_matmul_shapes(
    const precision::QuantizedTensor& lhs,
    const precision::QuantizedTensor& rhs) {
    if (lhs.shape.size() != 2 || rhs.shape.size() != 2) throw std::invalid_argument("quantized matmul requires rank-2 tensors");
    if (lhs.shape[1] != rhs.shape[0]) throw std::invalid_argument("quantized matmul inner dimension mismatch");
    if (lhs.values.size() != lhs.shape[0] * lhs.shape[1] || rhs.values.size() != rhs.shape[0] * rhs.shape[1]) {
        throw std::invalid_argument("quantized matmul storage mismatch");
    }
    if (!(lhs.scale > 0.0F) || !(rhs.scale > 0.0F)) throw std::invalid_argument("quantized matmul requires positive scales");
}

std::int32_t quantized_dot(
    const std::int8_t* lhs,
    const std::int8_t* rhs,
    std::size_t count) {
    std::int32_t sum = 0;
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        sum += static_cast<std::int32_t>(lhs[i]) * static_cast<std::int32_t>(rhs[i]);
        sum += static_cast<std::int32_t>(lhs[i + 1]) * static_cast<std::int32_t>(rhs[i + 1]);
        sum += static_cast<std::int32_t>(lhs[i + 2]) * static_cast<std::int32_t>(rhs[i + 2]);
        sum += static_cast<std::int32_t>(lhs[i + 3]) * static_cast<std::int32_t>(rhs[i + 3]);
    }
    for (; i < count; ++i) sum += static_cast<std::int32_t>(lhs[i]) * static_cast<std::int32_t>(rhs[i]);
    return sum;
}

} // namespace

ThreadPool::ThreadPool(std::size_t worker_count) {
    const std::size_t count = resolved_workers(worker_count);
    workers_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) workers_.emplace_back([this]() { worker_loop(); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [this]() { return stopping_ || !tasks_.empty(); });
            if (stopping_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

ScratchArena::ScratchArena(std::size_t capacity_bytes, std::size_t base_alignment)
    : capacity_(capacity_bytes), base_alignment_(base_alignment) {
    if (capacity_bytes == 0) throw std::invalid_argument("ScratchArena capacity must be non-zero");
    if (!is_power_of_two(base_alignment_) || base_alignment_ < alignof(std::max_align_t)) {
        throw std::invalid_argument("ScratchArena base alignment must be a power of two >= max_align_t");
    }
    data_ = static_cast<std::byte*>(::operator new(capacity_, std::align_val_t{base_alignment_}));
}

ScratchArena::~ScratchArena() {
    if (data_ != nullptr) ::operator delete(data_, std::align_val_t{base_alignment_});
}

void* ScratchArena::allocate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) return data_ + offset_;
    if (!is_power_of_two(alignment) || alignment > base_alignment_) {
        throw std::invalid_argument("ScratchArena allocation alignment is unsupported");
    }
    const std::uintptr_t current = reinterpret_cast<std::uintptr_t>(data_ + offset_);
    const std::uintptr_t aligned = (current + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1U);
    const std::size_t padding = static_cast<std::size_t>(aligned - current);
    if (padding > capacity_ - offset_ || bytes > capacity_ - offset_ - padding) throw std::bad_alloc{};
    offset_ += padding;
    void* result = data_ + offset_;
    offset_ += bytes;
    high_watermark_ = std::max(high_watermark_, offset_);
    return result;
}

std::size_t simd_width() noexcept {
#if SPIRAL_COMPUTE_SSE2
    return 4;
#else
    return 1;
#endif
}

std::string_view simd_backend_name() noexcept {
#if SPIRAL_COMPUTE_SSE2
    return "sse2";
#else
    return "scalar";
#endif
}

float dot_product(std::span<const float> lhs, std::span<const float> rhs) {
    if (lhs.size() != rhs.size()) throw std::invalid_argument("dot_product size mismatch");
    std::size_t i = 0;
    float sum = 0.0F;
#if SPIRAL_COMPUTE_SSE2
    __m128 accumulator = _mm_setzero_ps();
    for (; i + 4 <= lhs.size(); i += 4) {
        const __m128 a = _mm_loadu_ps(lhs.data() + i);
        const __m128 b = _mm_loadu_ps(rhs.data() + i);
        accumulator = _mm_add_ps(accumulator, _mm_mul_ps(a, b));
    }
    alignas(16) float lanes[4];
    _mm_store_ps(lanes, accumulator);
    sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
#endif
    for (; i < lhs.size(); ++i) sum += lhs[i] * rhs[i];
    return sum;
}

Tensor parallel_matmul(const Tensor& lhs, const Tensor& rhs, ThreadPool& pool) {
    require_matmul_shapes(lhs, rhs);
    const std::size_t m = lhs.shape()[0];
    const std::size_t k = lhs.shape()[1];
    const std::size_t n = rhs.shape()[1];
    Tensor out({m, n});
    std::vector<float> rhs_transposed(n * k);
    for (std::size_t inner = 0; inner < k; ++inner) {
        for (std::size_t col = 0; col < n; ++col) rhs_transposed[col * k + inner] = rhs.data()[inner * n + col];
    }
    pool.parallel_for(0, m, 1, [&](std::size_t row) {
        const std::span<const float> lhs_row(lhs.data().data() + row * k, k);
        for (std::size_t col = 0; col < n; ++col) {
            const std::span<const float> rhs_col(rhs_transposed.data() + col * k, k);
            out.data()[row * n + col] = dot_product(lhs_row, rhs_col);
        }
    });
    return out;
}

Tensor quantized_matmul(
    const precision::QuantizedTensor& lhs,
    const precision::QuantizedTensor& rhs,
    ThreadPool* pool) {
    require_quantized_matmul_shapes(lhs, rhs);
    const std::size_t m = lhs.shape[0];
    const std::size_t k = lhs.shape[1];
    const std::size_t n = rhs.shape[1];
    Tensor out({m, n});
    std::vector<std::int8_t> rhs_transposed(n * k);
    for (std::size_t inner = 0; inner < k; ++inner) {
        for (std::size_t col = 0; col < n; ++col) rhs_transposed[col * k + inner] = rhs.values[inner * n + col];
    }
    const float output_scale = lhs.scale * rhs.scale;
    auto compute_row = [&](std::size_t row) {
        const auto* lhs_row = lhs.values.data() + row * k;
        for (std::size_t col = 0; col < n; ++col) {
            const auto* rhs_col = rhs_transposed.data() + col * k;
            out.data()[row * n + col] = static_cast<float>(quantized_dot(lhs_row, rhs_col, k)) * output_scale;
        }
    };
    if (pool != nullptr) pool->parallel_for(0, m, 1, compute_row);
    else for (std::size_t row = 0; row < m; ++row) compute_row(row);
    return out;
}

CpuBackend::CpuBackend(ComputeConfig config)
    : config_(config), pool_(config.workers) {
    name_ = std::string("cpu-") + std::string(simd_backend_name()) + "-threaded";
}

Tensor CpuBackend::matmul(const Tensor& lhs, const Tensor& rhs) {
    require_matmul_shapes(lhs, rhs);
    const long double operations = static_cast<long double>(lhs.shape()[0]) *
        static_cast<long double>(lhs.shape()[1]) * static_cast<long double>(rhs.shape()[1]);
    if (pool_.worker_count() > 1 && operations >= static_cast<long double>(config_.parallel_threshold_ops)) {
        return parallel_matmul(lhs, rhs, pool_);
    }
    return lhs.matmul(rhs);
}

Tensor CpuBackend::matmul_int8(
    const precision::QuantizedTensor& lhs,
    const precision::QuantizedTensor& rhs) {
    return quantized_matmul(lhs, rhs, pool_.worker_count() > 1 ? &pool_ : nullptr);
}

BenchmarkResult benchmark(
    std::string label,
    std::size_t iterations,
    const std::function<void()>& operation) {
    if (iterations == 0) throw std::invalid_argument("benchmark iterations must be non-zero");
    if (!operation) throw std::invalid_argument("benchmark operation must be callable");
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) operation();
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    BenchmarkResult result;
    result.label = std::move(label);
    result.iterations = iterations;
    result.seconds = seconds;
    result.iterations_per_second = seconds > 0.0 ? static_cast<double>(iterations) / seconds : std::numeric_limits<double>::infinity();
    return result;
}

} // namespace spiral::compute
