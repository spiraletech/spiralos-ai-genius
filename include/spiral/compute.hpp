#pragma once

#include "spiral/precision.hpp"
#include "spiral/tensor.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <new>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace spiral::compute {

class ThreadPool final {
public:
    explicit ThreadPool(std::size_t worker_count = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    [[nodiscard]] std::size_t worker_count() const noexcept { return workers_.size(); }

    template <typename Fn>
    auto submit(Fn&& fn) -> std::future<std::invoke_result_t<Fn>> {
        using Result = std::invoke_result_t<Fn>;
        auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<Fn>(fn));
        auto future = task->get_future();
        {
            std::lock_guard lock(mutex_);
            if (stopping_) throw std::runtime_error("ThreadPool is stopping");
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return future;
    }

    template <typename Fn>
    void parallel_for(std::size_t begin, std::size_t end, std::size_t grain, Fn&& fn) {
        if (end <= begin) return;
        if (grain == 0) grain = 1;
        const std::size_t total = end - begin;
        const std::size_t chunks = (total + grain - 1) / grain;
        if (workers_.size() <= 1 || chunks <= 1) {
            for (std::size_t i = begin; i < end; ++i) fn(i);
            return;
        }

        const std::size_t task_count = std::min(chunks, workers_.size());
        auto next = std::make_shared<std::atomic<std::size_t>>(begin);
        auto shared_fn = std::make_shared<std::decay_t<Fn>>(std::forward<Fn>(fn));
        std::vector<std::future<void>> futures;
        futures.reserve(task_count);
        for (std::size_t task = 0; task < task_count; ++task) {
            futures.push_back(submit([next, shared_fn, end, grain]() {
                while (true) {
                    const std::size_t chunk_begin = next->fetch_add(grain, std::memory_order_relaxed);
                    if (chunk_begin >= end) break;
                    const std::size_t chunk_end = std::min(end, chunk_begin + grain);
                    for (std::size_t i = chunk_begin; i < chunk_end; ++i) (*shared_fn)(i);
                }
            }));
        }
        for (auto& future : futures) future.get();
    }

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_ = false;
};

class ScratchArena final {
public:
    explicit ScratchArena(std::size_t capacity_bytes, std::size_t base_alignment = 64);
    ~ScratchArena();

    ScratchArena(const ScratchArena&) = delete;
    ScratchArena& operator=(const ScratchArena&) = delete;

    void reset() noexcept { offset_ = 0; }
    [[nodiscard]] void* allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t));

    template <typename T>
    [[nodiscard]] std::span<T> allocate_array(std::size_t count) {
        if (count > static_cast<std::size_t>(-1) / sizeof(T)) throw std::bad_alloc{};
        auto* ptr = static_cast<T*>(allocate(count * sizeof(T), alignof(T)));
        return {ptr, count};
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t used() const noexcept { return offset_; }
    [[nodiscard]] std::size_t high_watermark() const noexcept { return high_watermark_; }

private:
    std::byte* data_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t offset_ = 0;
    std::size_t high_watermark_ = 0;
    std::size_t base_alignment_ = 64;
};

[[nodiscard]] std::size_t simd_width() noexcept;
[[nodiscard]] std::string_view simd_backend_name() noexcept;
[[nodiscard]] float dot_product(std::span<const float> lhs, std::span<const float> rhs);
[[nodiscard]] Tensor parallel_matmul(const Tensor& lhs, const Tensor& rhs, ThreadPool& pool);
[[nodiscard]] Tensor quantized_matmul(
    const precision::QuantizedTensor& lhs,
    const precision::QuantizedTensor& rhs,
    ThreadPool* pool = nullptr);

struct ComputeConfig {
    std::size_t workers = 0;
    std::size_t parallel_threshold_ops = 32 * 1024;
};

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual std::size_t worker_count() const noexcept = 0;
    [[nodiscard]] virtual Tensor matmul(const Tensor& lhs, const Tensor& rhs) = 0;
    [[nodiscard]] virtual Tensor matmul_int8(
        const precision::QuantizedTensor& lhs,
        const precision::QuantizedTensor& rhs) = 0;
};

class CpuBackend final : public ComputeBackend {
public:
    explicit CpuBackend(ComputeConfig config = {});

    [[nodiscard]] std::string_view name() const noexcept override { return name_; }
    [[nodiscard]] std::size_t worker_count() const noexcept override { return pool_.worker_count(); }
    [[nodiscard]] Tensor matmul(const Tensor& lhs, const Tensor& rhs) override;
    [[nodiscard]] Tensor matmul_int8(
        const precision::QuantizedTensor& lhs,
        const precision::QuantizedTensor& rhs) override;

    [[nodiscard]] ThreadPool& pool() noexcept { return pool_; }

private:
    ComputeConfig config_;
    ThreadPool pool_;
    std::string name_;
};

struct BenchmarkResult {
    std::string label;
    std::size_t iterations = 0;
    double seconds = 0.0;
    double iterations_per_second = 0.0;
};

[[nodiscard]] BenchmarkResult benchmark(
    std::string label,
    std::size_t iterations,
    const std::function<void()>& operation);

} // namespace spiral::compute
