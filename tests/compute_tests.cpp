#include "spiral/compute.hpp"
#include "spiral/precision.hpp"
#include "spiral/tensor.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace {

bool tensors_close(const spiral::Tensor& lhs, const spiral::Tensor& rhs, float tolerance) {
    if (lhs.shape() != rhs.shape()) return false;
    for (std::size_t i = 0; i < lhs.numel(); ++i) {
        if (std::abs(lhs.data()[i] - rhs.data()[i]) > tolerance) return false;
    }
    return true;
}

spiral::Tensor make_matrix(std::size_t rows, std::size_t cols, float phase) {
    spiral::Tensor tensor({rows, cols});
    for (std::size_t i = 0; i < tensor.numel(); ++i) {
        tensor.data()[i] = std::sin(static_cast<float>(i) * 0.173F + phase) * 0.75F;
    }
    return tensor;
}

} // namespace

int main() {
    using namespace spiral;
    using namespace spiral::compute;

    ThreadPool pool(4);
    assert(pool.worker_count() == 4);
    std::mutex ids_mutex;
    std::set<std::thread::id> thread_ids;
    std::atomic<std::size_t> visited{0};
    pool.parallel_for(0, 48, 1, [&](std::size_t) {
        {
            std::lock_guard lock(ids_mutex);
            thread_ids.insert(std::this_thread::get_id());
        }
        visited.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    });
    assert(visited.load() == 48);
    assert(thread_ids.size() >= 2);

    ScratchArena arena(4096, 64);
    void* first = arena.allocate(13, 8);
    void* second = arena.allocate(64, 32);
    assert(reinterpret_cast<std::uintptr_t>(first) % 8 == 0);
    assert(reinterpret_cast<std::uintptr_t>(second) % 32 == 0);
    assert(arena.used() > 0);
    const std::size_t watermark = arena.high_watermark();
    auto floats = arena.allocate_array<float>(16);
    for (std::size_t i = 0; i < floats.size(); ++i) floats[i] = static_cast<float>(i);
    assert(arena.high_watermark() >= watermark);
    arena.reset();
    assert(arena.used() == 0);
    assert(arena.high_watermark() > 0);

    std::vector<float> a(19);
    std::vector<float> b(19);
    float reference_dot = 0.0F;
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i + 1) * 0.125F;
        b[i] = std::cos(static_cast<float>(i) * 0.3F);
        reference_dot += a[i] * b[i];
    }
    const float accelerated_dot = dot_product(a, b);
    assert(std::abs(accelerated_dot - reference_dot) < 1.0e-5F);
    assert(simd_width() >= 1);
    assert(!simd_backend_name().empty());

    const Tensor lhs = make_matrix(32, 24, 0.2F);
    const Tensor rhs = make_matrix(24, 20, 0.7F);
    const Tensor reference = lhs.matmul(rhs);
    const Tensor parallel = parallel_matmul(lhs, rhs, pool);
    assert(tensors_close(reference, parallel, 2.0e-4F));

    CpuBackend backend({4, 1});
    assert(backend.worker_count() == 4);
    assert(!backend.name().empty());
    const Tensor backend_result = backend.matmul(lhs, rhs);
    assert(tensors_close(reference, backend_result, 2.0e-4F));

    const Tensor qlhs_source = make_matrix(8, 12, 0.1F);
    const Tensor qrhs_source = make_matrix(12, 7, 1.1F);
    const auto qlhs = precision::quantize_symmetric_int8(qlhs_source);
    const auto qrhs = precision::quantize_symmetric_int8(qrhs_source);
    const Tensor q_reference = qlhs_source.matmul(qrhs_source);
    const Tensor q_result = backend.matmul_int8(qlhs, qrhs);
    assert(tensors_close(q_reference, q_result, 0.08F));

    std::unique_ptr<ComputeBackend> polymorphic = std::make_unique<CpuBackend>(ComputeConfig{2, 1});
    const Tensor polymorphic_result = polymorphic->matmul(lhs, rhs);
    assert(tensors_close(reference, polymorphic_result, 2.0e-4F));

    const auto bench = benchmark("dot19", 100, [&]() {
        const volatile float sink = dot_product(a, b);
        (void)sink;
    });
    assert(bench.label == "dot19");
    assert(bench.iterations == 100);
    assert(bench.seconds >= 0.0);
    assert(bench.iterations_per_second > 0.0);

    std::cout << "L18 workers observed: " << thread_ids.size() << '\n';
    std::cout << "L18 SIMD backend: " << simd_backend_name() << " width=" << simd_width() << '\n';
    std::cout << "L18 benchmark iterations/s: " << bench.iterations_per_second << '\n';
    std::cout << "L18 accelerated compute tests passed\n";
    return 0;
}
