#include "spiral/nn.hpp"
#include "spiral/random.hpp"
#include "spiral/tensor.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace {
bool near(float a, float b, float epsilon = 1.0e-4F) { return std::fabs(a - b) <= epsilon; }
}

int main() {
    using spiral::Random;
    using spiral::Tensor;

    Tensor matrix({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    assert(matrix.strides() == std::vector<std::size_t>({3, 1}));

    auto transposed = matrix.transpose2d();
    assert(transposed.shape() == std::vector<std::size_t>({3, 2}));
    assert(transposed.strides() == std::vector<std::size_t>({1, 3}));
    const std::size_t index[] = {1, 0};
    assert(near(transposed.at(index), 2.0F));
    transposed.at(index) = 20.0F;
    assert(near(matrix.data()[1], 20.0F));

    Random a(42);
    Random b(42);
    for (int i = 0; i < 8; ++i) assert(a.next_u64() == b.next_u64());

    Random rng(1337);
    spiral::nn::Embedding embedding(258, 8, rng);
    const std::uint32_t ids[] = {256, 83, 112, 257};
    const Tensor embedded = embedding.forward(ids);
    assert(embedded.shape() == std::vector<std::size_t>({4, 8}));

    spiral::nn::LayerNorm layer_norm(4);
    const Tensor normalized = layer_norm.forward(Tensor({1, 4}, {1.0F, 2.0F, 3.0F, 4.0F}));
    float mean = 0.0F;
    for (float value : normalized.data()) mean += value;
    assert(near(mean / 4.0F, 0.0F));

    spiral::nn::RMSNorm rms_norm(4);
    const Tensor rms = rms_norm.forward(Tensor({1, 4}, {1.0F, 2.0F, 3.0F, 4.0F}));
    float mean_square = 0.0F;
    for (float value : rms.data()) mean_square += value * value;
    assert(near(mean_square / 4.0F, 1.0F, 1.0e-3F));

    spiral::nn::Sequential graph;
    graph.add(std::make_unique<spiral::nn::Linear>(4, 6, rng));
    graph.add(std::make_unique<spiral::nn::RMSNorm>(6));
    graph.add(std::make_unique<spiral::nn::Linear>(6, 2, rng));

    const Tensor output = graph.forward(Tensor({2, 4}, {
        1.0F, 2.0F, 3.0F, 4.0F,
        4.0F, 3.0F, 2.0F, 1.0F,
    }));
    assert(output.shape() == std::vector<std::size_t>({2, 2}));
    assert(graph.parameters().size() == 5);

    std::cout << "spiral_nn_tests: PASS\n";
    return 0;
}
