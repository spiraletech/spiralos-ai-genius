#include "spiral/attention.hpp"
#include "spiral/tensor_ops.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool near(float a, float b, float epsilon = 1.0e-4F) {
    return std::fabs(a - b) <= epsilon;
}

} // namespace

int main() {
    using spiral::Tensor;

    const Tensor lhs({1, 2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    const Tensor rhs({1, 2, 2}, {5.0F, 6.0F, 7.0F, 8.0F});
    const auto product = spiral::ops::batched_matmul(lhs, rhs);
    assert(product.shape() == std::vector<std::size_t>({1, 2, 2}));
    assert(near(product.data()[0], 19.0F));
    assert(near(product.data()[3], 50.0F));

    const Tensor logits({2, 3}, {1.0F, 2.0F, 3.0F, 3.0F, 2.0F, 1.0F});
    const auto probabilities = spiral::ops::softmax_last_dim(logits);
    assert(near(probabilities.data()[0] + probabilities.data()[1] + probabilities.data()[2], 1.0F));
    assert(near(probabilities.data()[3] + probabilities.data()[4] + probabilities.data()[5], 1.0F));

    const auto mask = spiral::ops::causal_mask(3);
    assert(mask.data()[0] == 0.0F);
    assert(std::isinf(mask.data()[1]) && mask.data()[1] < 0.0F);
    assert(mask.data()[3] == 0.0F);
    assert(std::isinf(mask.data()[5]) && mask.data()[5] < 0.0F);

    Tensor rotary({1, 2, 4}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    const float norm_before = rotary.data()[4] * rotary.data()[4] + rotary.data()[5] * rotary.data()[5];
    spiral::nn::apply_rotary_inplace(rotary);
    const float norm_after = rotary.data()[4] * rotary.data()[4] + rotary.data()[5] * rotary.data()[5];
    assert(near(norm_before, norm_after, 1.0e-3F));

    spiral::Random rng(123);
    spiral::nn::CausalSelfAttention attention(4, 2, rng, false);
    const Tensor first({3, 4}, {
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
        9.0F, 10.0F, 11.0F, 12.0F});
    const Tensor changed_future({3, 4}, {
        1.0F, 2.0F, 3.0F, 4.0F,
        50.0F, 60.0F, 70.0F, 80.0F,
        90.0F, 100.0F, 110.0F, 120.0F});

    const auto output_a = attention.forward(first);
    const auto output_b = attention.forward(changed_future);
    assert(output_a.shape() == std::vector<std::size_t>({3, 4}));
    for (std::size_t feature = 0; feature < 4; ++feature) {
        assert(near(output_a.data()[feature], output_b.data()[feature]));
    }
    assert(attention.parameters().size() == 4);

    std::cout << "spiral_attention_tests: PASS\n";
    return 0;
}
