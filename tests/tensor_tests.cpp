#include "spiral/tensor.hpp"
#include "spiral/tokenizer.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

namespace {

bool near(float a, float b, float epsilon = 1.0e-5F) {
    return std::fabs(a - b) <= epsilon;
}

} // namespace

int main() {
    using spiral::Tensor;

    const Tensor a({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    const Tensor b({2, 2}, {5.0F, 6.0F, 7.0F, 8.0F});

    const auto sum = a.add(b);
    assert(sum.data().size() == 4);
    assert(near(sum.data()[0], 6.0F));
    assert(near(sum.data()[3], 12.0F));

    const auto product = a.matmul(b);
    assert(product.shape() == std::vector<std::size_t>({2, 2}));
    assert(near(product.data()[0], 19.0F));
    assert(near(product.data()[1], 22.0F));
    assert(near(product.data()[2], 43.0F));
    assert(near(product.data()[3], 50.0F));

    const Tensor activations({4}, {-2.0F, 0.0F, 1.0F, 3.0F});
    const auto relu = activations.relu();
    assert(near(relu.data()[0], 0.0F));
    assert(near(relu.data()[3], 3.0F));

    const auto probabilities = activations.softmax();
    float probability_sum = 0.0F;
    for (const float value : probabilities.data()) {
        probability_sum += value;
    }
    assert(near(probability_sum, 1.0F));

    spiral::ByteTokenizer tokenizer;
    const std::string phrase = "SpiralOS";
    const auto tokens = tokenizer.encode(phrase);
    assert(tokens.front() == spiral::ByteTokenizer::bos_token);
    assert(tokens.back() == spiral::ByteTokenizer::eos_token);
    assert(tokenizer.decode(tokens) == phrase);

    std::cout << "spiral_core_tests: PASS\n";
    return 0;
}
