#include "spiral/tensor.hpp"
#include "spiral/tokenizer.hpp"

#include <iostream>
#include <string>

int main() {
    spiral::ByteTokenizer tokenizer;
    const std::string seed = "Spiral awake";
    const auto tokens = tokenizer.encode(seed);

    spiral::Tensor a({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    spiral::Tensor b({2, 2}, {5.0F, 6.0F, 7.0F, 8.0F});
    const auto product = a.matmul(b);

    std::cout << "SPIRAL-AI GENIUS / L0\n";
    std::cout << "native C++ sovereign foundation online\n";
    std::cout << "tokenizer vocab: " << spiral::ByteTokenizer::vocabulary_size << "\n";
    std::cout << "seed tokens: " << tokens.size() << "\n";
    std::cout << product.describe() << "\n";
    std::cout << "matmul[0,0]: " << product.data().at(0) << "\n";
    std::cout << "decoded: " << tokenizer.decode(tokens) << "\n";
    return 0;
}
