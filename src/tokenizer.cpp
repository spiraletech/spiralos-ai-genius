#include "spiral/tokenizer.hpp"

#include <stdexcept>

namespace spiral {

std::vector<ByteTokenizer::Token> ByteTokenizer::encode(
    std::string_view text,
    bool add_bos,
    bool add_eos) const {
    std::vector<Token> tokens;
    tokens.reserve(text.size() + static_cast<std::size_t>(add_bos) + static_cast<std::size_t>(add_eos));

    if (add_bos) {
        tokens.push_back(bos_token);
    }

    for (const unsigned char byte : text) {
        tokens.push_back(static_cast<Token>(byte));
    }

    if (add_eos) {
        tokens.push_back(eos_token);
    }
    return tokens;
}

std::string ByteTokenizer::decode(const std::vector<Token>& tokens) const {
    std::string text;
    text.reserve(tokens.size());

    for (const Token token : tokens) {
        if (token == bos_token || token == eos_token) {
            continue;
        }
        if (token > 255) {
            throw std::invalid_argument("Unknown byte token");
        }
        text.push_back(static_cast<char>(static_cast<unsigned char>(token)));
    }
    return text;
}

} // namespace spiral
