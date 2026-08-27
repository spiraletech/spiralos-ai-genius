#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace spiral {

class ByteTokenizer {
public:
    using Token = std::uint32_t;

    static constexpr Token bos_token = 256;
    static constexpr Token eos_token = 257;
    static constexpr std::size_t vocabulary_size = 258;

    [[nodiscard]] std::vector<Token> encode(
        std::string_view text,
        bool add_bos = true,
        bool add_eos = true) const;

    [[nodiscard]] std::string decode(const std::vector<Token>& tokens) const;
};

} // namespace spiral
