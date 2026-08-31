#include "spiral/gguf.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
template <typename T> void append(std::vector<char>& bytes, T value) {
    const auto* first = reinterpret_cast<const char*>(&value);
    bytes.insert(bytes.end(), first, first + sizeof(value));
}
void append_string(std::vector<char>& bytes, const std::string& value) {
    append(bytes, static_cast<std::uint64_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}
std::vector<char> valid_file(std::uint64_t tensor_offset = 0) {
    std::vector<char> bytes{'G', 'G', 'U', 'F'};
    append(bytes, std::uint32_t{3});
    append(bytes, std::uint64_t{1});
    append(bytes, std::uint64_t{1});
    append_string(bytes, "general.alignment");
    append(bytes, std::uint32_t{4});
    append(bytes, std::uint32_t{32});
    append_string(bytes, "x");
    append(bytes, std::uint32_t{1});
    append(bytes, std::uint64_t{1});
    append(bytes, std::uint32_t{0});
    append(bytes, tensor_offset);
    bytes.resize(96, '\0');
    append(bytes, float{1.0F});
    return bytes;
}
std::filesystem::path write_case(const std::string& name, const std::vector<char>& bytes) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream stream(path, std::ios::binary);
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return path;
}
bool rejects(const std::vector<char>& bytes, const std::string& name) {
    const auto path = write_case(name, bytes);
    spiral::gguf::Reader reader;
    std::string error;
    const bool rejected = !reader.open(path, &error) && !error.empty();
    std::filesystem::remove(path);
    return rejected;
}
}

int main() {
    const auto path = write_case("spiral_gguf_valid.gguf", valid_file());
    spiral::gguf::Reader reader;
    std::string error;
    assert(reader.open(path, &error));
    assert(error.empty());
    assert(reader.model().version == 3);
    assert(reader.model().alignment == 32);
    assert(reader.model().tensors.size() == 1);
    assert(reader.model().tensors[0].absolute_offset == 96);
    assert(reader.model().tensors[0].byte_size == 4);
    std::filesystem::remove(path);

    auto truncated = valid_file();
    truncated.resize(95);
    assert(rejects(truncated, "spiral_gguf_truncated.gguf"));

    auto bad_magic = valid_file();
    bad_magic[0] = 'B';
    assert(rejects(bad_magic, "spiral_gguf_magic.gguf"));

    auto absurd = valid_file();
    const std::uint64_t count = UINT64_MAX;
    std::memcpy(absurd.data() + 16, &count, sizeof(count));
    assert(rejects(absurd, "spiral_gguf_absurd.gguf"));

    assert(rejects(valid_file(32), "spiral_gguf_offset.gguf"));
    std::cout << "spiral_gguf_tests: PASS\n";
    return 0;
}
