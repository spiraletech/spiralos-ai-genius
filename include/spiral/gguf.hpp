#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace spiral::gguf {

enum class ValueType : std::uint32_t {
    uint8 = 0, int8 = 1, uint16 = 2, int16 = 3, uint32 = 4, int32 = 5,
    float32 = 6, boolean = 7, string = 8, array = 9, uint64 = 10,
    int64 = 11, float64 = 12,
};

struct Value {
    using Array = std::vector<Value>;
    using Storage = std::variant<std::uint8_t, std::int8_t, std::uint16_t, std::int16_t,
        std::uint32_t, std::int32_t, float, bool, std::string, Array,
        std::uint64_t, std::int64_t, double>;

    ValueType type{};
    ValueType array_element_type{};
    Storage data{std::uint8_t{0}};
};

struct MetadataEntry {
    std::string key;
    Value value;
};

struct TensorInfo {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::uint32_t type_id = 0;
    std::uint64_t relative_offset = 0;
    std::uint64_t absolute_offset = 0;
    std::uint64_t byte_size = 0;
};

struct ModelFile {
    std::uint32_t version = 0;
    std::uint64_t file_size = 0;
    std::uint64_t tensor_data_offset = 0;
    std::uint32_t alignment = 32;
    std::vector<MetadataEntry> metadata;
    std::vector<TensorInfo> tensors;

    [[nodiscard]] const Value* find(std::string_view key) const noexcept;
};

class Reader final {
public:
    [[nodiscard]] bool open(const std::filesystem::path& path, std::string* error = nullptr) noexcept;
    [[nodiscard]] const ModelFile& model() const noexcept { return model_; }

private:
    ModelFile model_;
};

[[nodiscard]] std::string value_summary(const Value& value, std::size_t max_items = 8);

} // namespace spiral::gguf
