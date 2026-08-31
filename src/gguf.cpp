#include "spiral/gguf.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace spiral::gguf {
namespace {

constexpr std::uint64_t max_entries = 16'000'000;
constexpr std::uint64_t max_dimensions = 4;
constexpr std::uint64_t max_nesting = 8;

class Parser final {
public:
    explicit Parser(const std::filesystem::path& path) : input_(path, std::ios::binary) {
        if (!input_) throw std::runtime_error("cannot open file");
        input_.seekg(0, std::ios::end);
        const auto end = input_.tellg();
        if (end < 0) throw std::runtime_error("cannot determine file size");
        size_ = static_cast<std::uint64_t>(end);
        input_.seekg(0);
    }

    ModelFile parse() {
        ModelFile result;
        result.file_size = size_;
        char magic[4]{};
        read_bytes(magic, sizeof(magic));
        if (std::memcmp(magic, "GGUF", 4) != 0) fail("invalid GGUF magic");
        result.version = read<std::uint32_t>();
        if (result.version != 3) fail("unsupported GGUF version (expected v3)");
        const auto tensor_count = read<std::uint64_t>();
        const auto metadata_count = read<std::uint64_t>();
        check_count(tensor_count, "tensor count");
        check_count(metadata_count, "metadata count");
        result.metadata.reserve(to_size(metadata_count));
        for (std::uint64_t i = 0; i < metadata_count; ++i) {
            MetadataEntry entry;
            entry.key = read_string();
            if (entry.key.empty()) fail("empty metadata key");
            entry.value = read_value(read_type(), 0);
            result.metadata.push_back(std::move(entry));
        }
        if (const auto* value = find(result.metadata, "general.alignment")) {
            if (value->type != ValueType::uint32) fail("general.alignment must be uint32");
            result.alignment = std::get<std::uint32_t>(value->data);
        }
        if (result.alignment == 0 || (result.alignment & (result.alignment - 1)) != 0 || result.alignment > (1U << 20))
            fail("invalid GGUF alignment");

        result.tensors.reserve(to_size(tensor_count));
        for (std::uint64_t i = 0; i < tensor_count; ++i) {
            TensorInfo tensor;
            tensor.name = read_string();
            if (tensor.name.empty()) fail("empty tensor name");
            const auto dimensions = read<std::uint32_t>();
            if (dimensions == 0 || dimensions > max_dimensions) fail("invalid tensor dimension count");
            tensor.shape.reserve(dimensions);
            for (std::uint32_t d = 0; d < dimensions; ++d) {
                const auto extent = read<std::uint64_t>();
                if (extent == 0) fail("zero tensor dimension");
                tensor.shape.push_back(extent);
            }
            tensor.type_id = read<std::uint32_t>();
            tensor.relative_offset = read<std::uint64_t>();
            result.tensors.push_back(std::move(tensor));
        }
        result.tensor_data_offset = align_up(position(), result.alignment);
        if (result.tensor_data_offset > size_) fail("tensor data starts past end of file");
        for (auto& tensor : result.tensors) validate_tensor(result, tensor);
        return result;
    }

private:
    template <typename T> T read() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        read_bytes(&value, sizeof(value));
        return value;
    }

    void read_bytes(void* destination, std::size_t count) {
        if (count > size_ - position()) fail("truncated file");
        input_.read(static_cast<char*>(destination), static_cast<std::streamsize>(count));
        if (!input_) fail("truncated file");
    }

    std::string read_string() {
        const auto length = read<std::uint64_t>();
        if (length > size_ - position() || length > std::numeric_limits<std::size_t>::max())
            fail("invalid or truncated string length");
        std::string value(to_size(length), '\0');
        if (length != 0) read_bytes(value.data(), value.size());
        return value;
    }

    ValueType read_type() {
        const auto raw = read<std::uint32_t>();
        if (raw > static_cast<std::uint32_t>(ValueType::float64)) fail("unknown metadata value type");
        return static_cast<ValueType>(raw);
    }

    Value read_value(ValueType type, std::uint64_t depth) {
        if (depth > max_nesting) fail("metadata arrays nested too deeply");
        Value value;
        value.type = type;
        switch (type) {
        case ValueType::uint8: value.data = read<std::uint8_t>(); break;
        case ValueType::int8: value.data = read<std::int8_t>(); break;
        case ValueType::uint16: value.data = read<std::uint16_t>(); break;
        case ValueType::int16: value.data = read<std::int16_t>(); break;
        case ValueType::uint32: value.data = read<std::uint32_t>(); break;
        case ValueType::int32: value.data = read<std::int32_t>(); break;
        case ValueType::float32: value.data = read<float>(); break;
        case ValueType::boolean: {
            const auto raw = read<std::uint8_t>();
            if (raw > 1) fail("invalid boolean metadata value");
            value.data = raw != 0;
            break;
        }
        case ValueType::string: value.data = read_string(); break;
        case ValueType::array: {
            value.array_element_type = read_type();
            if (value.array_element_type == ValueType::array) fail("GGUF arrays cannot contain arrays");
            const auto count = read<std::uint64_t>();
            check_count(count, "array length");
            Value::Array items;
            items.reserve(to_size(count));
            for (std::uint64_t i = 0; i < count; ++i) items.push_back(read_value(value.array_element_type, depth + 1));
            value.data = std::move(items);
            break;
        }
        case ValueType::uint64: value.data = read<std::uint64_t>(); break;
        case ValueType::int64: value.data = read<std::int64_t>(); break;
        case ValueType::float64: value.data = read<double>(); break;
        }
        return value;
    }

    static const Value* find(const std::vector<MetadataEntry>& entries, std::string_view key) {
        const auto it = std::find_if(entries.begin(), entries.end(), [key](const auto& e) { return e.key == key; });
        return it == entries.end() ? nullptr : &it->value;
    }

    void validate_tensor(const ModelFile& model, TensorInfo& tensor) {
        if ((tensor.relative_offset % model.alignment) != 0) fail("unaligned tensor offset: " + tensor.name);
        if (tensor.relative_offset > std::numeric_limits<std::uint64_t>::max() - model.tensor_data_offset)
            fail("tensor offset overflow: " + tensor.name);
        tensor.absolute_offset = model.tensor_data_offset + tensor.relative_offset;
        if (tensor.absolute_offset > size_) fail("tensor offset past end of file: " + tensor.name);
        const auto [block, bytes] = type_layout(tensor.type_id);
        std::uint64_t elements = 1;
        for (const auto extent : tensor.shape) {
            if (extent > std::numeric_limits<std::uint64_t>::max() / elements) fail("tensor element count overflow: " + tensor.name);
            elements *= extent;
        }
        if (tensor.shape.front() % block != 0) fail("tensor row is incompatible with quantization block: " + tensor.name);
        const auto blocks = elements / block;
        if (blocks > std::numeric_limits<std::uint64_t>::max() / bytes) fail("tensor byte size overflow: " + tensor.name);
        tensor.byte_size = blocks * bytes;
        if (tensor.byte_size > size_ - tensor.absolute_offset) fail("tensor payload extends past end of file: " + tensor.name);
    }

    static std::pair<std::uint64_t, std::uint64_t> type_layout(std::uint32_t type) {
        // GGML on-disk type layouts. Unsupported future types fail closed.
        switch (type) {
        case 0: return {1, 4}; case 1: return {1, 2};
        case 2: return {32, 18}; case 3: return {32, 20}; case 6: return {32, 22}; case 7: return {32, 24};
        case 8: return {32, 34}; case 9: return {32, 36}; case 10: return {256, 84}; case 11: return {256, 110};
        case 12: return {256, 144}; case 13: return {256, 176}; case 14: return {256, 210}; case 15: return {256, 292};
        case 16: return {256, 66}; case 17: return {256, 74}; case 18: return {256, 98};
        case 19: return {256, 50}; case 20: return {32, 18}; case 21: return {256, 110};
        case 22: return {256, 82}; case 23: return {256, 136};
        case 24: return {1, 1}; case 25: return {1, 2}; case 26: return {1, 4}; case 27: return {1, 8};
        case 28: return {1, 8}; case 29: return {256, 56}; case 30: return {1, 2};
        case 34: return {256, 54}; case 35: return {256, 66};
        default: throw std::runtime_error("unsupported tensor type id " + std::to_string(type));
        }
    }

    static std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
        const auto remainder = value % alignment;
        if (remainder == 0) return value;
        const auto addition = alignment - remainder;
        if (value > std::numeric_limits<std::uint64_t>::max() - addition) throw std::runtime_error("alignment overflow");
        return value + addition;
    }
    void check_count(std::uint64_t count, const char* what) const {
        if (count > max_entries || count > size_) throw std::runtime_error(std::string("absurd ") + what);
    }
    static std::size_t to_size(std::uint64_t value) { return static_cast<std::size_t>(value); }
    std::uint64_t position() {
        const auto value = input_.tellg();
        if (value < 0) fail("invalid stream position");
        return static_cast<std::uint64_t>(value);
    }
    [[noreturn]] static void fail(const std::string& message) { throw std::runtime_error(message); }

    std::ifstream input_;
    std::uint64_t size_ = 0;
};

} // namespace

const Value* ModelFile::find(std::string_view key) const noexcept {
    const auto it = std::find_if(metadata.begin(), metadata.end(), [key](const auto& e) { return e.key == key; });
    return it == metadata.end() ? nullptr : &it->value;
}

bool Reader::open(const std::filesystem::path& path, std::string* error) noexcept {
    try {
        ModelFile parsed = Parser(path).parse();
        model_ = std::move(parsed);
        if (error) error->clear();
        return true;
    } catch (const std::exception& exception) {
        model_ = {};
        if (error) *error = exception.what();
        return false;
    } catch (...) {
        model_ = {};
        if (error) *error = "unknown GGUF parsing error";
        return false;
    }
}

std::string value_summary(const Value& value, std::size_t max_items) {
    std::ostringstream out;
    std::visit([&](const auto& item) {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, Value::Array>) {
            out << '[';
            for (std::size_t i = 0; i < std::min(item.size(), max_items); ++i) {
                if (i) out << ", ";
                out << value_summary(item[i], max_items);
            }
            if (item.size() > max_items) out << ", ... (" << item.size() << " items)";
            out << ']';
        } else if constexpr (std::is_same_v<T, std::string>) out << item;
        else if constexpr (std::is_same_v<T, bool>) out << (item ? "true" : "false");
        else if constexpr (std::is_same_v<T, std::uint8_t> || std::is_same_v<T, std::int8_t>) out << +item;
        else out << item;
    }, value.data);
    return out.str();
}

} // namespace spiral::gguf
