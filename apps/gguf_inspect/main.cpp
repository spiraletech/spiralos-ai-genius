#include "spiral/gguf.hpp"

#include <iostream>
#include <string_view>

namespace {
void print_metadata(const spiral::gguf::ModelFile& model, std::string_view key) {
    if (const auto* value = model.find(key))
        std::cout << key << ": " << spiral::gguf::value_summary(*value) << '\n';
}
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: SpiralGGUFInspect <model.gguf>\n";
        return 2;
    }
    spiral::gguf::Reader reader;
    std::string error;
    if (!reader.open(argv[1], &error)) {
        std::cerr << "GGUF error: " << error << '\n';
        return 1;
    }
    const auto& model = reader.model();
    std::cout << "GGUF version: " << model.version << '\n';
    print_metadata(model, "general.architecture");
    print_metadata(model, "general.name");
    std::cout << "general.alignment: " << model.alignment << '\n';
    for (const auto& entry : model.metadata) {
        if (entry.key.starts_with("tokenizer.") &&
            (entry.key.find("model") != std::string::npos || entry.key.find("chat_template") != std::string::npos ||
             entry.key.find("bos_token_id") != std::string::npos || entry.key.find("eos_token_id") != std::string::npos))
            std::cout << entry.key << ": " << spiral::gguf::value_summary(entry.value) << '\n';
    }
    std::cout << "Tensor count: " << model.tensors.size() << '\n';
    for (const auto& tensor : model.tensors) {
        std::cout << tensor.name << " [";
        for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
            if (i) std::cout << " x ";
            std::cout << tensor.shape[i];
        }
        std::cout << "] type=" << tensor.type_id << " offset=" << tensor.absolute_offset
                  << " bytes=" << tensor.byte_size << '\n';
    }
    return 0;
}
