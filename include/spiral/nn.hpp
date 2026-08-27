#pragma once

#include "spiral/random.hpp"
#include "spiral/tensor.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace spiral::nn {

struct Parameter {
    std::string name;
    Tensor value;
    bool trainable = true;
};

class Module {
public:
    virtual ~Module() = default;
    [[nodiscard]] virtual Tensor forward(const Tensor& input) const = 0;
    [[nodiscard]] virtual std::vector<Parameter*> parameters();
    [[nodiscard]] virtual std::vector<const Parameter*> parameters() const;
};

class Linear final : public Module {
public:
    Linear(std::size_t in_features, std::size_t out_features, Random& rng, bool use_bias = true);
    [[nodiscard]] Tensor forward(const Tensor& input) const override;
    [[nodiscard]] std::vector<Parameter*> parameters() override;
    [[nodiscard]] std::vector<const Parameter*> parameters() const override;
    [[nodiscard]] const Parameter& weight() const noexcept { return weight_; }
    [[nodiscard]] const Parameter& bias() const noexcept { return bias_; }

private:
    std::size_t in_features_;
    std::size_t out_features_;
    bool use_bias_;
    Parameter weight_;
    Parameter bias_;
};

class Embedding final {
public:
    Embedding(std::size_t vocabulary_size, std::size_t embedding_dim, Random& rng);
    [[nodiscard]] Tensor forward(std::span<const std::uint32_t> token_ids) const;
    [[nodiscard]] Parameter& table() noexcept { return table_; }
    [[nodiscard]] const Parameter& table() const noexcept { return table_; }

private:
    std::size_t vocabulary_size_;
    std::size_t embedding_dim_;
    Parameter table_;
};

class RMSNorm final : public Module {
public:
    explicit RMSNorm(std::size_t feature_size, float epsilon = 1.0e-5F);
    [[nodiscard]] Tensor forward(const Tensor& input) const override;
    [[nodiscard]] std::vector<Parameter*> parameters() override;
    [[nodiscard]] std::vector<const Parameter*> parameters() const override;

private:
    std::size_t feature_size_;
    float epsilon_;
    Parameter scale_;
};

class LayerNorm final : public Module {
public:
    explicit LayerNorm(std::size_t feature_size, float epsilon = 1.0e-5F);
    [[nodiscard]] Tensor forward(const Tensor& input) const override;
    [[nodiscard]] std::vector<Parameter*> parameters() override;
    [[nodiscard]] std::vector<const Parameter*> parameters() const override;

private:
    std::size_t feature_size_;
    float epsilon_;
    Parameter scale_;
    Parameter bias_;
};

class Sequential final : public Module {
public:
    Sequential() = default;
    void add(std::unique_ptr<Module> module);

    template <typename T, typename... Args>
    T& emplace(Args&&... args) {
        auto module = std::make_unique<T>(std::forward<Args>(args)...);
        T& ref = *module;
        add(std::move(module));
        return ref;
    }

    [[nodiscard]] Tensor forward(const Tensor& input) const override;
    [[nodiscard]] std::vector<Parameter*> parameters() override;
    [[nodiscard]] std::vector<const Parameter*> parameters() const override;
    [[nodiscard]] std::size_t size() const noexcept { return modules_.size(); }

private:
    std::vector<std::unique_ptr<Module>> modules_;
};

} // namespace spiral::nn
