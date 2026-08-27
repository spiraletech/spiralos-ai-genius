#pragma once

#include <cstddef>
#include <initializer_list>
#include <span>
#include <string>
#include <vector>

namespace spiral {

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape, float fill = 0.0F);
    Tensor(std::vector<std::size_t> shape, std::vector<float> values);

    static Tensor zeros(std::vector<std::size_t> shape);
    static Tensor ones(std::vector<std::size_t> shape);

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] std::size_t rank() const noexcept { return shape_.size(); }
    [[nodiscard]] std::size_t numel() const noexcept { return data_.size(); }

    [[nodiscard]] const std::vector<float>& data() const noexcept { return data_; }
    [[nodiscard]] std::vector<float>& data() noexcept { return data_; }

    [[nodiscard]] float at(std::span<const std::size_t> indices) const;
    float& at(std::span<const std::size_t> indices);

    [[nodiscard]] Tensor add(const Tensor& rhs) const;
    [[nodiscard]] Tensor matmul(const Tensor& rhs) const;
    [[nodiscard]] Tensor relu() const;
    [[nodiscard]] Tensor softmax() const;
    [[nodiscard]] std::string describe() const;

private:
    [[nodiscard]] std::size_t offset(std::span<const std::size_t> indices) const;
    static std::size_t checked_numel(const std::vector<std::size_t>& shape);

    std::vector<std::size_t> shape_;
    std::vector<float> data_;
};

} // namespace spiral
