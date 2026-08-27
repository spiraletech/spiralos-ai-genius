#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace spiral {

class ConstTensorView {
public:
    ConstTensorView() = default;
    ConstTensorView(const float* data, std::vector<std::size_t> shape, std::vector<std::size_t> strides);

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] const std::vector<std::size_t>& strides() const noexcept { return strides_; }
    [[nodiscard]] std::size_t rank() const noexcept { return shape_.size(); }
    [[nodiscard]] std::size_t numel() const noexcept;
    [[nodiscard]] float at(std::span<const std::size_t> indices) const;
    [[nodiscard]] ConstTensorView transpose2d() const;

private:
    [[nodiscard]] std::size_t offset(std::span<const std::size_t> indices) const;
    const float* data_ = nullptr;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
};

class TensorView {
public:
    TensorView() = default;
    TensorView(float* data, std::vector<std::size_t> shape, std::vector<std::size_t> strides);

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] const std::vector<std::size_t>& strides() const noexcept { return strides_; }
    [[nodiscard]] std::size_t rank() const noexcept { return shape_.size(); }
    [[nodiscard]] std::size_t numel() const noexcept;
    [[nodiscard]] float at(std::span<const std::size_t> indices) const;
    float& at(std::span<const std::size_t> indices);
    [[nodiscard]] TensorView transpose2d() const;
    [[nodiscard]] ConstTensorView as_const() const;

private:
    [[nodiscard]] std::size_t offset(std::span<const std::size_t> indices) const;
    float* data_ = nullptr;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
};

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape, float fill = 0.0F);
    Tensor(std::vector<std::size_t> shape, std::vector<float> values);

    static Tensor zeros(std::vector<std::size_t> shape);
    static Tensor ones(std::vector<std::size_t> shape);

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] const std::vector<std::size_t>& strides() const noexcept { return strides_; }
    [[nodiscard]] std::size_t rank() const noexcept { return shape_.size(); }
    [[nodiscard]] std::size_t numel() const noexcept { return data_.size(); }
    [[nodiscard]] const std::vector<float>& data() const noexcept { return data_; }
    [[nodiscard]] std::vector<float>& data() noexcept { return data_; }

    [[nodiscard]] float at(std::span<const std::size_t> indices) const;
    float& at(std::span<const std::size_t> indices);
    [[nodiscard]] TensorView view();
    [[nodiscard]] ConstTensorView view() const;
    [[nodiscard]] TensorView transpose2d();
    [[nodiscard]] ConstTensorView transpose2d() const;

    [[nodiscard]] Tensor add(const Tensor& rhs) const;
    [[nodiscard]] Tensor matmul(const Tensor& rhs) const;
    [[nodiscard]] Tensor relu() const;
    [[nodiscard]] Tensor softmax() const;
    [[nodiscard]] std::string describe() const;

private:
    [[nodiscard]] std::size_t offset(std::span<const std::size_t> indices) const;
    static std::size_t checked_numel(const std::vector<std::size_t>& shape);
    static std::vector<std::size_t> contiguous_strides(const std::vector<std::size_t>& shape);

    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<float> data_;
};

} // namespace spiral
