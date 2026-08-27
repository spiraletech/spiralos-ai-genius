#include "spiral/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace spiral {
namespace {

std::size_t view_numel(const std::vector<std::size_t>& shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>{});
}

std::size_t view_offset(
    std::span<const std::size_t> indices,
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides) {
    if (indices.size() != shape.size()) throw std::out_of_range("Tensor view index rank mismatch");
    std::size_t flat = 0;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (indices[axis] >= shape[axis]) throw std::out_of_range("Tensor view index out of bounds");
        flat += indices[axis] * strides[axis];
    }
    return flat;
}

void validate_view_shape(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& strides) {
    if (shape.size() != strides.size()) throw std::invalid_argument("Tensor view shape/stride rank mismatch");
}

} // namespace

ConstTensorView::ConstTensorView(
    const float* data,
    std::vector<std::size_t> shape,
    std::vector<std::size_t> strides)
    : data_(data), shape_(std::move(shape)), strides_(std::move(strides)) {
    validate_view_shape(shape_, strides_);
}

std::size_t ConstTensorView::numel() const noexcept { return view_numel(shape_); }
std::size_t ConstTensorView::offset(std::span<const std::size_t> indices) const { return view_offset(indices, shape_, strides_); }

float ConstTensorView::at(std::span<const std::size_t> indices) const {
    if (data_ == nullptr && numel() != 0) throw std::runtime_error("Tensor view has no backing storage");
    return data_[offset(indices)];
}

ConstTensorView ConstTensorView::transpose2d() const {
    if (rank() != 2) throw std::invalid_argument("transpose2d requires rank-2 tensor view");
    return ConstTensorView(data_, {shape_[1], shape_[0]}, {strides_[1], strides_[0]});
}

TensorView::TensorView(float* data, std::vector<std::size_t> shape, std::vector<std::size_t> strides)
    : data_(data), shape_(std::move(shape)), strides_(std::move(strides)) {
    validate_view_shape(shape_, strides_);
}

std::size_t TensorView::numel() const noexcept { return view_numel(shape_); }
std::size_t TensorView::offset(std::span<const std::size_t> indices) const { return view_offset(indices, shape_, strides_); }

float TensorView::at(std::span<const std::size_t> indices) const {
    if (data_ == nullptr && numel() != 0) throw std::runtime_error("Tensor view has no backing storage");
    return data_[offset(indices)];
}

float& TensorView::at(std::span<const std::size_t> indices) {
    if (data_ == nullptr && numel() != 0) throw std::runtime_error("Tensor view has no backing storage");
    return data_[offset(indices)];
}

TensorView TensorView::transpose2d() const {
    if (rank() != 2) throw std::invalid_argument("transpose2d requires rank-2 tensor view");
    return TensorView(data_, {shape_[1], shape_[0]}, {strides_[1], strides_[0]});
}

ConstTensorView TensorView::as_const() const { return ConstTensorView(data_, shape_, strides_); }

std::size_t Tensor::checked_numel(const std::vector<std::size_t>& shape) {
    if (shape.empty()) return 0;
    std::size_t total = 1;
    for (const auto dim : shape) {
        if (dim == 0) return 0;
        if (total > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::overflow_error("Tensor shape overflows size_t");
        }
        total *= dim;
    }
    return total;
}

std::vector<std::size_t> Tensor::contiguous_strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> strides(shape.size(), 1);
    std::size_t stride = 1;
    for (std::size_t axis = shape.size(); axis-- > 0;) {
        strides[axis] = stride;
        stride *= shape[axis];
    }
    return strides;
}

Tensor::Tensor(std::vector<std::size_t> shape, float fill)
    : shape_(std::move(shape)), strides_(contiguous_strides(shape_)), data_(checked_numel(shape_), fill) {}

Tensor::Tensor(std::vector<std::size_t> shape, std::vector<float> values)
    : shape_(std::move(shape)), strides_(contiguous_strides(shape_)), data_(std::move(values)) {
    if (data_.size() != checked_numel(shape_)) throw std::invalid_argument("Tensor value count does not match shape");
}

Tensor Tensor::zeros(std::vector<std::size_t> shape) { return Tensor(std::move(shape), 0.0F); }
Tensor Tensor::ones(std::vector<std::size_t> shape) { return Tensor(std::move(shape), 1.0F); }
std::size_t Tensor::offset(std::span<const std::size_t> indices) const { return view_offset(indices, shape_, strides_); }
float Tensor::at(std::span<const std::size_t> indices) const { return data_.at(offset(indices)); }
float& Tensor::at(std::span<const std::size_t> indices) { return data_.at(offset(indices)); }
TensorView Tensor::view() { return TensorView(data_.data(), shape_, strides_); }
ConstTensorView Tensor::view() const { return ConstTensorView(data_.data(), shape_, strides_); }
TensorView Tensor::transpose2d() { return view().transpose2d(); }
ConstTensorView Tensor::transpose2d() const { return view().transpose2d(); }

Tensor Tensor::add(const Tensor& rhs) const {
    if (shape_ != rhs.shape_) throw std::invalid_argument("Tensor add requires equal shapes");
    Tensor out(shape_);
    for (std::size_t i = 0; i < data_.size(); ++i) out.data_[i] = data_[i] + rhs.data_[i];
    return out;
}

Tensor Tensor::matmul(const Tensor& rhs) const {
    if (rank() != 2 || rhs.rank() != 2) throw std::invalid_argument("Tensor matmul currently supports rank-2 tensors only");
    const auto m = shape_[0];
    const auto k = shape_[1];
    const auto rhs_k = rhs.shape_[0];
    const auto n = rhs.shape_[1];
    if (k != rhs_k) throw std::invalid_argument("Tensor matmul inner dimensions must match");

    Tensor out({m, n});
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            float sum = 0.0F;
            for (std::size_t inner = 0; inner < k; ++inner) {
                sum += data_[row * k + inner] * rhs.data_[inner * n + col];
            }
            out.data_[row * n + col] = sum;
        }
    }
    return out;
}

Tensor Tensor::relu() const {
    Tensor out(shape_);
    std::transform(data_.begin(), data_.end(), out.data_.begin(), [](float value) { return std::max(0.0F, value); });
    return out;
}

Tensor Tensor::softmax() const {
    if (data_.empty()) return Tensor(shape_);
    const float max_value = *std::max_element(data_.begin(), data_.end());
    Tensor out(shape_);
    float sum = 0.0F;
    for (std::size_t i = 0; i < data_.size(); ++i) {
        out.data_[i] = std::exp(data_[i] - max_value);
        sum += out.data_[i];
    }
    if (sum == 0.0F || !std::isfinite(sum)) throw std::runtime_error("Tensor softmax normalization failed");
    for (auto& value : out.data_) value /= sum;
    return out;
}

std::string Tensor::describe() const {
    std::ostringstream out;
    out << "Tensor(shape=[";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (i != 0) out << ',';
        out << shape_[i];
    }
    out << "], numel=" << numel() << ')';
    return out.str();
}

} // namespace spiral
