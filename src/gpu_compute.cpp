#include "spiral/gpu_compute.hpp"

// L23 deliberately compiles the L22 D3D11 device implementation and the compute
// engine in one translation unit. This keeps the native D3D11 handles private:
// D3D11ComputeEngine is a friend of D3D11GpuDevice, while public headers remain
// free of Win32/D3D11 types.
#include "gpu.cpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <d3dcompiler.h>
#endif

namespace spiral::gpu {
namespace {

std::size_t checked_numel(const std::vector<std::size_t>& shape) {
    if (shape.empty()) return 0;
    std::size_t result = 1;
    for (const std::size_t dim : shape) {
        if (dim == 0) return 0;
        if (result > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::overflow_error("GPU tensor element count overflow");
        }
        result *= dim;
    }
    return result;
}

std::size_t checked_float_bytes(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("GPU tensor byte count overflow");
    }
    return count * sizeof(float);
}

std::uint64_t saturated_matmul_ops(const Tensor& lhs, const Tensor& rhs) noexcept {
    if (lhs.rank() != 2 || rhs.rank() != 2 || lhs.shape()[1] != rhs.shape()[0]) return 0;
    const std::uint64_t m = static_cast<std::uint64_t>(lhs.shape()[0]);
    const std::uint64_t k = static_cast<std::uint64_t>(lhs.shape()[1]);
    const std::uint64_t n = static_cast<std::uint64_t>(rhs.shape()[1]);
    constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
    if (m != 0 && k > max / m) return max;
    const std::uint64_t mk = m * k;
    if (mk != 0 && n > max / mk) return max;
    return mk * n;
}

#ifdef _WIN32

[[noreturn]] void throw_compute_hresult(const char* operation, HRESULT hr) {
    std::ostringstream stream;
    stream << operation << " failed with HRESULT 0x" << std::hex << std::uppercase
           << static_cast<unsigned long>(hr);
    throw std::runtime_error(stream.str());
}

UINT checked_dispatch_dim(std::size_t value, const char* label) {
    if (value == 0) return 1;
    if (value > static_cast<std::size_t>(D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION)) {
        throw std::overflow_error(std::string(label) + " exceeds D3D11 dispatch limits");
    }
    return static_cast<UINT>(value);
}

struct ShaderParams {
    std::uint32_t m = 0;
    std::uint32_t n = 0;
    std::uint32_t k = 0;
    std::uint32_t count = 0;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t activation = 0;
    std::uint32_t reserved = 0;
    float epsilon = 1.0e-5F;
    float scalar1 = 0.0F;
    float scalar2 = 0.0F;
    float scalar3 = 0.0F;
};
static_assert(sizeof(ShaderParams) % 16 == 0);

constexpr const char* kComputeShaderSource = R"HLSL(
ByteAddressBuffer Input0 : register(t0);
ByteAddressBuffer Input1 : register(t1);
RWByteAddressBuffer Output : register(u0);

cbuffer Params : register(b0) {
    uint4 Dim0;   // M, N, K, Count
    uint4 Dim1;   // Rows, Cols, Activation, Reserved
    float4 Scalar; // epsilon, spare...
};

float load0(uint index) { return asfloat(Input0.Load(index * 4)); }
float load1(uint index) { return asfloat(Input1.Load(index * 4)); }
void store_out(uint index, float value) { Output.Store(index * 4, asuint(value)); }

[numthreads(8, 8, 1)]
void MatMulCS(uint3 tid : SV_DispatchThreadID) {
    uint col = tid.x;
    uint row = tid.y;
    uint M = Dim0.x;
    uint N = Dim0.y;
    uint K = Dim0.z;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (uint inner = 0; inner < K; ++inner) {
        sum += load0(row * K + inner) * load1(inner * N + col);
    }
    store_out(row * N + col, sum);
}

[numthreads(64, 1, 1)]
void ActivationCS(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.x;
    uint count = Dim0.w;
    if (index >= count) return;
    float value = load0(index);
    float result;
    if (Dim1.z == 0) {
        result = max(value, 0.0f);
    } else {
        result = value / (1.0f + exp(-value));
    }
    store_out(index, result);
}

[numthreads(64, 1, 1)]
void LayerNormRowsCS(uint3 tid : SV_DispatchThreadID) {
    uint row = tid.x;
    uint rows = Dim1.x;
    uint cols = Dim1.y;
    if (row >= rows || cols == 0) return;
    uint base = row * cols;
    float mean = 0.0f;
    for (uint col = 0; col < cols; ++col) mean += load0(base + col);
    mean /= (float)cols;
    float variance = 0.0f;
    for (uint col = 0; col < cols; ++col) {
        float delta = load0(base + col) - mean;
        variance += delta * delta;
    }
    variance /= (float)cols;
    float inverse = rsqrt(variance + Scalar.x);
    for (uint col = 0; col < cols; ++col) {
        store_out(base + col, (load0(base + col) - mean) * inverse);
    }
}

[numthreads(64, 1, 1)]
void SoftmaxRowsCS(uint3 tid : SV_DispatchThreadID) {
    uint row = tid.x;
    uint rows = Dim1.x;
    uint cols = Dim1.y;
    if (row >= rows || cols == 0) return;
    uint base = row * cols;
    float maximum = -3.402823466e+38f;
    for (uint col = 0; col < cols; ++col) maximum = max(maximum, load0(base + col));
    float denominator = 0.0f;
    for (uint col = 0; col < cols; ++col) denominator += exp(load0(base + col) - maximum);
    denominator = max(denominator, 1.0e-20f);
    for (uint col = 0; col < cols; ++col) {
        store_out(base + col, exp(load0(base + col) - maximum) / denominator);
    }
}
)HLSL";

ID3D11ComputeShader* compile_compute_shader(
    ID3D11Device* device,
    const char* entry_point) {
    ID3DBlob* bytecode = nullptr;
    ID3DBlob* errors = nullptr;
    const UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;
    const HRESULT compile_hr = D3DCompile(
        kComputeShaderSource,
        std::strlen(kComputeShaderSource),
        "spiral_l23_compute",
        nullptr,
        nullptr,
        entry_point,
        "cs_5_0",
        flags,
        0,
        &bytecode,
        &errors);
    if (FAILED(compile_hr)) {
        std::string message = std::string("D3DCompile ") + entry_point;
        if (errors != nullptr && errors->GetBufferPointer() != nullptr) {
            message += ": ";
            message.append(
                static_cast<const char*>(errors->GetBufferPointer()),
                errors->GetBufferSize());
        }
        if (errors != nullptr) errors->Release();
        if (bytecode != nullptr) bytecode->Release();
        throw std::runtime_error(message);
    }
    if (errors != nullptr) errors->Release();

    ID3D11ComputeShader* shader = nullptr;
    const HRESULT shader_hr = device->CreateComputeShader(
        bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, &shader);
    bytecode->Release();
    if (FAILED(shader_hr)) throw_compute_hresult("ID3D11Device::CreateComputeShader", shader_hr);
    return shader;
}

ID3D11ShaderResourceView* make_raw_srv(
    ID3D11Device* device,
    ID3D11Buffer* buffer,
    std::size_t float_count) {
    if (float_count > std::numeric_limits<UINT>::max()) {
        throw std::overflow_error("GPU SRV element count exceeds UINT range");
    }
    D3D11_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_R32_TYPELESS;
    desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    desc.BufferEx.FirstElement = 0;
    desc.BufferEx.NumElements = static_cast<UINT>(float_count);
    desc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
    ID3D11ShaderResourceView* view = nullptr;
    const HRESULT hr = device->CreateShaderResourceView(buffer, &desc, &view);
    if (FAILED(hr)) throw_compute_hresult("ID3D11Device::CreateShaderResourceView", hr);
    return view;
}

ID3D11UnorderedAccessView* make_raw_uav(
    ID3D11Device* device,
    ID3D11Buffer* buffer,
    std::size_t float_count) {
    if (float_count > std::numeric_limits<UINT>::max()) {
        throw std::overflow_error("GPU UAV element count exceeds UINT range");
    }
    D3D11_UNORDERED_ACCESS_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_R32_TYPELESS;
    desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    desc.Buffer.FirstElement = 0;
    desc.Buffer.NumElements = static_cast<UINT>(float_count);
    desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    ID3D11UnorderedAccessView* view = nullptr;
    const HRESULT hr = device->CreateUnorderedAccessView(buffer, &desc, &view);
    if (FAILED(hr)) throw_compute_hresult("ID3D11Device::CreateUnorderedAccessView compute", hr);
    return view;
}

ID3D11Buffer* make_params_buffer(ID3D11Device* device, const ShaderParams& params) {
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = static_cast<UINT>(sizeof(ShaderParams));
    desc.Usage = D3D11_USAGE_IMMUTABLE;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA initial{};
    initial.pSysMem = &params;
    ID3D11Buffer* buffer = nullptr;
    const HRESULT hr = device->CreateBuffer(&desc, &initial, &buffer);
    if (FAILED(hr)) throw_compute_hresult("ID3D11Device::CreateBuffer compute params", hr);
    return buffer;
}

#endif

} // namespace

GpuTensor::GpuTensor(
    D3D11GpuDevice& device,
    device::BufferHandle handle,
    std::vector<std::size_t> shape)
    : device_(&device), handle_(handle), shape_(std::move(shape)) {}

GpuTensor::~GpuTensor() {
    reset();
}

GpuTensor::GpuTensor(GpuTensor&& other) noexcept
    : device_(other.device_), handle_(other.handle_), shape_(std::move(other.shape_)) {
    other.device_ = nullptr;
    other.handle_ = 0;
}

GpuTensor& GpuTensor::operator=(GpuTensor&& other) noexcept {
    if (this == &other) return *this;
    reset();
    device_ = other.device_;
    handle_ = other.handle_;
    shape_ = std::move(other.shape_);
    other.device_ = nullptr;
    other.handle_ = 0;
    return *this;
}

void GpuTensor::reset() noexcept {
    if (device_ != nullptr && handle_ != 0) {
        try {
            device_->destroy_buffer(handle_);
        } catch (...) {
            // Destructors must not throw; explicit device failures are surfaced on operations.
        }
    }
    device_ = nullptr;
    handle_ = 0;
    shape_.clear();
}

std::size_t GpuTensor::numel() const noexcept {
    if (shape_.empty()) return 0;
    std::size_t result = 1;
    for (const std::size_t dim : shape_) {
        if (dim == 0 || result > std::numeric_limits<std::size_t>::max() / dim) return 0;
        result *= dim;
    }
    return result;
}

struct D3D11ComputeEngine::Impl {
#ifdef _WIN32
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    ID3D11ComputeShader* matmul = nullptr;
    ID3D11ComputeShader* activation = nullptr;
    ID3D11ComputeShader* layer_norm = nullptr;
    ID3D11ComputeShader* softmax = nullptr;
#endif
    GpuComputeStats stats;

#ifdef _WIN32
    Impl(ID3D11Device* native_device, ID3D11DeviceContext* native_context)
        : device(native_device), context(native_context) {
        if (device == nullptr || context == nullptr) {
            throw std::invalid_argument("D3D11 compute engine requires initialized native handles");
        }
        matmul = compile_compute_shader(device, "MatMulCS");
        try {
            activation = compile_compute_shader(device, "ActivationCS");
            layer_norm = compile_compute_shader(device, "LayerNormRowsCS");
            softmax = compile_compute_shader(device, "SoftmaxRowsCS");
        } catch (...) {
            if (softmax != nullptr) softmax->Release();
            if (layer_norm != nullptr) layer_norm->Release();
            if (activation != nullptr) activation->Release();
            if (matmul != nullptr) matmul->Release();
            matmul = nullptr;
            throw;
        }
    }
#else
    Impl() = default;
#endif

    ~Impl() {
#ifdef _WIN32
        if (softmax != nullptr) softmax->Release();
        if (layer_norm != nullptr) layer_norm->Release();
        if (activation != nullptr) activation->Release();
        if (matmul != nullptr) matmul->Release();
#endif
    }

#ifdef _WIN32
    void dispatch(
        ID3D11ComputeShader* shader,
        ID3D11Buffer* input0,
        std::size_t input0_count,
        ID3D11Buffer* input1,
        std::size_t input1_count,
        ID3D11Buffer* output,
        std::size_t output_count,
        const ShaderParams& params,
        UINT groups_x,
        UINT groups_y = 1) {
        ID3D11ShaderResourceView* srv0 = nullptr;
        ID3D11ShaderResourceView* srv1 = nullptr;
        ID3D11UnorderedAccessView* uav = nullptr;
        ID3D11Buffer* constants = nullptr;
        try {
            srv0 = make_raw_srv(device, input0, input0_count);
            if (input1 != nullptr) srv1 = make_raw_srv(device, input1, input1_count);
            uav = make_raw_uav(device, output, output_count);
            constants = make_params_buffer(device, params);

            ID3D11ShaderResourceView* srvs[2] = {srv0, srv1};
            context->CSSetShader(shader, nullptr, 0);
            context->CSSetShaderResources(0, 2, srvs);
            context->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            context->CSSetConstantBuffers(0, 1, &constants);
            context->Dispatch(groups_x, groups_y, 1);

            ID3D11ShaderResourceView* null_srvs[2] = {nullptr, nullptr};
            ID3D11UnorderedAccessView* null_uav = nullptr;
            ID3D11Buffer* null_constant = nullptr;
            context->CSSetShaderResources(0, 2, null_srvs);
            context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
            context->CSSetConstantBuffers(0, 1, &null_constant);
            context->CSSetShader(nullptr, nullptr, 0);
            context->Flush();
            ++stats.dispatches;
        } catch (...) {
            if (constants != nullptr) constants->Release();
            if (uav != nullptr) uav->Release();
            if (srv1 != nullptr) srv1->Release();
            if (srv0 != nullptr) srv0->Release();
            throw;
        }
        constants->Release();
        uav->Release();
        if (srv1 != nullptr) srv1->Release();
        srv0->Release();
    }
#endif
};

D3D11ComputeEngine::D3D11ComputeEngine(D3D11GpuDevice& device)
    : device_(device) {
#ifdef _WIN32
    if (device_.impl_ == nullptr || device_.impl_->device == nullptr || device_.impl_->context == nullptr) {
        throw std::invalid_argument("D3D11 compute engine requires initialized GPU device");
    }
    if (device_.impl_->feature_level < D3D_FEATURE_LEVEL_11_0) {
        throw std::runtime_error("L23 compute shaders require D3D feature level 11.0 or newer");
    }
    impl_ = std::make_unique<Impl>(device_.impl_->device, device_.impl_->context);
#else
    throw std::runtime_error("D3D11 compute engine is only available on Windows");
#endif
}

D3D11ComputeEngine::~D3D11ComputeEngine() = default;

bool D3D11ComputeEngine::platform_supported() noexcept {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

bool D3D11ComputeEngine::available() const noexcept {
#ifdef _WIN32
    return impl_ != nullptr && impl_->device != nullptr && impl_->matmul != nullptr;
#else
    return false;
#endif
}

GpuTensor D3D11ComputeEngine::upload(const Tensor& tensor) {
    if (tensor.numel() == 0) throw std::invalid_argument("cannot upload empty tensor to GPU");
    const std::size_t bytes = checked_float_bytes(tensor.numel());
    const auto handle = device_.create_buffer(device::BufferDesc{
        bytes, device::BufferUsage::Storage, true});
    try {
        device_.upload(handle, std::as_bytes(std::span<const float>(tensor.data().data(), tensor.data().size())));
    } catch (...) {
        device_.destroy_buffer(handle);
        throw;
    }
    impl_->stats.uploaded_bytes += bytes;
    return GpuTensor(device_, handle, tensor.shape());
}

Tensor D3D11ComputeEngine::download(const GpuTensor& tensor) const {
    if (!tensor.valid() || tensor.device_ != &device_) {
        throw std::invalid_argument("GPU tensor does not belong to this compute engine device");
    }
    const std::size_t count = tensor.numel();
    const std::size_t bytes = checked_float_bytes(count);
    const auto raw = device_.download(tensor.handle_, 0, bytes);
    if (raw.size() != bytes) throw std::runtime_error("GPU tensor readback size mismatch");
    std::vector<float> values(count);
    std::memcpy(values.data(), raw.data(), bytes);
    impl_->stats.downloaded_bytes += bytes;
    return Tensor(tensor.shape_, std::move(values));
}

GpuTensor D3D11ComputeEngine::zeros(std::vector<std::size_t> shape) {
    const std::size_t count = checked_numel(shape);
    if (count == 0) throw std::invalid_argument("GPU tensor shape must contain non-zero dimensions");
    const auto handle = device_.create_buffer(device::BufferDesc{
        checked_float_bytes(count), device::BufferUsage::Storage, true});
    device::CommandList clear;
    clear.fill_buffer(handle, std::byte{0});
    try {
        device_.submit(clear);
    } catch (...) {
        device_.destroy_buffer(handle);
        throw;
    }
    return GpuTensor(device_, handle, std::move(shape));
}

GpuTensor D3D11ComputeEngine::matmul(const GpuTensor& lhs, const GpuTensor& rhs) {
#ifdef _WIN32
    if (!lhs.valid() || !rhs.valid() || lhs.device_ != &device_ || rhs.device_ != &device_) {
        throw std::invalid_argument("GPU matmul tensors must belong to this compute engine device");
    }
    if (lhs.rank() != 2 || rhs.rank() != 2) throw std::invalid_argument("GPU matmul requires rank-2 tensors");
    if (lhs.shape_[1] != rhs.shape_[0]) throw std::invalid_argument("GPU matmul inner dimensions must match");
    const std::size_t m = lhs.shape_[0];
    const std::size_t k = lhs.shape_[1];
    const std::size_t n = rhs.shape_[1];
    if (m > std::numeric_limits<std::uint32_t>::max() ||
        n > std::numeric_limits<std::uint32_t>::max() ||
        k > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("GPU matmul dimensions exceed shader parameter range");
    }

    auto output = zeros({m, n});
    ShaderParams params;
    params.m = static_cast<std::uint32_t>(m);
    params.n = static_cast<std::uint32_t>(n);
    params.k = static_cast<std::uint32_t>(k);

    auto& lhs_buffer = device_.impl_->require_buffer(lhs.handle_);
    auto& rhs_buffer = device_.impl_->require_buffer(rhs.handle_);
    auto& out_buffer = device_.impl_->require_buffer(output.handle_);
    impl_->dispatch(
        impl_->matmul,
        lhs_buffer.resource, lhs.numel(),
        rhs_buffer.resource, rhs.numel(),
        out_buffer.resource, output.numel(),
        params,
        checked_dispatch_dim((n + 7U) / 8U, "GPU matmul X groups"),
        checked_dispatch_dim((m + 7U) / 8U, "GPU matmul Y groups"));
    return output;
#else
    (void)lhs; (void)rhs;
    throw std::runtime_error("D3D11 GPU matmul is unavailable on this platform");
#endif
}

GpuTensor D3D11ComputeEngine::activation(const GpuTensor& input, GpuActivation activation_kind) {
#ifdef _WIN32
    if (!input.valid() || input.device_ != &device_) {
        throw std::invalid_argument("GPU activation tensor must belong to this compute engine device");
    }
    auto output = zeros(input.shape_);
    ShaderParams params;
    if (input.numel() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("GPU activation element count exceeds shader parameter range");
    }
    params.count = static_cast<std::uint32_t>(input.numel());
    params.activation = activation_kind == GpuActivation::Relu ? 0U : 1U;
    auto& input_buffer = device_.impl_->require_buffer(input.handle_);
    auto& out_buffer = device_.impl_->require_buffer(output.handle_);
    impl_->dispatch(
        impl_->activation,
        input_buffer.resource, input.numel(),
        nullptr, 0,
        out_buffer.resource, output.numel(),
        params,
        checked_dispatch_dim((input.numel() + 63U) / 64U, "GPU activation groups"));
    return output;
#else
    (void)input; (void)activation_kind;
    throw std::runtime_error("D3D11 GPU activation is unavailable on this platform");
#endif
}

GpuTensor D3D11ComputeEngine::layer_norm_rows(const GpuTensor& input, float epsilon) {
#ifdef _WIN32
    if (!input.valid() || input.device_ != &device_) {
        throw std::invalid_argument("GPU layer norm tensor must belong to this compute engine device");
    }
    if (input.rank() == 0 || input.shape_.back() == 0) throw std::invalid_argument("GPU layer norm requires non-empty last dimension");
    if (!(epsilon > 0.0F)) throw std::invalid_argument("GPU layer norm epsilon must be positive");
    const std::size_t cols = input.shape_.back();
    const std::size_t rows = input.numel() / cols;
    if (rows > std::numeric_limits<std::uint32_t>::max() || cols > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("GPU layer norm dimensions exceed shader parameter range");
    }
    auto output = zeros(input.shape_);
    ShaderParams params;
    params.rows = static_cast<std::uint32_t>(rows);
    params.cols = static_cast<std::uint32_t>(cols);
    params.epsilon = epsilon;
    auto& input_buffer = device_.impl_->require_buffer(input.handle_);
    auto& out_buffer = device_.impl_->require_buffer(output.handle_);
    impl_->dispatch(
        impl_->layer_norm,
        input_buffer.resource, input.numel(),
        nullptr, 0,
        out_buffer.resource, output.numel(),
        params,
        checked_dispatch_dim((rows + 63U) / 64U, "GPU layer norm groups"));
    return output;
#else
    (void)input; (void)epsilon;
    throw std::runtime_error("D3D11 GPU layer norm is unavailable on this platform");
#endif
}

GpuTensor D3D11ComputeEngine::softmax_rows(const GpuTensor& input) {
#ifdef _WIN32
    if (!input.valid() || input.device_ != &device_) {
        throw std::invalid_argument("GPU softmax tensor must belong to this compute engine device");
    }
    if (input.rank() == 0 || input.shape_.back() == 0) throw std::invalid_argument("GPU softmax requires non-empty last dimension");
    const std::size_t cols = input.shape_.back();
    const std::size_t rows = input.numel() / cols;
    if (rows > std::numeric_limits<std::uint32_t>::max() || cols > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("GPU softmax dimensions exceed shader parameter range");
    }
    auto output = zeros(input.shape_);
    ShaderParams params;
    params.rows = static_cast<std::uint32_t>(rows);
    params.cols = static_cast<std::uint32_t>(cols);
    auto& input_buffer = device_.impl_->require_buffer(input.handle_);
    auto& out_buffer = device_.impl_->require_buffer(output.handle_);
    impl_->dispatch(
        impl_->softmax,
        input_buffer.resource, input.numel(),
        nullptr, 0,
        out_buffer.resource, output.numel(),
        params,
        checked_dispatch_dim((rows + 63U) / 64U, "GPU softmax groups"));
    return output;
#else
    (void)input;
    throw std::runtime_error("D3D11 GPU softmax is unavailable on this platform");
#endif
}

Tensor D3D11ComputeEngine::matmul(const Tensor& lhs, const Tensor& rhs) {
    auto gpu_lhs = upload(lhs);
    auto gpu_rhs = upload(rhs);
    auto gpu_out = matmul(gpu_lhs, gpu_rhs);
    return download(gpu_out);
}

Tensor D3D11ComputeEngine::activation(const Tensor& input, GpuActivation activation_kind) {
    auto gpu_input = upload(input);
    auto gpu_out = activation(gpu_input, activation_kind);
    return download(gpu_out);
}

Tensor D3D11ComputeEngine::layer_norm_rows(const Tensor& input, float epsilon) {
    auto gpu_input = upload(input);
    auto gpu_out = layer_norm_rows(gpu_input, epsilon);
    return download(gpu_out);
}

Tensor D3D11ComputeEngine::softmax_rows(const Tensor& input) {
    auto gpu_input = upload(input);
    auto gpu_out = softmax_rows(gpu_input);
    return download(gpu_out);
}

GpuComputeStats D3D11ComputeEngine::stats() const noexcept {
    return impl_ == nullptr ? GpuComputeStats{} : impl_->stats;
}

HybridComputeBackend::HybridComputeBackend(
    D3D11GpuDevice* gpu_device,
    HybridComputeConfig config)
    : config_(config), cpu_(config.cpu) {
    if (gpu_device != nullptr) {
        try {
            gpu_ = std::make_unique<D3D11ComputeEngine>(*gpu_device);
        } catch (...) {
            gpu_.reset();
        }
    }
    name_ = gpu_ != nullptr ? "spiral-hybrid-d3d11" : "spiral-hybrid-cpu";
}

bool HybridComputeBackend::should_use_gpu_matmul(const Tensor& lhs, const Tensor& rhs) const noexcept {
    if (gpu_ == nullptr || lhs.rank() != 2 || rhs.rank() != 2 || lhs.shape()[1] != rhs.shape()[0]) return false;
    return saturated_matmul_ops(lhs, rhs) >= static_cast<std::uint64_t>(config_.gpu_matmul_threshold_ops);
}

Tensor HybridComputeBackend::matmul(const Tensor& lhs, const Tensor& rhs) {
    if (should_use_gpu_matmul(lhs, rhs)) {
        last_backend_ = ExecutionBackend::Gpu;
        return gpu_->matmul(lhs, rhs);
    }
    last_backend_ = ExecutionBackend::Cpu;
    return cpu_.matmul(lhs, rhs);
}

Tensor HybridComputeBackend::matmul_int8(
    const precision::QuantizedTensor& lhs,
    const precision::QuantizedTensor& rhs) {
    last_backend_ = ExecutionBackend::Cpu;
    return cpu_.matmul_int8(lhs, rhs);
}

std::string_view execution_backend_name(ExecutionBackend backend) noexcept {
    switch (backend) {
        case ExecutionBackend::Cpu: return "cpu";
        case ExecutionBackend::Gpu: return "gpu";
    }
    return "unknown";
}

std::string_view gpu_activation_name(GpuActivation activation_kind) noexcept {
    switch (activation_kind) {
        case GpuActivation::Relu: return "relu";
        case GpuActivation::Silu: return "silu";
    }
    return "unknown";
}

} // namespace spiral::gpu
