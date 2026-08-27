#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::device {

using BufferHandle = std::uint64_t;

enum class DeviceKind {
    CpuReference,
    Gpu,
};

enum class BufferUsage {
    Generic,
    Vertex,
    Index,
    Storage,
    Transfer,
};

struct BufferDesc {
    std::size_t size_bytes = 0;
    BufferUsage usage = BufferUsage::Generic;
    bool host_visible = true;
};

struct DeviceInfo {
    std::string name;
    DeviceKind kind = DeviceKind::CpuReference;
    std::size_t buffer_count = 0;
    std::size_t allocated_bytes = 0;
};

enum class CommandKind {
    FillBuffer,
    CopyBuffer,
};

struct DeviceCommand {
    CommandKind kind = CommandKind::FillBuffer;
    BufferHandle source = 0;
    BufferHandle destination = 0;
    std::size_t source_offset = 0;
    std::size_t destination_offset = 0;
    std::size_t size_bytes = 0;
    std::byte fill_value{0};
};

class CommandList final {
public:
    void fill_buffer(
        BufferHandle destination,
        std::byte value,
        std::size_t offset = 0,
        std::size_t size_bytes = 0);
    void copy_buffer(
        BufferHandle source,
        BufferHandle destination,
        std::size_t size_bytes,
        std::size_t source_offset = 0,
        std::size_t destination_offset = 0);
    void clear() noexcept { commands_.clear(); }

    [[nodiscard]] const std::vector<DeviceCommand>& commands() const noexcept { return commands_; }
    [[nodiscard]] bool empty() const noexcept { return commands_.empty(); }

private:
    std::vector<DeviceCommand> commands_;
};

class Device {
public:
    virtual ~Device() = default;

    [[nodiscard]] virtual DeviceInfo info() const = 0;
    [[nodiscard]] virtual BufferHandle create_buffer(BufferDesc desc) = 0;
    virtual void destroy_buffer(BufferHandle handle) = 0;
    virtual void upload(BufferHandle handle, std::span<const std::byte> data, std::size_t offset = 0) = 0;
    [[nodiscard]] virtual std::vector<std::byte> download(
        BufferHandle handle,
        std::size_t offset = 0,
        std::size_t size_bytes = 0) const = 0;
    virtual void submit(const CommandList& commands) = 0;
};

class CommandQueue final {
public:
    explicit CommandQueue(Device& device) : device_(device) {}

    void submit(const CommandList& commands);
    [[nodiscard]] std::uint64_t submission_count() const noexcept { return submission_count_; }

private:
    Device& device_;
    std::uint64_t submission_count_ = 0;
};

class CpuReferenceDevice final : public Device {
public:
    explicit CpuReferenceDevice(std::string name = "spiral-cpu-reference");

    [[nodiscard]] DeviceInfo info() const override;
    [[nodiscard]] BufferHandle create_buffer(BufferDesc desc) override;
    void destroy_buffer(BufferHandle handle) override;
    void upload(BufferHandle handle, std::span<const std::byte> data, std::size_t offset = 0) override;
    [[nodiscard]] std::vector<std::byte> download(
        BufferHandle handle,
        std::size_t offset = 0,
        std::size_t size_bytes = 0) const override;
    void submit(const CommandList& commands) override;

private:
    struct Buffer {
        BufferDesc desc;
        std::vector<std::byte> bytes;
    };

    [[nodiscard]] Buffer& require_buffer(BufferHandle handle);
    [[nodiscard]] const Buffer& require_buffer(BufferHandle handle) const;
    static void require_range(const Buffer& buffer, std::size_t offset, std::size_t size_bytes);

    std::string name_;
    BufferHandle next_handle_ = 1;
    std::map<BufferHandle, Buffer> buffers_;
};

[[nodiscard]] std::string_view device_kind_name(DeviceKind kind) noexcept;

} // namespace spiral::device
