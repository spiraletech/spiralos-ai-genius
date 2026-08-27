#include "spiral/device.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace spiral::device {
namespace {

std::size_t resolved_size(std::size_t total, std::size_t offset, std::size_t requested) {
    if (offset > total) throw std::out_of_range("device buffer offset exceeds buffer size");
    return requested == 0 ? total - offset : requested;
}

} // namespace

void CommandList::fill_buffer(
    BufferHandle destination,
    std::byte value,
    std::size_t offset,
    std::size_t size_bytes) {
    if (destination == 0) throw std::invalid_argument("fill_buffer destination must be valid");
    commands_.push_back(DeviceCommand{
        CommandKind::FillBuffer,
        0,
        destination,
        0,
        offset,
        size_bytes,
        value});
}

void CommandList::copy_buffer(
    BufferHandle source,
    BufferHandle destination,
    std::size_t size_bytes,
    std::size_t source_offset,
    std::size_t destination_offset) {
    if (source == 0 || destination == 0) throw std::invalid_argument("copy_buffer handles must be valid");
    if (size_bytes == 0) throw std::invalid_argument("copy_buffer size must be non-zero");
    commands_.push_back(DeviceCommand{
        CommandKind::CopyBuffer,
        source,
        destination,
        source_offset,
        destination_offset,
        size_bytes,
        std::byte{0}});
}

void CommandQueue::submit(const CommandList& commands) {
    device_.submit(commands);
    if (submission_count_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("device queue submission counter overflow");
    }
    ++submission_count_;
}

CpuReferenceDevice::CpuReferenceDevice(std::string name)
    : name_(std::move(name)) {
    if (name_.empty()) throw std::invalid_argument("device name must not be empty");
}

DeviceInfo CpuReferenceDevice::info() const {
    DeviceInfo result;
    result.name = name_;
    result.kind = DeviceKind::CpuReference;
    result.buffer_count = buffers_.size();
    for (const auto& [handle, buffer] : buffers_) {
        (void)handle;
        result.allocated_bytes += buffer.bytes.size();
    }
    return result;
}

BufferHandle CpuReferenceDevice::create_buffer(BufferDesc desc) {
    if (desc.size_bytes == 0) throw std::invalid_argument("device buffer size must be non-zero");
    if (next_handle_ == 0) throw std::overflow_error("device buffer handle overflow");
    const BufferHandle handle = next_handle_++;
    buffers_.emplace(handle, Buffer{desc, std::vector<std::byte>(desc.size_bytes, std::byte{0})});
    return handle;
}

void CpuReferenceDevice::destroy_buffer(BufferHandle handle) {
    if (buffers_.erase(handle) == 0) throw std::invalid_argument("device buffer handle not found");
}

CpuReferenceDevice::Buffer& CpuReferenceDevice::require_buffer(BufferHandle handle) {
    const auto it = buffers_.find(handle);
    if (it == buffers_.end()) throw std::invalid_argument("device buffer handle not found");
    return it->second;
}

const CpuReferenceDevice::Buffer& CpuReferenceDevice::require_buffer(BufferHandle handle) const {
    const auto it = buffers_.find(handle);
    if (it == buffers_.end()) throw std::invalid_argument("device buffer handle not found");
    return it->second;
}

void CpuReferenceDevice::require_range(const Buffer& buffer, std::size_t offset, std::size_t size_bytes) {
    if (offset > buffer.bytes.size() || size_bytes > buffer.bytes.size() - offset) {
        throw std::out_of_range("device buffer range exceeds allocation");
    }
}

void CpuReferenceDevice::upload(BufferHandle handle, std::span<const std::byte> data, std::size_t offset) {
    auto& buffer = require_buffer(handle);
    if (!buffer.desc.host_visible) throw std::logic_error("device buffer is not host visible");
    require_range(buffer, offset, data.size());
    std::copy(data.begin(), data.end(), buffer.bytes.begin() + static_cast<std::ptrdiff_t>(offset));
}

std::vector<std::byte> CpuReferenceDevice::download(
    BufferHandle handle,
    std::size_t offset,
    std::size_t size_bytes) const {
    const auto& buffer = require_buffer(handle);
    if (!buffer.desc.host_visible) throw std::logic_error("device buffer is not host visible");
    const std::size_t count = resolved_size(buffer.bytes.size(), offset, size_bytes);
    require_range(buffer, offset, count);
    return std::vector<std::byte>(
        buffer.bytes.begin() + static_cast<std::ptrdiff_t>(offset),
        buffer.bytes.begin() + static_cast<std::ptrdiff_t>(offset + count));
}

void CpuReferenceDevice::submit(const CommandList& commands) {
    for (const auto& command : commands.commands()) {
        switch (command.kind) {
            case CommandKind::FillBuffer: {
                auto& destination = require_buffer(command.destination);
                const std::size_t count = resolved_size(
                    destination.bytes.size(),
                    command.destination_offset,
                    command.size_bytes);
                require_range(destination, command.destination_offset, count);
                std::fill(
                    destination.bytes.begin() + static_cast<std::ptrdiff_t>(command.destination_offset),
                    destination.bytes.begin() + static_cast<std::ptrdiff_t>(command.destination_offset + count),
                    command.fill_value);
                break;
            }
            case CommandKind::CopyBuffer: {
                const auto& source = require_buffer(command.source);
                auto& destination = require_buffer(command.destination);
                require_range(source, command.source_offset, command.size_bytes);
                require_range(destination, command.destination_offset, command.size_bytes);
                std::memmove(
                    destination.bytes.data() + command.destination_offset,
                    source.bytes.data() + command.source_offset,
                    command.size_bytes);
                break;
            }
        }
    }
}

std::string_view device_kind_name(DeviceKind kind) noexcept {
    switch (kind) {
        case DeviceKind::CpuReference: return "cpu-reference";
        case DeviceKind::Gpu: return "gpu";
    }
    return "unknown";
}

} // namespace spiral::device
