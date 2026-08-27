#pragma once

#include "spiral/vision.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace spiral::data {

struct ImagePromptRecord {
    std::string prompt;
    vision::RgbImage image;
};

void write_image_prompt_shard(const std::string& path, const std::vector<ImagePromptRecord>& records);

class ImagePromptShard final {
public:
    explicit ImagePromptShard(std::string path);

    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    [[nodiscard]] ImagePromptRecord load(std::size_t index) const;
    [[nodiscard]] const std::string& path() const noexcept { return path_; }

private:
    struct Entry {
        std::uint64_t payload_offset = 0;
        std::uint32_t prompt_bytes = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t pixel_bytes = 0;
    };

    std::string path_;
    std::vector<Entry> entries_;
};

class ShardedImagePromptDataset final {
public:
    explicit ShardedImagePromptDataset(std::size_t cache_capacity = 8);

    void add_shard(const std::string& path);
    [[nodiscard]] std::size_t size() const noexcept { return total_size_; }
    [[nodiscard]] std::size_t shard_count() const noexcept { return shards_.size(); }
    [[nodiscard]] ImagePromptRecord load(std::size_t index) const;
    void prefetch(std::size_t start, std::size_t count) const;
    void clear_cache() const;
    [[nodiscard]] std::size_t resident_count() const noexcept { return cache_.size(); }

private:
    struct ShardSpan {
        std::size_t begin = 0;
        std::size_t end = 0;
        std::shared_ptr<ImagePromptShard> shard;
    };

    [[nodiscard]] const ShardSpan& span_for(std::size_t index) const;
    void remember(std::size_t index, ImagePromptRecord record) const;

    std::size_t cache_capacity_;
    std::size_t total_size_ = 0;
    std::vector<ShardSpan> shards_;
    mutable std::unordered_map<std::size_t, ImagePromptRecord> cache_;
    mutable std::deque<std::size_t> cache_order_;
};

} // namespace spiral::data
