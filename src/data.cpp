#include "spiral/data.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace spiral::data {
namespace {

constexpr std::array<char, 8> kMagic{'S','P','S','H','A','R','D','1'};

void write_u32(std::ostream& out, std::uint32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) throw std::runtime_error("failed while writing Spiral data shard");
}

void write_u64(std::ostream& out, std::uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) throw std::runtime_error("failed while writing Spiral data shard");
}

std::uint32_t read_u32(std::istream& in) {
    std::uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("truncated Spiral data shard");
    return value;
}

std::uint64_t read_u64(std::istream& in) {
    std::uint64_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("truncated Spiral data shard");
    return value;
}

std::uint32_t checked_u32(std::size_t value, const char* field) {
    if (value > std::numeric_limits<std::uint32_t>::max()) throw std::overflow_error(std::string(field) + " exceeds shard format limit");
    return static_cast<std::uint32_t>(value);
}

} // namespace

void write_image_prompt_shard(const std::string& path, const std::vector<ImagePromptRecord>& records) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to open Spiral data shard for writing");
    out.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    write_u64(out, static_cast<std::uint64_t>(records.size()));

    for (const auto& record : records) {
        const auto& pixels = record.image.pixels();
        if (record.image.width() == 0 || record.image.height() == 0 || pixels.size() != record.image.width() * record.image.height() * 3U) {
            throw std::invalid_argument("shard record image must be non-empty RGB8");
        }
        write_u32(out, checked_u32(record.prompt.size(), "prompt"));
        write_u32(out, checked_u32(record.image.width(), "image width"));
        write_u32(out, checked_u32(record.image.height(), "image height"));
        write_u32(out, checked_u32(pixels.size(), "pixel payload"));
        out.write(record.prompt.data(), static_cast<std::streamsize>(record.prompt.size()));
        out.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
        if (!out) throw std::runtime_error("failed while writing Spiral data shard record");
    }
}

ImagePromptShard::ImagePromptShard(std::string path) : path_(std::move(path)) {
    std::ifstream in(path_, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open Spiral data shard");
    std::array<char, 8> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!in || magic != kMagic) throw std::runtime_error("invalid Spiral data shard header");
    const std::uint64_t count = read_u64(in);
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) throw std::overflow_error("shard record count too large");
    entries_.reserve(static_cast<std::size_t>(count));

    for (std::uint64_t i = 0; i < count; ++i) {
        Entry entry;
        entry.prompt_bytes = read_u32(in);
        entry.width = read_u32(in);
        entry.height = read_u32(in);
        entry.pixel_bytes = read_u32(in);
        const std::uint64_t expected_pixels = static_cast<std::uint64_t>(entry.width) * entry.height * 3ULL;
        if (entry.width == 0 || entry.height == 0 || entry.pixel_bytes != expected_pixels) {
            throw std::runtime_error("invalid RGB payload in Spiral data shard");
        }
        const auto position = in.tellg();
        if (position < 0) throw std::runtime_error("failed to index Spiral data shard");
        entry.payload_offset = static_cast<std::uint64_t>(position);
        const std::uint64_t skip = static_cast<std::uint64_t>(entry.prompt_bytes) + entry.pixel_bytes;
        if (skip > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) throw std::overflow_error("shard record too large");
        in.seekg(static_cast<std::streamoff>(skip), std::ios::cur);
        if (!in) throw std::runtime_error("truncated Spiral data shard payload");
        entries_.push_back(entry);
    }
}

ImagePromptRecord ImagePromptShard::load(std::size_t index) const {
    if (index >= entries_.size()) throw std::out_of_range("shard record index out of range");
    const Entry& entry = entries_[index];
    std::ifstream in(path_, std::ios::binary);
    if (!in) throw std::runtime_error("failed to reopen Spiral data shard");
    in.seekg(static_cast<std::streamoff>(entry.payload_offset), std::ios::beg);
    if (!in) throw std::runtime_error("failed to seek Spiral data shard");

    std::string prompt(entry.prompt_bytes, '\0');
    std::vector<std::uint8_t> pixels(entry.pixel_bytes);
    in.read(prompt.data(), static_cast<std::streamsize>(prompt.size()));
    in.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    if (!in) throw std::runtime_error("truncated Spiral data shard record");
    return ImagePromptRecord{std::move(prompt), vision::RgbImage(entry.width, entry.height, std::move(pixels))};
}

ShardedImagePromptDataset::ShardedImagePromptDataset(std::size_t cache_capacity)
    : cache_capacity_(cache_capacity) {
    if (cache_capacity_ == 0) throw std::invalid_argument("dataset cache capacity must be non-zero");
}

void ShardedImagePromptDataset::add_shard(const std::string& path) {
    auto shard = std::make_shared<ImagePromptShard>(path);
    const std::size_t begin = total_size_;
    if (shard->size() > std::numeric_limits<std::size_t>::max() - total_size_) throw std::overflow_error("dataset size overflow");
    total_size_ += shard->size();
    shards_.push_back(ShardSpan{begin, total_size_, std::move(shard)});
}

const ShardedImagePromptDataset::ShardSpan& ShardedImagePromptDataset::span_for(std::size_t index) const {
    if (index >= total_size_) throw std::out_of_range("dataset index out of range");
    for (const auto& span : shards_) {
        if (index >= span.begin && index < span.end) return span;
    }
    throw std::runtime_error("dataset shard span lookup failed");
}

void ShardedImagePromptDataset::remember(std::size_t index, ImagePromptRecord record) const {
    const auto existing = cache_.find(index);
    if (existing != cache_.end()) {
        existing->second = std::move(record);
        return;
    }
    while (cache_.size() >= cache_capacity_ && !cache_order_.empty()) {
        const std::size_t evict = cache_order_.front();
        cache_order_.pop_front();
        cache_.erase(evict);
    }
    cache_order_.push_back(index);
    cache_.emplace(index, std::move(record));
}

ImagePromptRecord ShardedImagePromptDataset::load(std::size_t index) const {
    const auto cached = cache_.find(index);
    if (cached != cache_.end()) return cached->second;
    const auto& span = span_for(index);
    ImagePromptRecord record = span.shard->load(index - span.begin);
    remember(index, record);
    return record;
}

void ShardedImagePromptDataset::prefetch(std::size_t start, std::size_t count) const {
    if (start > total_size_) throw std::out_of_range("prefetch start out of range");
    const std::size_t end = count > total_size_ - start ? total_size_ : start + count;
    for (std::size_t index = start; index < end; ++index) (void)load(index);
}

void ShardedImagePromptDataset::clear_cache() const {
    cache_.clear();
    cache_order_.clear();
}

} // namespace spiral::data
