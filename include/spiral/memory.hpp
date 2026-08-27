#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::memory {

struct MemoryRecord {
    std::uint64_t id = 0;
    std::string text;
    std::vector<std::string> tags;
    std::int64_t created_unix_ms = 0;
};

struct MemoryHit {
    MemoryRecord record;
    float score = 0.0F;
};

class MemoryStore {
public:
    std::uint64_t remember(std::string text, std::vector<std::string> tags = {}, std::int64_t created_unix_ms = 0);
    [[nodiscard]] std::vector<MemoryHit> search(std::string_view query, std::size_t top_k = 5) const;
    [[nodiscard]] const std::vector<MemoryRecord>& records() const noexcept { return records_; }
    void clear() noexcept;
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::uint64_t next_id_ = 1;
    std::vector<MemoryRecord> records_;
};

} // namespace spiral::memory
