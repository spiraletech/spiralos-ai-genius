#include "spiral/memory.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace spiral::memory {
namespace {
constexpr std::uint32_t memory_magic = 0x534D454DU;
constexpr std::uint32_t memory_version = 1U;

void write_u32(std::ostream& out, std::uint32_t v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
void write_u64(std::ostream& out, std::uint64_t v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
void write_i64(std::ostream& out, std::int64_t v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
std::uint32_t read_u32(std::istream& in) { std::uint32_t v{}; in.read(reinterpret_cast<char*>(&v), sizeof(v)); if (!in) throw std::runtime_error("memory file ended unexpectedly"); return v; }
std::uint64_t read_u64(std::istream& in) { std::uint64_t v{}; in.read(reinterpret_cast<char*>(&v), sizeof(v)); if (!in) throw std::runtime_error("memory file ended unexpectedly"); return v; }
std::int64_t read_i64(std::istream& in) { std::int64_t v{}; in.read(reinterpret_cast<char*>(&v), sizeof(v)); if (!in) throw std::runtime_error("memory file ended unexpectedly"); return v; }
void write_string(std::ostream& out, const std::string& s) { write_u64(out, s.size()); out.write(s.data(), static_cast<std::streamsize>(s.size())); }
std::string read_string(std::istream& in) { const auto n = read_u64(in); if (n > (1ULL << 31)) throw std::runtime_error("memory string too large"); std::string s(static_cast<std::size_t>(n), '\0'); in.read(s.data(), static_cast<std::streamsize>(s.size())); if (!in) throw std::runtime_error("memory file ended unexpectedly"); return s; }

std::unordered_map<std::string, float> terms(std::string_view text) {
    std::unordered_map<std::string, float> out;
    std::string word;
    auto flush = [&] {
        if (!word.empty()) { out[word] += 1.0F; word.clear(); }
    };
    for (const unsigned char ch : text) {
        if (std::isalnum(ch) || ch == '_') word.push_back(static_cast<char>(std::tolower(ch)));
        else flush();
    }
    flush();
    return out;
}

float cosine(const std::unordered_map<std::string, float>& a, const std::unordered_map<std::string, float>& b) {
    if (a.empty() || b.empty()) return 0.0F;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (const auto& [k, v] : a) { na += static_cast<double>(v) * v; if (auto it = b.find(k); it != b.end()) dot += static_cast<double>(v) * it->second; }
    for (const auto& [k, v] : b) nb += static_cast<double>(v) * v;
    if (na == 0.0 || nb == 0.0) return 0.0F;
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}
}

std::uint64_t MemoryStore::remember(std::string text, std::vector<std::string> tags, std::int64_t created_unix_ms) {
    if (text.empty()) throw std::invalid_argument("memory text must not be empty");
    const auto id = next_id_++;
    records_.push_back(MemoryRecord{id, std::move(text), std::move(tags), created_unix_ms});
    return id;
}

std::vector<MemoryHit> MemoryStore::search(std::string_view query, std::size_t top_k) const {
    if (top_k == 0 || query.empty()) return {};
    const auto q = terms(query);
    std::vector<MemoryHit> hits;
    for (const auto& record : records_) {
        auto document = terms(record.text);
        for (const auto& tag : record.tags) {
            const auto tag_terms = terms(tag);
            for (const auto& [term, weight] : tag_terms) document[term] += weight * 1.5F;
        }
        const float score = cosine(q, document);
        if (score > 0.0F) hits.push_back(MemoryHit{record, score});
    }
    std::sort(hits.begin(), hits.end(), [](const MemoryHit& a, const MemoryHit& b) {
        if (a.score == b.score) return a.record.id < b.record.id;
        return a.score > b.score;
    });
    if (hits.size() > top_k) hits.resize(top_k);
    return hits;
}

void MemoryStore::clear() noexcept { records_.clear(); next_id_ = 1; }

void MemoryStore::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to open memory store for writing");
    write_u32(out, memory_magic); write_u32(out, memory_version); write_u64(out, next_id_); write_u64(out, records_.size());
    for (const auto& r : records_) {
        write_u64(out, r.id); write_i64(out, r.created_unix_ms); write_string(out, r.text); write_u64(out, r.tags.size());
        for (const auto& tag : r.tags) write_string(out, tag);
    }
    if (!out) throw std::runtime_error("failed while writing memory store");
}

void MemoryStore::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open memory store for reading");
    if (read_u32(in) != memory_magic || read_u32(in) != memory_version) throw std::runtime_error("unsupported memory store format");
    const auto next = read_u64(in); const auto count = read_u64(in);
    if (count > (1ULL << 28)) throw std::runtime_error("memory record count is unreasonable");
    std::vector<MemoryRecord> loaded; loaded.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        MemoryRecord r; r.id = read_u64(in); r.created_unix_ms = read_i64(in); r.text = read_string(in); const auto tags = read_u64(in);
        if (tags > (1ULL << 20)) throw std::runtime_error("memory tag count is unreasonable");
        r.tags.reserve(static_cast<std::size_t>(tags)); for (std::uint64_t t = 0; t < tags; ++t) r.tags.push_back(read_string(in)); loaded.push_back(std::move(r));
    }
    records_ = std::move(loaded); next_id_ = next;
}

} // namespace spiral::memory
