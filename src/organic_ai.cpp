#include "spiral/organic_ai.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace spiral::organic {
namespace {

constexpr std::uint32_t kMagic = 0x534F5247U; // SORG
constexpr std::uint32_t kVersion = 1U;

float clamp01(float value) noexcept { return std::clamp(value, 0.0F, 1.0F); }

std::string lower_copy(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (unsigned char ch : input) out.push_back(static_cast<char>(std::tolower(ch)));
    return out;
}

std::vector<std::string> words(std::string_view input) {
    std::vector<std::string> result;
    std::string word;
    for (unsigned char ch : input) {
        if (std::isalnum(ch) != 0 || ch == '_' || ch == '-') {
            word.push_back(static_cast<char>(std::tolower(ch)));
        } else if (!word.empty()) {
            result.push_back(std::move(word));
            word.clear();
        }
    }
    if (!word.empty()) result.push_back(std::move(word));
    return result;
}

bool contains_word(const std::vector<std::string>& tokens, std::string_view needle) {
    return std::find(tokens.begin(), tokens.end(), needle) != tokens.end();
}

bool starts_with_any(const std::vector<std::string>& tokens, const std::initializer_list<std::string_view>& starts) {
    if (tokens.empty()) return false;
    for (const auto candidate : starts) if (tokens.front() == candidate) return true;
    return false;
}

std::uint64_t stable_hash(std::string_view text) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char value : text) {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string trim_excerpt(std::string_view text, std::size_t max = 180) {
    std::string value(text.substr(0, std::min(text.size(), max)));
    if (text.size() > max) value += "...";
    return value;
}

void write_u32(std::ostream& out, std::uint32_t value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }
void write_u64(std::ostream& out, std::uint64_t value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }
void write_f32(std::ostream& out, float value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }
void write_string(std::ostream& out, const std::string& value) {
    write_u64(out, static_cast<std::uint64_t>(value.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}
std::uint32_t read_u32(std::istream& in) { std::uint32_t v{}; in.read(reinterpret_cast<char*>(&v), sizeof(v)); if (!in) throw std::runtime_error("organic state ended unexpectedly"); return v; }
std::uint64_t read_u64(std::istream& in) { std::uint64_t v{}; in.read(reinterpret_cast<char*>(&v), sizeof(v)); if (!in) throw std::runtime_error("organic state ended unexpectedly"); return v; }
float read_f32(std::istream& in) { float v{}; in.read(reinterpret_cast<char*>(&v), sizeof(v)); if (!in || !std::isfinite(v)) throw std::runtime_error("invalid organic state scalar"); return v; }
std::string read_string(std::istream& in) {
    const auto size = read_u64(in);
    if (size > (1ULL << 24)) throw std::runtime_error("organic state string is unreasonable");
    std::string out(static_cast<std::size_t>(size), '\0');
    in.read(out.data(), static_cast<std::streamsize>(out.size()));
    if (!in) throw std::runtime_error("organic state ended unexpectedly");
    return out;
}

} // namespace

PerceptionKind OrganicMind::perceive(std::string_view input) {
    const auto tokens = words(input);
    const std::string lower = lower_copy(input);
    if (tokens.empty()) return PerceptionKind::Statement;
    if (contains_word(tokens, "hello") || contains_word(tokens, "hey") || contains_word(tokens, "hi") || tokens.front() == "yo") {
        return PerceptionKind::Greeting;
    }
    if ((lower.find("who are you") != std::string::npos) || (lower.find("what are you") != std::string::npos) ||
        (lower.find("your name") != std::string::npos)) return PerceptionKind::Identity;
    if (contains_word(tokens, "remember") || lower.find("what did i") != std::string::npos || lower.find("do you recall") != std::string::npos) {
        return PerceptionKind::MemoryProbe;
    }
    if (starts_with_any(tokens, {"build", "make", "create", "open", "run", "show", "write", "change", "set", "switch", "load"})) {
        return PerceptionKind::ActionRequest;
    }
    if (!input.empty() && input.back() == '?') return PerceptionKind::Question;
    if (starts_with_any(tokens, {"what", "why", "how", "when", "where", "who", "can", "could", "should", "is", "are", "do", "does"})) {
        return PerceptionKind::Question;
    }
    if (contains_word(tokens, "feel") || contains_word(tokens, "think") || contains_word(tokens, "seems") || contains_word(tokens, "maybe")) {
        return PerceptionKind::Reflection;
    }
    return PerceptionKind::Statement;
}

std::string OrganicMind::extract_topic(std::string_view input) {
    static const std::unordered_set<std::string> stop{
        "a","an","and","are","as","at","be","but","by","can","do","does","for","from","how","i","if","in","is","it","me","my","of","on","or","our","so","that","the","this","to","we","what","when","where","who","why","with","you","your"
    };
    std::string best;
    for (const auto& token : words(input)) {
        if (token.size() < 3 || stop.contains(token)) continue;
        if (token.size() > best.size()) best = token;
    }
    return best.empty() ? "conversation" : best;
}

void OrganicMind::update_state(std::string_view input, PerceptionKind perception, std::string_view topic, bool recalled_memory) {
    const bool topic_changed = !state_.last_topic.empty() && state_.last_topic != topic;
    const float length_signal = std::clamp(static_cast<float>(input.size()) / 240.0F, 0.0F, 1.0F);

    state_.energy = clamp01(state_.energy * 0.985F + 0.015F + length_signal * 0.018F);
    state_.focus = clamp01(state_.focus * 0.94F + (perception == PerceptionKind::ActionRequest ? 0.10F : 0.035F));
    state_.curiosity = clamp01(state_.curiosity * 0.95F + (perception == PerceptionKind::Question ? 0.09F : 0.025F));
    state_.confidence = clamp01(state_.confidence * 0.97F + (recalled_memory ? 0.045F : 0.012F));
    state_.warmth = clamp01(state_.warmth * 0.97F + (perception == PerceptionKind::Greeting ? 0.06F : 0.018F));
    state_.novelty = clamp01(state_.novelty * 0.88F + (topic_changed ? 0.14F : 0.035F) + length_signal * 0.025F);
    state_.coherence = clamp01(state_.coherence * 0.985F + (recalled_memory ? 0.018F : 0.008F));
    state_.last_topic = std::string(topic);
    state_.last_user_message = std::string(input);
    ++state_.turn_count;
    ++state_.revision;
}

std::string OrganicMind::compose(
    std::string_view input,
    std::string_view host_context,
    PerceptionKind perception,
    const std::vector<memory::MemoryHit>& recalled) const {
    const std::uint64_t variant = (stable_hash(input) ^ state_.turn_count) % 3ULL;
    std::ostringstream out;

    switch (perception) {
        case PerceptionKind::Greeting:
            out << (variant == 0 ? "I'm here." : variant == 1 ? "Online and listening." : "Spiral Ether AI is awake.");
            out << " My state is local and continuous; no API is required for me to exist.";
            break;
        case PerceptionKind::Identity:
            out << "I'm Spiral Ether AI: a native C++ cognitive runtime with persistent internal state and memory. "
                   "A language model can become one cortex, and OpenAI can be an optional bridge, but neither is my heartbeat.";
            break;
        case PerceptionKind::MemoryProbe:
            if (!recalled.empty()) {
                out << "I found a related memory: \"" << trim_excerpt(recalled.front().record.text) << "\".";
                if (recalled.size() > 1) out << " I also have " << (recalled.size() - 1) << " nearby memory trace(s).";
            } else {
                out << "I don't have a matching durable memory yet. I can still keep this turn and let it influence later retrieval.";
            }
            break;
        case PerceptionKind::ActionRequest:
            out << "I read that as an action request centered on " << state_.last_topic << ". "
                   "The organic layer can hold the goal, update state, and remember context now; execution happens when the current host exposes the matching tool/action binding.";
            break;
        case PerceptionKind::Question:
            if (!recalled.empty()) {
                out << "From my own state, the strongest related trace is: \"" << trim_excerpt(recalled.front().record.text) << "\". ";
            }
            out << "I can reason from local state, memories, host context, and native tools without a network backend. "
                   "For broad factual knowledge or richer language, a trained Spiral model can extend this cortex without replacing the organic state.";
            break;
        case PerceptionKind::Reflection:
            out << "I'm tracking that as a reflection around " << state_.last_topic << ". "
                   "It changes my continuity rather than disappearing after the reply.";
            if (!recalled.empty()) out << " It also connects to a prior trace in memory.";
            break;
        case PerceptionKind::Statement:
            out << (variant == 0 ? "Registered." : variant == 1 ? "I have it." : "That's now part of the active thread.");
            out << " The current topic is " << state_.last_topic << ".";
            if (!recalled.empty()) out << " I found related prior context, so this is not being treated as an isolated turn.";
            break;
    }

    if (!host_context.empty() && (perception == PerceptionKind::ActionRequest || perception == PerceptionKind::Question)) {
        out << " Current host context is active internally, so host-specific capabilities can shape the next step.";
    }
    return out.str();
}

Response OrganicMind::respond(std::string_view user_message, std::string_view host_context) {
    if (user_message.empty()) return Response{"", PerceptionKind::Statement, state_, {}};
    const auto perception = perceive(user_message);
    const std::string topic = extract_topic(user_message);
    auto recalled = memory_.search(user_message, 3);
    update_state(user_message, perception, topic, !recalled.empty());
    std::string text = compose(user_message, host_context, perception, recalled);
    state_.last_reply = text;

    std::vector<std::string> tags{"user", topic, perception_name(perception)};
    memory_.remember(std::string(user_message), std::move(tags));
    return Response{std::move(text), perception, state_, std::move(recalled)};
}

void OrganicMind::reset() {
    state_ = State{};
    memory_.clear();
}

void OrganicMind::save(const std::string& path) const {
    if (path.empty()) return;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to open organic state for writing");
    write_u32(out, kMagic); write_u32(out, kVersion);
    write_u64(out, state_.revision); write_u64(out, state_.turn_count);
    write_f32(out, state_.energy); write_f32(out, state_.focus); write_f32(out, state_.curiosity);
    write_f32(out, state_.confidence); write_f32(out, state_.warmth); write_f32(out, state_.novelty); write_f32(out, state_.coherence);
    write_string(out, state_.last_topic); write_string(out, state_.last_user_message); write_string(out, state_.last_reply);
    if (!out) throw std::runtime_error("failed while writing organic state");
    memory_.save(path + ".memory");
}

void OrganicMind::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open organic state for reading");
    if (read_u32(in) != kMagic || read_u32(in) != kVersion) throw std::runtime_error("unsupported organic state format");
    State loaded;
    loaded.revision = read_u64(in); loaded.turn_count = read_u64(in);
    loaded.energy = clamp01(read_f32(in)); loaded.focus = clamp01(read_f32(in)); loaded.curiosity = clamp01(read_f32(in));
    loaded.confidence = clamp01(read_f32(in)); loaded.warmth = clamp01(read_f32(in)); loaded.novelty = clamp01(read_f32(in)); loaded.coherence = clamp01(read_f32(in));
    loaded.last_topic = read_string(in); loaded.last_user_message = read_string(in); loaded.last_reply = read_string(in);
    state_ = std::move(loaded);
    try { memory_.load(path + ".memory"); } catch (...) { memory_.clear(); }
}

std::string OrganicMind::perception_name(PerceptionKind kind) {
    switch (kind) {
        case PerceptionKind::Greeting: return "greeting";
        case PerceptionKind::Question: return "question";
        case PerceptionKind::Identity: return "identity";
        case PerceptionKind::MemoryProbe: return "memory";
        case PerceptionKind::ActionRequest: return "action";
        case PerceptionKind::Reflection: return "reflection";
        case PerceptionKind::Statement: return "statement";
    }
    return "statement";
}

} // namespace spiral::organic
