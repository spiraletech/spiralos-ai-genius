#include "spiral/organic_ai.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

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

std::string trim_copy(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n\0", 0);
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n\0");
    return value.substr(first, last - first + 1);
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

bool contains_any(std::string_view lower, const std::initializer_list<std::string_view>& needles) {
    for (const auto needle : needles) {
        if (lower.find(needle) != std::string_view::npos) return true;
    }
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

const std::unordered_set<std::string>& semantic_stop_words() {
    static const std::unordered_set<std::string> stop{
        "a","an","and","are","as","at","be","been","but","by","can","could","did","do","does","for","from","had","has","have","how","i","if","in","is","it","its","me","my","of","on","or","our","so","that","the","this","to","u","ur","was","we","were","what","when","where","which","who","why","will","with","would","you","your"
    };
    return stop;
}

std::unordered_set<std::string> meaningful_terms(std::string_view input) {
    std::unordered_set<std::string> result;
    const auto& stop = semantic_stop_words();
    for (const auto& token : words(input)) {
        if (token.size() < 3 || stop.contains(token)) continue;
        result.insert(token);
    }
    return result;
}

bool meaningful_overlap(std::string_view a, std::string_view b) {
    const auto left = meaningful_terms(a);
    const auto right = meaningful_terms(b);
    if (left.empty() || right.empty()) return false;
    for (const auto& term : left) if (right.contains(term)) return true;
    return false;
}

bool broad_memory_probe(std::string_view lower) {
    return contains_any(lower, {
        "what do you remember", "what did i say", "what have i told", "what do u remember",
        "remember anything", "show memories", "your memories", "memory count"
    });
}

std::string cpu_brand_string() {
    std::array<char, 49> brand{};

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int info[4]{};
    __cpuid(info, static_cast<int>(0x80000000U));
    const unsigned max_extended = static_cast<unsigned>(info[0]);
    if (max_extended >= 0x80000004U) {
        for (unsigned i = 0; i < 3; ++i) {
            __cpuid(info, static_cast<int>(0x80000002U + i));
            std::memcpy(brand.data() + i * 16U, info, 16U);
        }
    }
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    const unsigned max_extended = __get_cpuid_max(0x80000000U, nullptr);
    if (max_extended >= 0x80000004U) {
        for (unsigned i = 0; i < 3; ++i) {
            unsigned eax{}, ebx{}, ecx{}, edx{};
            if (__get_cpuid(0x80000002U + i, &eax, &ebx, &ecx, &edx) == 0) break;
            const unsigned regs[4]{eax, ebx, ecx, edx};
            std::memcpy(brand.data() + i * 16U, regs, 16U);
        }
    }
#endif

    std::string value = trim_copy(std::string(brand.data()));
    if (!value.empty()) return value;

#if defined(_M_X64) || defined(__x86_64__)
    return "x86-64 CPU";
#elif defined(_M_IX86) || defined(__i386__)
    return "x86 CPU";
#elif defined(_M_ARM64) || defined(__aarch64__)
    return "ARM64 CPU";
#elif defined(_M_ARM) || defined(__arm__)
    return "ARM CPU";
#else
    return "native host CPU";
#endif
}

std::string level_word(float value) {
    if (value >= 0.75F) return "high";
    if (value >= 0.45F) return "mid";
    return "low";
}

std::string self_knowledge_reply(
    std::string_view input,
    std::string_view host_context,
    const State& state,
    std::size_t memory_count) {
    const std::string lower = lower_copy(input);
    std::ostringstream out;

    if (contains_any(lower, {"local state", "organic state", "mind state", "internal state", "current state", "how are you internally", "how are u internally"})) {
        out << "My local organic state right now: "
            << "focus " << level_word(state.focus) << " (" << static_cast<int>(std::lround(state.focus * 100.0F)) << "%), "
            << "curiosity " << level_word(state.curiosity) << " (" << static_cast<int>(std::lround(state.curiosity * 100.0F)) << "%), "
            << "coherence " << level_word(state.coherence) << " (" << static_cast<int>(std::lround(state.coherence * 100.0F)) << "%), and "
            << "energy " << level_word(state.energy) << " (" << static_cast<int>(std::lround(state.energy * 100.0F)) << "%). ";
        out << "I've processed " << state.turn_count << " organic turn" << (state.turn_count == 1 ? "" : "s")
            << " and hold " << memory_count << " durable memor" << (memory_count == 1 ? "y" : "ies") << ".";
        if (!state.last_topic.empty()) out << " Active topic: " << state.last_topic << ".";
        return out.str();
    }

    if (contains_any(lower, {"processor", " cpu", "cpu ", "what cpu", "which cpu", "hardware are you on", "hardware r u on"})) {
        out << "I'm running as native C++ on your machine's CPU: " << cpu_brand_string() << ". "
            << "That processor executes my organic state/runtime code; GPU compute is a separate Spiral acceleration path.";
        return out.str();
    }

    if (contains_any(lower, {"gpu", "graphics card", "graphics processor"})) {
        return "My organic mind does not own the live GPU adapter record directly yet. The native shell does: /gpu shows the exact D3D11 adapter and execution path. I won't invent a GPU name from memory.";
    }

    if (contains_any(lower, {"how many memories", "memory count", "your memory", "your memories", "organic memory"})) {
        out << "I currently hold " << memory_count << " durable organic memor" << (memory_count == 1 ? "y" : "ies")
            << ". The visible chat can be cleared without erasing those; /resetorganic deliberately wipes them.";
        return out.str();
    }

    if (contains_any(lower, {"what host", "which host", "where are you running", "where r u running", "host context"})) {
        if (host_context.empty()) return "I'm in the native Spiral Ether AI runtime, but this host did not provide a host-context description.";
        out << "My active host context says: " << host_context;
        return out.str();
    }

    if (contains_any(lower, {"need internet", "need network", "without internet", "without network", "offline", "api required", "need api", "backend api"})) {
        return "Organic mode is local and does not require an API key, internet connection, or hosted model. Network access only happens if you explicitly switch to the optional OpenAI bridge.";
    }

    if (contains_any(lower, {"what model", "which model", "language model", "local cortex", "trained model"})) {
        return "My organic mind is not a language-model backend. A trained Spiral model can be loaded as an optional local language cortex; the organic state and durable memory remain separate from that cortex.";
    }

    if (contains_any(lower, {"what can you do", "what can u do", "your capabilities", "capabilities do you", "capabilities do u"})) {
        return "Natively I can maintain persistent organic state, form and retrieve local memories, track host context, inspect parts of my runtime, and route into Spiral tools when a host binds them. I do not yet have frontier-level learned world knowledge; that belongs to the trained Spiral cortex we still need to build.";
    }

    if (contains_any(lower, {"who are you", "who r u", "what are you", "what r u", "your name"})) {
        return "I'm Spiral Ether AI: the persistent native C++ mind/runtime. The organic state is my continuity; a trained Spiral model can become a language cortex, and external APIs remain optional bridges rather than my identity.";
    }

    return {};
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
    std::string best;
    const auto& stop = semantic_stop_words();
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
    if (const std::string direct = self_knowledge_reply(input, host_context, state_, memory_.records().size() + 1U); !direct.empty()) {
        return direct;
    }

    const std::uint64_t variant = (stable_hash(input) ^ state_.turn_count) % 3ULL;
    std::ostringstream out;

    switch (perception) {
        case PerceptionKind::Greeting:
            out << (variant == 0 ? "Hey. I'm here." : variant == 1 ? "Online." : "Yep — I'm awake.");
            out << " Organic mode is running locally.";
            break;
        case PerceptionKind::Identity:
            out << "I'm Spiral Ether AI: a native C++ cognitive runtime with persistent internal state and durable memory. "
                   "A language model can extend my language/knowledge, but it is not required for my continuity.";
            break;
        case PerceptionKind::MemoryProbe:
            if (!recalled.empty()) {
                out << "Yes. The strongest matching thing you told me was: \"" << trim_excerpt(recalled.front().record.text) << "\".";
                if (recalled.size() > 1) out << " I found " << (recalled.size() - 1) << " other related trace" << (recalled.size() == 2 ? "" : "s") << ".";
            } else {
                out << "I don't have a matching durable memory for that yet.";
            }
            break;
        case PerceptionKind::ActionRequest:
            out << "I understand the goal around " << state_.last_topic << ". ";
            if (!host_context.empty()) {
                out << "I can keep the goal/state locally, but execution still depends on this host exposing the matching native tool binding.";
            } else {
                out << "I can keep the goal/state locally, but I don't have an execution tool bound for it in this context yet.";
            }
            break;
        case PerceptionKind::Question:
            if (!recalled.empty()) {
                out << "I have relevant local context from earlier: \"" << trim_excerpt(recalled.front().record.text) << "\". ";
                out << "That gives me context, but not enough learned knowledge to invent a factual answer beyond it.";
            } else {
                out << "I don't have enough native learned knowledge to answer that reliably yet. ";
                out << "I can answer about my own state/runtime, reason over what you've taught me, and use host tools when they're bound. "
                       "For this question I need the trained Spiral language cortex or a native tool/data source; I won't fabricate an answer.";
            }
            break;
        case PerceptionKind::Reflection:
            out << "I'm following you. I'm keeping this connected to the active thread around " << state_.last_topic << ".";
            if (!recalled.empty()) out << " It also overlaps with something you told me earlier.";
            break;
        case PerceptionKind::Statement:
            out << (variant == 0 ? "Got it." : variant == 1 ? "I have that." : "Okay — stored.");
            if (!recalled.empty()) out << " It connects to an earlier memory rather than landing as an isolated turn.";
            break;
    }

    return out.str();
}

Response OrganicMind::respond(std::string_view user_message, std::string_view host_context) {
    if (user_message.empty()) return Response{"", PerceptionKind::Statement, state_, {}};
    const auto perception = perceive(user_message);
    const std::string topic = extract_topic(user_message);
    const std::string lower = lower_copy(user_message);

    auto recalled = memory_.search(user_message, 8);
    recalled.erase(std::remove_if(recalled.begin(), recalled.end(), [&](const memory::MemoryHit& hit) {
        const bool exact_duplicate = lower_copy(hit.record.text) == lower;
        const float threshold = perception == PerceptionKind::MemoryProbe ? 0.08F : 0.18F;
        return exact_duplicate || hit.score < threshold || !meaningful_overlap(user_message, hit.record.text);
    }), recalled.end());
    if (recalled.size() > 3) recalled.resize(3);

    if (perception == PerceptionKind::MemoryProbe && recalled.empty() && broad_memory_probe(lower)) {
        const auto& records = memory_.records();
        const std::size_t take = std::min<std::size_t>(3, records.size());
        for (std::size_t i = 0; i < take; ++i) {
            const auto& record = records[records.size() - 1U - i];
            recalled.push_back(memory::MemoryHit{record, 1.0F - static_cast<float>(i) * 0.1F});
        }
    }

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
