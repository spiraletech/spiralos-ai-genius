#pragma once

#include "spiral/memory.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace spiral::organic {

enum class PerceptionKind {
    Greeting,
    Question,
    Identity,
    MemoryProbe,
    ActionRequest,
    Reflection,
    Statement,
};

struct State {
    std::uint64_t revision = 0;
    std::uint64_t turn_count = 0;
    float energy = 0.72F;
    float focus = 0.62F;
    float curiosity = 0.68F;
    float confidence = 0.48F;
    float warmth = 0.58F;
    float novelty = 0.50F;
    float coherence = 0.74F;
    std::string last_topic;
    std::string last_user_message;
    std::string last_reply;
};

struct Response {
    std::string text;
    PerceptionKind perception = PerceptionKind::Statement;
    State state;
    std::vector<memory::MemoryHit> recalled;
};

class OrganicMind final {
public:
    OrganicMind() = default;

    [[nodiscard]] Response respond(std::string_view user_message, std::string_view host_context = {});
    [[nodiscard]] const State& state() const noexcept { return state_; }
    [[nodiscard]] const memory::MemoryStore& memory() const noexcept { return memory_; }
    [[nodiscard]] std::size_t memory_count() const noexcept { return memory_.records().size(); }
    void adopt_reply(std::string reply) { state_.last_reply = std::move(reply); }

    void reset();
    void save(const std::string& path) const;
    void load(const std::string& path);

    [[nodiscard]] static std::string perception_name(PerceptionKind kind);

private:
    [[nodiscard]] static PerceptionKind perceive(std::string_view input);
    [[nodiscard]] static std::string extract_topic(std::string_view input);
    [[nodiscard]] std::string compose(
        std::string_view input,
        std::string_view host_context,
        PerceptionKind perception,
        const std::vector<memory::MemoryHit>& recalled) const;
    void update_state(std::string_view input, PerceptionKind perception, std::string_view topic, bool recalled_memory);

    State state_;
    memory::MemoryStore memory_;
};

} // namespace spiral::organic
