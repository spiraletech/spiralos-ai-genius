#include "spiral/organic_ai.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

int main() {
    using namespace spiral;

    organic::OrganicMind mind;
    const auto initial = mind.state();
    assert(initial.turn_count == 0);
    assert(mind.memory_count() == 0);

    const auto hello = mind.respond("hello spiral", "standalone Windows host");
    assert(hello.perception == organic::PerceptionKind::Greeting);
    assert(hello.state.turn_count == 1);
    assert(hello.text.find("Organic mode") != std::string::npos);
    assert(mind.memory_count() == 1);

    // Exact screenshot regression: state questions must return live values, not the old canned paragraph.
    const auto local_state = mind.respond("what is the local state", "standalone Windows host");
    assert(local_state.perception == organic::PerceptionKind::Question);
    assert(local_state.text.find("My local organic state right now") != std::string::npos);
    assert(local_state.text.find("focus") != std::string::npos);
    assert(local_state.text.find("durable") != std::string::npos);
    assert(local_state.text.find("For broad factual knowledge") == std::string::npos);

    // Exact screenshot regression: CPU question must answer CPU/self-knowledge and must not retrieve the prior state question.
    const auto processor = mind.respond("what is ur processor", "standalone Windows host");
    assert(processor.perception == organic::PerceptionKind::Question);
    assert(processor.text.find("native C++") != std::string::npos);
    assert(processor.text.find("CPU") != std::string::npos);
    assert(processor.text.find("what is the local state") == std::string::npos);
    assert(processor.recalled.empty());

    const auto host = mind.respond("what host are you running in?", "EtherPlay host with playback and media tools");
    assert(host.text.find("EtherPlay") != std::string::npos);

    const auto offline = mind.respond("do you need internet or a backend api?", "standalone Windows host");
    assert(offline.text.find("does not require") != std::string::npos);
    assert(offline.text.find("OpenAI") != std::string::npos);

    // Unknown world-knowledge questions must state the knowledge boundary instead of repeating a stock capability speech.
    const auto unknown = mind.respond("what is the capital city of neptune?", "standalone Windows host");
    assert(unknown.text.find("don't have enough native learned knowledge") != std::string::npos);
    assert(unknown.text.find("won't fabricate") != std::string::npos);

    const auto fact = mind.respond("My favorite synth color is cobalt blue", "standalone Windows host");
    assert(mind.memory_count() >= 1);
    assert(fact.state.last_topic == "favorite" || fact.state.last_topic == "cobalt");

    const auto recall = mind.respond("Do you remember my cobalt blue synth?", "standalone Windows host");
    assert(recall.perception == organic::PerceptionKind::MemoryProbe);
    assert(!recall.recalled.empty());
    assert(recall.text.find("cobalt blue") != std::string::npos);

    // Repeating a question must never retrieve that identical question as evidence for itself.
    const auto repeated_state = mind.respond("what is the local state", "standalone Windows host");
    for (const auto& hit : repeated_state.recalled) {
        assert(hit.record.text != "what is the local state");
    }
    assert(repeated_state.text.find("strongest related trace") == std::string::npos);

    const auto memory_summary = mind.respond("how many memories do you have?", "standalone Windows host");
    assert(memory_summary.text.find("durable organic memor") != std::string::npos);

    const auto before_save = mind.state();
    const std::filesystem::path path = "spiral_organic_state_test.bin";
    mind.save(path.string());
    assert(std::filesystem::exists(path));
    assert(std::filesystem::exists(path.string() + ".memory"));

    organic::OrganicMind loaded;
    loaded.load(path.string());
    const auto restored = loaded.state();
    assert(restored.revision == before_save.revision);
    assert(restored.turn_count == before_save.turn_count);
    assert(restored.last_topic == before_save.last_topic);
    assert(restored.last_user_message == before_save.last_user_message);
    assert(restored.last_reply == before_save.last_reply);
    assert(std::abs(restored.focus - before_save.focus) < 1.0e-7F);
    assert(std::abs(restored.curiosity - before_save.curiosity) < 1.0e-7F);
    assert(loaded.memory_count() == mind.memory_count());

    const auto continued = loaded.respond("That synth memory should persist", "EtherPlay host");
    assert(continued.state.turn_count == before_save.turn_count + 1);
    assert(loaded.memory_count() == mind.memory_count() + 1);

    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + ".memory");

    std::cout << "Organic AI response intelligence regression passed\n";
    return 0;
}
