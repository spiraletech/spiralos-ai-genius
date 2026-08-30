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
    assert(hello.text.find("no API") != std::string::npos || hello.text.find("No API") != std::string::npos);
    assert(mind.memory_count() == 1);

    const auto fact = mind.respond("My favorite synth color is cobalt blue", "standalone Windows host");
    assert(fact.state.turn_count == 2);
    assert(mind.memory_count() == 2);
    assert(fact.state.last_topic == "favorite" || fact.state.last_topic == "cobalt");

    const auto recall = mind.respond("Do you remember my cobalt blue synth?", "standalone Windows host");
    assert(recall.perception == organic::PerceptionKind::MemoryProbe);
    assert(!recall.recalled.empty());
    assert(recall.text.find("memory") != std::string::npos || recall.text.find("Memory") != std::string::npos);
    assert(mind.memory_count() == 3);

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

    std::cout << "Organic AI state regression passed\n";
    return 0;
}
