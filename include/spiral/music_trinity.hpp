#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace spiral {

struct MusicFrameV1 {
    static constexpr std::uint32_t schema_version = 1;

    std::string track_id;
    std::string title;
    std::string artist;
    double position_seconds{0.0};

    double bpm{0.0};
    std::string estimated_key;

    double loudness{0.0};
    double brightness{0.0};
    double warmth{0.0};
    double spectral_density{0.0};
    double transient_density{0.0};
    double stereo_width{0.0};
    double harmonicity{0.0};

    double beat_phase{0.0};
    std::string section;
};

struct ProductionIntentV1 {
    static constexpr std::uint32_t schema_version = 1;

    std::string project_id;
    std::string instruction;

    double bpm{0.0};
    std::string key;
    double duration_seconds{10.0};
    std::uint64_t seed{0};
    double mutation_amount{0.35};

    double drum_density{0.5};
    double bass_weight{0.5};
    double vocal_space{0.5};
    double texture_grit{0.5};
    double transient_density{0.5};

    bool lock_drums{false};
    bool lock_bass{false};
    bool lock_melody{false};
    bool lock_harmony{false};
    bool lock_texture{false};
    bool lock_arrangement{false};

    std::string chord_progression;
    std::vector<std::string> arrangement;
};

struct MusicContinuityV1 {
    static constexpr std::uint32_t schema_version = 1;

    std::string active_project_id;
    std::string last_audition_path;
    std::string last_revision_note;
    std::vector<std::string> preference_notes;
};

} // namespace spiral
