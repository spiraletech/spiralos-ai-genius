#include "spiral/media_flow.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

using namespace spiral;
using namespace spiral::media_flow;

namespace {

bool close_tensor(const Tensor& a, const Tensor& b, float eps = 1.0e-5F) {
    if (a.shape() != b.shape()) return false;
    for (std::size_t i = 0; i < a.numel(); ++i) if (std::abs(a.data()[i] - b.data()[i]) > eps) return false;
    return true;
}

Tensor trajectory_a() {
    Tensor t({4, 4});
    for (std::size_t r = 0; r < 4; ++r) {
        t.data()[r * 4 + 0] = 0.65F + 0.08F * static_cast<float>(r);
        t.data()[r * 4 + 1] = 0.10F * static_cast<float>(r);
        t.data()[r * 4 + 2] = -0.35F;
        t.data()[r * 4 + 3] = 0.20F;
    }
    return t;
}

Tensor trajectory_b() {
    Tensor t({4, 4});
    for (std::size_t r = 0; r < 4; ++r) {
        t.data()[r * 4 + 0] = -0.55F;
        t.data()[r * 4 + 1] = 0.05F;
        t.data()[r * 4 + 2] = 0.60F + 0.09F * static_cast<float>(r);
        t.data()[r * 4 + 3] = -0.12F * static_cast<float>(r);
    }
    return t;
}

} // namespace

int main() {
    PromptTemporalFlowConfig config;
    config.latent_dim = 4;
    config.text_feature_dim = 16;
    config.time_feature_dim = 8;
    config.position_feature_dim = 8;
    config.hidden_dim = 48;

    Random rng(1701);
    PromptTemporalFlowModel model(config, rng);
    PromptTemporalFlowTrainer trainer(model, flow::NoiseScheduler{}, {{0.01F, 0.9F, 0.999F, 1.0e-8F, 0.0F}, 5.0F, 0.15F});

    PromptMediaExample red{"red pulse", trajectory_a()};
    PromptMediaExample blue{"blue drift", trajectory_b()};
    Random noise_rng(1702);
    Tensor eval_noise = flow::gaussian_noise({4, 4}, noise_rng);
    const float red_initial = trainer.evaluate(red, 0.6F, eval_noise);
    const float blue_initial = trainer.evaluate(blue, 0.6F, eval_noise);

    Random train_noise_rng(1703);
    for (int step = 0; step < 900; ++step) {
        const float time = 0.2F + 0.2F * static_cast<float>(step % 4);
        Tensor noise = flow::gaussian_noise({4, 4}, train_noise_rng);
        (void)trainer.train_step((step % 2 == 0) ? red : blue, time, noise);
    }

    const float red_final = trainer.evaluate(red, 0.6F, eval_noise);
    const float blue_final = trainer.evaluate(blue, 0.6F, eval_noise);
    assert(std::isfinite(red_final) && std::isfinite(blue_final));
    assert(red_final < red_initial * 0.35F);
    assert(blue_final < blue_initial * 0.35F);

    const Tensor red_gen = model.generate("red pulse", 4, 10, 777, 1.5F);
    const Tensor red_again = model.generate("red pulse", 4, 10, 777, 1.5F);
    const Tensor blue_gen = model.generate("blue drift", 4, 10, 777, 1.5F);
    assert(close_tensor(red_gen, red_again));
    assert(!close_tensor(red_gen, blue_gen, 1.0e-3F));

    const auto checkpoint = std::filesystem::temp_directory_path() / "spiral_l17_media_flow.bin";
    save_prompt_media_flow(model, checkpoint.string());
    Random clone_rng(9999);
    PromptTemporalFlowModel clone(config, clone_rng);
    load_prompt_media_flow(clone, checkpoint.string());
    assert(close_tensor(model.predict_clean(eval_noise, "red pulse", 0.6F), clone.predict_clean(eval_noise, "red pulse", 0.6F), 1.0e-6F));
    std::filesystem::remove(checkpoint);

    media_generation::ComplexStftConfig stft{32, 16};
    Random codec_rng(1704);
    media_generation::ComplexAudioCodec codec({68, 4}, codec_rng);
    PromptAudioGenerator audio_generator(stft, 2, codec, model);
    const Tensor audio_red_latents = audio_generator.generate_latents("red pulse", 4, 8, 811);
    const Tensor audio_blue_latents = audio_generator.generate_latents("blue drift", 4, 8, 811);
    assert(!close_tensor(audio_red_latents, audio_blue_latents, 1.0e-3F));
    const auto audio = audio_generator.generate("red pulse", 4, 8000, 8, 811);
    assert(audio.channels() == 1 && audio.sample_rate() == 8000 && audio.frame_count() > 0);
    float audio_energy = 0.0F;
    for (float sample : audio.samples()) { assert(std::isfinite(sample)); audio_energy += std::abs(sample); }
    assert(audio_energy > 0.0F);

    Random frame_rng(1705);
    media_generation::FrameEmbeddingDecoder frame_decoder({4, 4, 4, 16}, frame_rng);
    PromptVideoGenerator video_generator(frame_decoder, model);
    const auto red_frames = video_generator.generate("red pulse", 4, 8, 912);
    const auto blue_frames = video_generator.generate("blue drift", 4, 8, 912);
    assert(red_frames.size() == 4 && blue_frames.size() == 4);
    bool any_pixel_diff = false;
    for (std::size_t i = 0; i < red_frames.size(); ++i) {
        assert(red_frames[i].width() == 4 && red_frames[i].height() == 4);
        for (std::size_t p = 0; p < red_frames[i].pixels().size(); ++p) {
            if (red_frames[i].pixels()[p] != blue_frames[i].pixels()[p]) any_pixel_diff = true;
        }
    }
    assert(any_pixel_diff);

    std::cout << "L17 red flow loss: " << red_initial << " -> " << red_final << '\n';
    std::cout << "L17 blue flow loss: " << blue_initial << " -> " << blue_final << '\n';
    std::cout << "L17 prompt-to-media flow tests passed\n";
    return 0;
}
