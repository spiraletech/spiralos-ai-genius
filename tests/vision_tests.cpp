#include "spiral/vision.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>

namespace {
bool near(float a, float b, float epsilon = 1.0e-5F) { return std::fabs(a - b) <= epsilon; }
}

int main() {
    using spiral::Random;
    using spiral::Tensor;
    using namespace spiral::vision;

    RgbImage image(4, 4);
    for (std::size_t y = 0; y < image.height(); ++y) {
        for (std::size_t x = 0; x < image.width(); ++x) {
            image.at(x, y, 0) = static_cast<std::uint8_t>(x * 40 + y * 3);
            image.at(x, y, 1) = static_cast<std::uint8_t>(y * 50 + x * 2);
            image.at(x, y, 2) = static_cast<std::uint8_t>((x + y) * 20);
        }
    }

    const auto ppm_path = std::filesystem::temp_directory_path() / "spiral_l8_roundtrip.ppm";
    image.save_ppm(ppm_path.string());
    const RgbImage loaded = RgbImage::load_ppm(ppm_path.string());
    assert(loaded.width() == 4 && loaded.height() == 4);
    assert(loaded.pixels() == image.pixels());
    std::filesystem::remove(ppm_path);

    const Tensor image_tensor = image_to_tensor(image);
    assert(image_tensor.shape() == std::vector<std::size_t>({4, 4, 3}));
    const RgbImage roundtrip = tensor_to_image(image_tensor);
    assert(roundtrip.pixels() == image.pixels());

    const Tensor patches = patchify(image_tensor, 2);
    assert(patches.shape() == std::vector<std::size_t>({4, 12}));
    assert(near(patches.data()[0], static_cast<float>(image.at(0, 0, 0)) / 255.0F));
    assert(near(patches.data()[11], static_cast<float>(image.at(1, 1, 2)) / 255.0F));

    VisionConfig config;
    config.patch_size = 2;
    config.model_dim = 8;
    config.num_heads = 2;
    config.num_layers = 1;
    config.ffn_hidden_dim = 16;
    config.embedding_dim = 6;

    Random rng_a(8080);
    Random rng_b(8080);
    VisionEncoder encoder_a(config, rng_a);
    VisionEncoder encoder_b(config, rng_b);
    const Tensor tokens_a = encoder_a.encode_tokens(image);
    const Tensor tokens_b = encoder_b.encode_tokens(image);
    assert(tokens_a.shape() == std::vector<std::size_t>({4, 6}));
    assert(tokens_a.data().size() == tokens_b.data().size());
    for (std::size_t i = 0; i < tokens_a.numel(); ++i) assert(near(tokens_a.data()[i], tokens_b.data()[i]));

    const Tensor pooled = encoder_a.encode_pooled(image);
    assert(pooled.shape() == std::vector<std::size_t>({6}));

    RgbImage changed = image;
    changed.at(3, 3, 0) = static_cast<std::uint8_t>(255 - changed.at(3, 3, 0));
    changed.at(3, 3, 1) = static_cast<std::uint8_t>(255 - changed.at(3, 3, 1));
    changed.at(3, 3, 2) = static_cast<std::uint8_t>(255 - changed.at(3, 3, 2));
    const Tensor changed_tokens = encoder_a.encode_tokens(changed);
    bool first_patch_changed = false;
    for (std::size_t d = 0; d < config.embedding_dim; ++d) {
        if (!near(tokens_a.data()[d], changed_tokens.data()[d], 1.0e-6F)) first_patch_changed = true;
    }
    assert(first_patch_changed);

    Random projector_rng(991);
    CrossModalProjector projector(8, 6, 5, projector_rng);
    Tensor text_features({8}, 0.25F);
    const Tensor shared_text = projector.project_text(text_features);
    const Tensor shared_vision = projector.project_vision(pooled);
    assert(shared_text.shape() == std::vector<std::size_t>({5}));
    assert(shared_vision.shape() == std::vector<std::size_t>({5}));

    Random decoder_rng_a(1234);
    Random decoder_rng_b(1234);
    LatentRasterDecoder decoder_a({4, 2}, decoder_rng_a);
    LatentRasterDecoder decoder_b({4, 2}, decoder_rng_b);
    Random latent_rng_a(777);
    Random latent_rng_b(777);
    const Tensor latent_a = decoder_a.sample_latent(2, 3, latent_rng_a);
    const Tensor latent_b = decoder_b.sample_latent(2, 3, latent_rng_b);
    assert(latent_a.shape() == std::vector<std::size_t>({2, 3, 4}));
    for (std::size_t i = 0; i < latent_a.numel(); ++i) assert(near(latent_a.data()[i], latent_b.data()[i]));

    const RgbImage generated_a = decoder_a.decode(latent_a);
    const RgbImage generated_b = decoder_b.decode(latent_b);
    assert(generated_a.width() == 6 && generated_a.height() == 4);
    assert(generated_a.pixels() == generated_b.pixels());

    std::cout << "spiral_vision_tests: PASS tokens=" << tokens_a.shape()[0]
              << " embedding=" << tokens_a.shape()[1]
              << " generated=" << generated_a.width() << 'x' << generated_a.height() << '\n';
    return 0;
}
