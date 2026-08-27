#include "spiral/media_flow.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace spiral::media_flow {
namespace {

std::size_t conditioning_dim(const PromptTemporalFlowConfig& config) {
    return config.latent_dim * 2 + config.text_feature_dim + config.time_feature_dim + config.position_feature_dim;
}

float positional_value(std::size_t position, std::size_t component, std::size_t dimensions) {
    if (dimensions == 0) return 0.0F;
    const std::size_t pair = component / 2;
    const double exponent = 2.0 * static_cast<double>(pair) / static_cast<double>(dimensions);
    const double angle = static_cast<double>(position) / std::pow(10000.0, exponent);
    return static_cast<float>(component % 2 == 0 ? std::sin(angle) : std::cos(angle));
}

Tensor build_conditioning(
    const PromptTemporalFlowConfig& config,
    const Tensor& noisy,
    std::string_view prompt,
    float diffusion_time) {
    if (noisy.rank() != 2 || noisy.shape()[0] == 0 || noisy.shape()[1] != config.latent_dim) {
        throw std::invalid_argument("prompt temporal flow requires [sequence,latent_dim]");
    }
    const std::size_t rows = noisy.shape()[0];
    const std::size_t width = conditioning_dim(config);
    Tensor out({rows, width});
    Tensor prompt_vec = multimodal::prompt_features(prompt, config.text_feature_dim);
    Tensor time_vec = flow::timestep_features(diffusion_time, config.time_feature_dim);
    std::vector<float> mean(config.latent_dim, 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t d = 0; d < config.latent_dim; ++d) {
            mean[d] += noisy.data()[row * config.latent_dim + d] / static_cast<float>(rows);
        }
    }
    for (std::size_t row = 0; row < rows; ++row) {
        std::size_t col = 0;
        const std::size_t base = row * width;
        for (std::size_t d = 0; d < config.latent_dim; ++d) out.data()[base + col++] = noisy.data()[row * config.latent_dim + d];
        for (float value : mean) out.data()[base + col++] = value;
        for (float value : prompt_vec.data()) out.data()[base + col++] = value;
        for (std::size_t d = 0; d < config.position_feature_dim; ++d) out.data()[base + col++] = positional_value(row, d, config.position_feature_dim);
        for (float value : time_vec.data()) out.data()[base + col++] = value;
    }
    return out;
}

Tensor silu(const Tensor& input) {
    Tensor out(input.shape());
    for (std::size_t i = 0; i < input.numel(); ++i) {
        const float x = input.data()[i];
        const float s = 1.0F / (1.0F + std::exp(-x));
        out.data()[i] = x * s;
    }
    return out;
}

float silu_derivative(float x) {
    const float s = 1.0F / (1.0F + std::exp(-x));
    return s * (1.0F + x * (1.0F - s));
}

Tensor linear_backward(nn::Linear& layer, const Tensor& input, const Tensor& grad_output) {
    if (input.rank() != 2 || grad_output.rank() != 2 || input.shape()[0] != grad_output.shape()[0] ||
        input.shape()[1] != layer.in_features() || grad_output.shape()[1] != layer.out_features()) {
        throw std::invalid_argument("media flow linear backward shape mismatch");
    }
    const std::size_t rows = input.shape()[0];
    const std::size_t in_dim = layer.in_features();
    const std::size_t out_dim = layer.out_features();
    auto& weight = layer.weight();
    weight.ensure_grad();
    nn::Parameter* bias = nullptr;
    if (layer.uses_bias()) { bias = &layer.bias(); bias->ensure_grad(); }
    Tensor grad_input({rows, in_dim});
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t out = 0; out < out_dim; ++out) {
            const float g = grad_output.data()[row * out_dim + out];
            if (bias != nullptr) bias->grad.data()[out] += g;
            for (std::size_t in = 0; in < in_dim; ++in) {
                weight.grad.data()[in * out_dim + out] += input.data()[row * in_dim + in] * g;
                grad_input.data()[row * in_dim + in] += weight.value.data()[in * out_dim + out] * g;
            }
        }
    }
    return grad_input;
}

float sequence_loss(const Tensor& prediction, const Tensor& target, float consistency_weight) {
    if (prediction.shape() != target.shape() || prediction.rank() != 2 || prediction.numel() == 0) {
        throw std::invalid_argument("sequence loss shape mismatch");
    }
    float loss = multimodal::mean_squared_error(prediction, target);
    const std::size_t rows = prediction.shape()[0];
    const std::size_t dim = prediction.shape()[1];
    if (rows > 1 && consistency_weight > 0.0F) {
        float consistency = 0.0F;
        for (std::size_t row = 1; row < rows; ++row) {
            for (std::size_t d = 0; d < dim; ++d) {
                const float pd = prediction.data()[row * dim + d] - prediction.data()[(row - 1) * dim + d];
                const float td = target.data()[row * dim + d] - target.data()[(row - 1) * dim + d];
                const float e = pd - td;
                consistency += e * e;
            }
        }
        loss += consistency_weight * consistency / static_cast<float>((rows - 1) * dim);
    }
    return loss;
}

Tensor sequence_gradient(const Tensor& prediction, const Tensor& target, float consistency_weight) {
    Tensor grad(prediction.shape());
    const float base_scale = 2.0F / static_cast<float>(prediction.numel());
    for (std::size_t i = 0; i < prediction.numel(); ++i) grad.data()[i] = base_scale * (prediction.data()[i] - target.data()[i]);
    const std::size_t rows = prediction.shape()[0];
    const std::size_t dim = prediction.shape()[1];
    if (rows > 1 && consistency_weight > 0.0F) {
        const float scale = 2.0F * consistency_weight / static_cast<float>((rows - 1) * dim);
        for (std::size_t row = 1; row < rows; ++row) {
            for (std::size_t d = 0; d < dim; ++d) {
                const std::size_t a = (row - 1) * dim + d;
                const std::size_t b = row * dim + d;
                const float e = (prediction.data()[b] - prediction.data()[a]) - (target.data()[b] - target.data()[a]);
                grad.data()[b] += scale * e;
                grad.data()[a] -= scale * e;
            }
        }
    }
    return grad;
}

template <typename T>
void write_value(std::ofstream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T read_value(std::ifstream& stream) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream) throw std::runtime_error("truncated prompt media flow checkpoint");
    return value;
}

} // namespace

PromptTemporalFlowModel::PromptTemporalFlowModel(PromptTemporalFlowConfig config, Random& rng)
    : config_(config),
      input_projection_(conditioning_dim(config), config.hidden_dim, rng, true),
      output_projection_(config.hidden_dim, config.latent_dim, rng, true) {
    if (config_.latent_dim == 0 || config_.text_feature_dim == 0 || config_.time_feature_dim == 0 ||
        config_.position_feature_dim == 0 || config_.hidden_dim == 0) {
        throw std::invalid_argument("invalid PromptTemporalFlowConfig");
    }
}

Tensor PromptTemporalFlowModel::conditioning_matrix(const Tensor& noisy_sequence, std::string_view prompt, float diffusion_time) const {
    return build_conditioning(config_, noisy_sequence, prompt, diffusion_time);
}

Tensor PromptTemporalFlowModel::predict_clean(const Tensor& noisy_sequence, std::string_view prompt, float diffusion_time) const {
    const Tensor conditioning = conditioning_matrix(noisy_sequence, prompt, diffusion_time);
    return output_projection_.forward(silu(input_projection_.forward(conditioning)));
}

Tensor PromptTemporalFlowModel::generate(
    std::string_view prompt,
    std::size_t sequence_length,
    std::size_t steps,
    std::uint64_t seed,
    float guidance_scale) const {
    if (sequence_length == 0 || steps == 0) throw std::invalid_argument("media flow generation requires non-zero sequence and steps");
    Random rng(seed);
    Tensor current = flow::gaussian_noise({sequence_length, config_.latent_dim}, rng);
    flow::NoiseScheduler scheduler;
    const auto times = scheduler.sampling_times(steps);
    for (std::size_t step = 0; step < steps; ++step) {
        const float time = times[step];
        const float next_time = times[step + 1];
        const Tensor unconditional = predict_clean(current, "", time);
        const Tensor conditional = predict_clean(current, prompt, time);
        const Tensor target = flow::guided_prediction(unconditional, conditional, guidance_scale);
        const float relaxation = time <= std::numeric_limits<float>::epsilon()
            ? 1.0F
            : std::clamp((time - next_time) / time, 0.0F, 1.0F);
        for (std::size_t i = 0; i < current.numel(); ++i) current.data()[i] += relaxation * (target.data()[i] - current.data()[i]);
    }
    return current;
}

std::vector<nn::Parameter*> PromptTemporalFlowModel::parameters() {
    auto out = input_projection_.parameters();
    auto tail = output_projection_.parameters();
    out.insert(out.end(), tail.begin(), tail.end());
    return out;
}

std::vector<const nn::Parameter*> PromptTemporalFlowModel::parameters() const {
    auto out = input_projection_.parameters();
    auto tail = output_projection_.parameters();
    out.insert(out.end(), tail.begin(), tail.end());
    return out;
}

PromptTemporalFlowTrainer::PromptTemporalFlowTrainer(
    PromptTemporalFlowModel& model,
    flow::NoiseScheduler scheduler,
    PromptTemporalTrainerConfig config)
    : model_(model), scheduler_(scheduler), config_(config), optimizer_(model.parameters(), config.optimizer) {
    if (config_.max_grad_norm <= 0.0F || config_.temporal_consistency_weight < 0.0F) throw std::invalid_argument("invalid prompt media trainer config");
}

float PromptTemporalFlowTrainer::evaluate(const PromptMediaExample& example, float diffusion_time, const Tensor& noise) const {
    if (example.latent_sequence.shape() != noise.shape() || example.latent_sequence.rank() != 2 ||
        example.latent_sequence.shape()[1] != model_.config().latent_dim) {
        throw std::invalid_argument("prompt media example/noise shape mismatch");
    }
    const Tensor noisy = scheduler_.add_noise(example.latent_sequence, noise, diffusion_time);
    return sequence_loss(model_.predict_clean(noisy, example.prompt, diffusion_time), example.latent_sequence, config_.temporal_consistency_weight);
}

float PromptTemporalFlowTrainer::train_step(const PromptMediaExample& example, float diffusion_time, const Tensor& noise) {
    if (example.latent_sequence.shape() != noise.shape() || example.latent_sequence.rank() != 2 ||
        example.latent_sequence.shape()[1] != model_.config().latent_dim) {
        throw std::invalid_argument("prompt media example/noise shape mismatch");
    }
    const Tensor noisy = scheduler_.add_noise(example.latent_sequence, noise, diffusion_time);
    const Tensor conditioning = build_conditioning(model_.config(), noisy, example.prompt, diffusion_time);
    const Tensor hidden_pre = model_.input_projection().forward(conditioning);
    const Tensor hidden = silu(hidden_pre);
    const Tensor prediction = model_.output_projection().forward(hidden);
    const float loss = sequence_loss(prediction, example.latent_sequence, config_.temporal_consistency_weight);

    optimizer_.zero_grad();
    Tensor grad_hidden = linear_backward(model_.output_projection(), hidden, sequence_gradient(prediction, example.latent_sequence, config_.temporal_consistency_weight));
    for (std::size_t i = 0; i < grad_hidden.numel(); ++i) grad_hidden.data()[i] *= silu_derivative(hidden_pre.data()[i]);
    (void)linear_backward(model_.input_projection(), conditioning, grad_hidden);
    auto parameters = model_.parameters();
    train::clip_grad_norm(parameters, config_.max_grad_norm);
    optimizer_.step();
    return loss;
}

PromptAudioGenerator::PromptAudioGenerator(
    media_generation::ComplexStftConfig stft,
    std::size_t frames_per_patch,
    const media_generation::ComplexAudioCodec& codec,
    const PromptTemporalFlowModel& flow_model)
    : stft_(stft), frames_per_patch_(frames_per_patch), codec_(&codec), flow_model_(&flow_model) {
    if (frames_per_patch_ == 0 || codec.config().latent_dim != flow_model.config().latent_dim) {
        throw std::invalid_argument("prompt audio generator dimension mismatch");
    }
}

Tensor PromptAudioGenerator::generate_latents(std::string_view prompt, std::size_t patch_count, std::size_t steps, std::uint64_t seed) const {
    return flow_model_->generate(prompt, patch_count, steps, seed, 1.5F);
}

audio::AudioBuffer PromptAudioGenerator::generate(
    std::string_view prompt,
    std::size_t patch_count,
    std::uint32_t sample_rate,
    std::size_t steps,
    std::uint64_t seed) const {
    const Tensor latents = generate_latents(prompt, patch_count, steps, seed);
    const Tensor patches = codec_->decode(latents);
    const std::size_t denominator = frames_per_patch_ * 2;
    if (codec_->config().patch_dim % denominator != 0) throw std::invalid_argument("complex audio codec patch dimension is incompatible with STFT patching");
    const std::size_t bins = codec_->config().patch_dim / denominator;
    const std::size_t frames = patch_count * frames_per_patch_;
    const Tensor spectrum = media_generation::complex_patches_to_spectrum(patches, frames, bins, frames_per_patch_);
    return media_generation::inverse_complex_stft(spectrum, sample_rate, stft_);
}

PromptVideoGenerator::PromptVideoGenerator(
    const media_generation::FrameEmbeddingDecoder& decoder,
    const PromptTemporalFlowModel& flow_model)
    : decoder_(&decoder), flow_model_(&flow_model) {
    if (decoder.config().embedding_dim != flow_model.config().latent_dim) throw std::invalid_argument("prompt video generator dimension mismatch");
}

std::vector<vision::RgbImage> PromptVideoGenerator::generate(
    std::string_view prompt,
    std::size_t frame_count,
    std::size_t steps,
    std::uint64_t seed) const {
    const Tensor embeddings = flow_model_->generate(prompt, frame_count, steps, seed, 1.5F);
    std::vector<vision::RgbImage> frames;
    frames.reserve(frame_count);
    for (std::size_t row = 0; row < frame_count; ++row) {
        Tensor embedding({flow_model_->config().latent_dim});
        std::copy_n(embeddings.data().begin() + static_cast<std::ptrdiff_t>(row * flow_model_->config().latent_dim), flow_model_->config().latent_dim, embedding.data().begin());
        frames.push_back(decoder_->decode(embedding));
    }
    return frames;
}

void save_prompt_media_flow(const PromptTemporalFlowModel& model, const std::string& path) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("failed to open prompt media flow checkpoint for writing");
    const std::uint32_t magic = 0x534D4631U;
    write_value(stream, magic);
    const auto& c = model.config();
    write_value(stream, static_cast<std::uint64_t>(c.latent_dim));
    write_value(stream, static_cast<std::uint64_t>(c.text_feature_dim));
    write_value(stream, static_cast<std::uint64_t>(c.time_feature_dim));
    write_value(stream, static_cast<std::uint64_t>(c.position_feature_dim));
    write_value(stream, static_cast<std::uint64_t>(c.hidden_dim));
    const auto params = model.parameters();
    write_value(stream, static_cast<std::uint64_t>(params.size()));
    for (const auto* parameter : params) {
        write_value(stream, static_cast<std::uint64_t>(parameter->value.numel()));
        stream.write(reinterpret_cast<const char*>(parameter->value.data().data()), static_cast<std::streamsize>(parameter->value.numel() * sizeof(float)));
    }
    if (!stream) throw std::runtime_error("failed while writing prompt media flow checkpoint");
}

void load_prompt_media_flow(PromptTemporalFlowModel& model, const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("failed to open prompt media flow checkpoint");
    if (read_value<std::uint32_t>(stream) != 0x534D4631U) throw std::runtime_error("invalid prompt media flow checkpoint magic");
    const auto& c = model.config();
    if (read_value<std::uint64_t>(stream) != c.latent_dim || read_value<std::uint64_t>(stream) != c.text_feature_dim ||
        read_value<std::uint64_t>(stream) != c.time_feature_dim || read_value<std::uint64_t>(stream) != c.position_feature_dim ||
        read_value<std::uint64_t>(stream) != c.hidden_dim) throw std::runtime_error("prompt media flow checkpoint config mismatch");
    auto params = model.parameters();
    if (read_value<std::uint64_t>(stream) != params.size()) throw std::runtime_error("prompt media flow checkpoint parameter count mismatch");
    for (auto* parameter : params) {
        if (read_value<std::uint64_t>(stream) != parameter->value.numel()) throw std::runtime_error("prompt media flow checkpoint tensor mismatch");
        stream.read(reinterpret_cast<char*>(parameter->value.data().data()), static_cast<std::streamsize>(parameter->value.numel() * sizeof(float)));
        if (!stream) throw std::runtime_error("truncated prompt media flow checkpoint weights");
    }
}

} // namespace spiral::media_flow
