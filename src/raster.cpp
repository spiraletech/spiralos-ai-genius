#include "spiral/raster.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace spiral::raster {
namespace {

std::size_t checked_area(std::size_t width, std::size_t height) {
    if (width == 0 || height == 0) return 0;
    if (width > std::numeric_limits<std::size_t>::max() / height) {
        throw std::overflow_error("framebuffer area overflow");
    }
    return width * height;
}

std::uint64_t hash_string(std::string_view text) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char value : text) {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool parse_rgb(std::string_view text, Rgba8& out) {
    if (text.size() != 7 || text.front() != '#') return false;
    auto hex = [](char ch) -> int {
        if (ch >= '0' && ch <= '9') return ch - '0';
        if (ch >= 'a' && ch <= 'f') return 10 + ch - 'a';
        if (ch >= 'A' && ch <= 'F') return 10 + ch - 'A';
        return -1;
    };
    const int r0 = hex(text[1]); const int r1 = hex(text[2]);
    const int g0 = hex(text[3]); const int g1 = hex(text[4]);
    const int b0 = hex(text[5]); const int b1 = hex(text[6]);
    if (r0 < 0 || r1 < 0 || g0 < 0 || g1 < 0 || b0 < 0 || b1 < 0) return false;
    out = Rgba8{
        static_cast<std::uint8_t>(r0 * 16 + r1),
        static_cast<std::uint8_t>(g0 * 16 + g1),
        static_cast<std::uint8_t>(b0 * 16 + b1),
        255};
    return true;
}

Rgba8 property_color(const units::Component& component, std::string_view key, Rgba8 fallback) {
    const auto it = component.properties.find(std::string(key));
    if (it == component.properties.end()) return fallback;
    Rgba8 parsed;
    return parse_rgb(it->second, parsed) ? parsed : fallback;
}

std::vector<float> parse_samples(std::string_view text) {
    std::vector<float> values;
    std::string copy(text);
    std::stringstream stream(copy);
    std::string token;
    while (std::getline(stream, token, ',')) {
        try {
            const float value = std::stof(token);
            if (std::isfinite(value)) values.push_back(std::clamp(value, -1.0F, 1.0F));
        } catch (...) {
            // Ignore malformed waveform tokens; the surface stays deterministic and valid.
        }
    }
    return values;
}

std::array<std::uint8_t, 7> glyph_rows(char ch) noexcept {
    switch (ch) {
        case '0': return {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E};
        case '1': return {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E};
        case '2': return {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F};
        case '3': return {0x1E,0x01,0x01,0x0E,0x01,0x01,0x1E};
        case '4': return {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02};
        case '5': return {0x1F,0x10,0x10,0x1E,0x01,0x01,0x1E};
        case '6': return {0x0E,0x10,0x10,0x1E,0x11,0x11,0x0E};
        case '7': return {0x1F,0x01,0x02,0x04,0x08,0x08,0x08};
        case '8': return {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E};
        case '9': return {0x0E,0x11,0x11,0x0F,0x01,0x01,0x0E};
        case 'A': return {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11};
        case 'B': return {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E};
        case 'C': return {0x0F,0x10,0x10,0x10,0x10,0x10,0x0F};
        case 'D': return {0x1E,0x11,0x11,0x11,0x11,0x11,0x1E};
        case 'E': return {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F};
        case 'F': return {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10};
        case 'G': return {0x0F,0x10,0x10,0x17,0x11,0x11,0x0F};
        case 'H': return {0x11,0x11,0x11,0x1F,0x11,0x11,0x11};
        case 'I': return {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E};
        case 'J': return {0x07,0x02,0x02,0x02,0x12,0x12,0x0C};
        case 'K': return {0x11,0x12,0x14,0x18,0x14,0x12,0x11};
        case 'L': return {0x10,0x10,0x10,0x10,0x10,0x10,0x1F};
        case 'M': return {0x11,0x1B,0x15,0x15,0x11,0x11,0x11};
        case 'N': return {0x11,0x19,0x15,0x13,0x11,0x11,0x11};
        case 'O': return {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E};
        case 'P': return {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10};
        case 'Q': return {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D};
        case 'R': return {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11};
        case 'S': return {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E};
        case 'T': return {0x1F,0x04,0x04,0x04,0x04,0x04,0x04};
        case 'U': return {0x11,0x11,0x11,0x11,0x11,0x11,0x0E};
        case 'V': return {0x11,0x11,0x11,0x11,0x11,0x0A,0x04};
        case 'W': return {0x11,0x11,0x11,0x15,0x15,0x15,0x0A};
        case 'X': return {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11};
        case 'Y': return {0x11,0x11,0x0A,0x04,0x04,0x04,0x04};
        case 'Z': return {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F};
        case '-': return {0,0,0,0x1F,0,0,0};
        case '.': return {0,0,0,0,0,0x06,0x06};
        case ':': return {0,0x06,0x06,0,0x06,0x06,0};
        case '/': return {0x01,0x02,0x02,0x04,0x08,0x08,0x10};
        case ' ': return {0,0,0,0,0,0,0};
        default: {
            std::array<std::uint8_t, 7> rows{};
            std::uint8_t seed = static_cast<std::uint8_t>(ch);
            for (std::size_t row = 0; row < rows.size(); ++row) {
                seed = static_cast<std::uint8_t>(seed * 33U + 17U);
                rows[row] = static_cast<std::uint8_t>((seed >> 1U) & 0x1FU);
            }
            rows.front() |= 0x11U;
            rows.back() |= 0x11U;
            return rows;
        }
    }
}

} // namespace

Framebuffer::Framebuffer(std::size_t width, std::size_t height, Rgba8 clear) {
    resize(width, height, clear);
}

void Framebuffer::resize(std::size_t width, std::size_t height, Rgba8 clear) {
    width_ = width;
    height_ = height;
    pixels_.assign(checked_area(width, height), clear);
}

void Framebuffer::clear(Rgba8 color) {
    std::fill(pixels_.begin(), pixels_.end(), color);
}

std::size_t Framebuffer::index(std::size_t x, std::size_t y) const {
    if (x >= width_ || y >= height_) throw std::out_of_range("framebuffer pixel out of range");
    return y * width_ + x;
}

void Framebuffer::set_pixel(std::size_t x, std::size_t y, Rgba8 color) {
    pixels_[index(x, y)] = color;
}

Rgba8 Framebuffer::pixel(std::size_t x, std::size_t y) const {
    return pixels_[index(x, y)];
}

void Framebuffer::clear_rect(render::Rect rect, Rgba8 color) {
    if (width_ == 0 || height_ == 0) return;
    const int left = std::max(0, static_cast<int>(std::floor(rect.x)));
    const int top = std::max(0, static_cast<int>(std::floor(rect.y)));
    const int right = std::min(static_cast<int>(width_), static_cast<int>(std::ceil(rect.x + rect.width)));
    const int bottom = std::min(static_cast<int>(height_), static_cast<int>(std::ceil(rect.y + rect.height)));
    for (int y = top; y < bottom; ++y) {
        for (int x = left; x < right; ++x) {
            pixels_[static_cast<std::size_t>(y) * width_ + static_cast<std::size_t>(x)] = color;
        }
    }
}

std::uint64_t Framebuffer::hash64() const noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    auto mix = [&hash](std::uint8_t value) {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= 1099511628211ULL;
    };
    for (const auto& pixel_value : pixels_) {
        mix(pixel_value.r); mix(pixel_value.g); mix(pixel_value.b); mix(pixel_value.a);
    }
    for (std::size_t shift = 0; shift < sizeof(std::size_t); ++shift) {
        mix(static_cast<std::uint8_t>((width_ >> (shift * 8U)) & 0xFFU));
        mix(static_cast<std::uint8_t>((height_ >> (shift * 8U)) & 0xFFU));
    }
    return hash;
}

std::size_t Framebuffer::count_non_background(Rgba8 background) const noexcept {
    return static_cast<std::size_t>(std::count_if(pixels_.begin(), pixels_.end(), [background](Rgba8 pixel_value) {
        return !(pixel_value == background);
    }));
}

void Framebuffer::save_ppm(const std::string& path) const {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("failed to open framebuffer snapshot");
    file << "P6\n" << width_ << ' ' << height_ << "\n255\n";
    for (const auto& pixel_value : pixels_) {
        const std::array<char, 3> rgb{
            static_cast<char>(pixel_value.r),
            static_cast<char>(pixel_value.g),
            static_cast<char>(pixel_value.b)};
        file.write(rgb.data(), static_cast<std::streamsize>(rgb.size()));
    }
    if (!file) throw std::runtime_error("failed to write framebuffer snapshot");
}

SoftwareRenderer::SoftwareRenderer(RasterTheme theme) : theme_(theme) {}

void SoftwareRenderer::begin_frame(
    const render::FrameTree& tree,
    std::span<const render::Rect> dirty_regions,
    device::Device*) {
    viewport_ = tree.viewport;
    const auto width = static_cast<std::size_t>(std::max(1.0F, std::ceil(tree.viewport.width)));
    const auto height = static_cast<std::size_t>(std::max(1.0F, std::ceil(tree.viewport.height)));
    if (framebuffer_.width() != width || framebuffer_.height() != height) {
        framebuffer_.resize(width, height, theme_.background);
    }
    last_dirty_regions_.assign(dirty_regions.begin(), dirty_regions.end());
    for (const auto& region : dirty_regions) {
        framebuffer_.clear_rect(render::Rect{
            region.x - viewport_.x,
            region.y - viewport_.y,
            region.width,
            region.height}, theme_.background);
    }
}

void SoftwareRenderer::fill_rect(render::Rect rect, render::Rect clip, Rgba8 color) {
    const auto base = render::intersect_rect(rect, clip);
    if (base.width <= 0.0F || base.height <= 0.0F) return;

    auto draw_region = [&](render::Rect region) {
        if (region.width <= 0.0F || region.height <= 0.0F) return;
        const int left = std::max(0, static_cast<int>(std::floor(region.x - viewport_.x)));
        const int top = std::max(0, static_cast<int>(std::floor(region.y - viewport_.y)));
        const int right = std::min(static_cast<int>(framebuffer_.width()), static_cast<int>(std::ceil(region.x + region.width - viewport_.x)));
        const int bottom = std::min(static_cast<int>(framebuffer_.height()), static_cast<int>(std::ceil(region.y + region.height - viewport_.y)));
        for (int y = top; y < bottom; ++y) {
            for (int x = left; x < right; ++x) {
                framebuffer_.set_pixel(static_cast<std::size_t>(x), static_cast<std::size_t>(y), color);
            }
        }
    };

    if (last_dirty_regions_.empty()) {
        draw_region(base);
        return;
    }
    for (const auto& dirty : last_dirty_regions_) {
        draw_region(render::intersect_rect(base, dirty));
    }
}

void SoftwareRenderer::stroke_rect(render::Rect rect, render::Rect clip, Rgba8 color) {
    if (rect.width <= 0.0F || rect.height <= 0.0F) return;
    fill_rect(render::Rect{rect.x, rect.y, rect.width, 1.0F}, clip, color);
    fill_rect(render::Rect{rect.x, rect.y + rect.height - 1.0F, rect.width, 1.0F}, clip, color);
    fill_rect(render::Rect{rect.x, rect.y, 1.0F, rect.height}, clip, color);
    fill_rect(render::Rect{rect.x + rect.width - 1.0F, rect.y, 1.0F, rect.height}, clip, color);
}

void SoftwareRenderer::draw_text(std::string_view text, render::Rect rect, render::Rect clip, Rgba8 color) {
    constexpr float glyph_width = 5.0F;
    constexpr float glyph_height = 7.0F;
    constexpr float spacing = 1.0F;
    float cursor_x = rect.x + 3.0F;
    const float cursor_y = rect.y + std::max(2.0F, (rect.height - glyph_height) * 0.5F);
    for (char raw : text) {
        const char ch = raw >= 'a' && raw <= 'z' ? static_cast<char>(raw - 'a' + 'A') : raw;
        if (cursor_x + glyph_width > rect.x + rect.width - 1.0F) break;
        const auto rows = glyph_rows(ch);
        for (std::size_t row = 0; row < rows.size(); ++row) {
            for (std::size_t col = 0; col < 5; ++col) {
                const std::uint8_t mask = static_cast<std::uint8_t>(1U << (4U - static_cast<unsigned>(col)));
                if ((rows[row] & mask) != 0U) {
                    fill_rect(render::Rect{
                        cursor_x + static_cast<float>(col),
                        cursor_y + static_cast<float>(row),
                        1.0F,
                        1.0F}, clip, color);
                }
            }
        }
        cursor_x += glyph_width + spacing;
    }
}

void SoftwareRenderer::draw_waveform(
    std::string_view samples,
    render::Rect rect,
    render::Rect clip,
    Rgba8 color) {
    const auto values = parse_samples(samples);
    const float center = rect.y + rect.height * 0.5F;
    fill_rect(render::Rect{rect.x, center, rect.width, 1.0F}, clip, theme_.border);
    if (values.empty()) return;
    const float usable_height = std::max(1.0F, rect.height * 0.42F);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const float fraction = values.size() == 1 ? 0.5F : static_cast<float>(i) / static_cast<float>(values.size() - 1);
        const float x = rect.x + fraction * std::max(0.0F, rect.width - 1.0F);
        const float magnitude = std::abs(values[i]) * usable_height;
        const float y = values[i] >= 0.0F ? center - magnitude : center;
        fill_rect(render::Rect{x, y, 1.0F, std::max(1.0F, magnitude)}, clip, color);
    }
}

void SoftwareRenderer::draw_image_placeholder(
    const units::Component& component,
    const render::FrameNode& frame) {
    const auto source_it = component.properties.find("source");
    const std::string_view source = source_it == component.properties.end() ? std::string_view(component.id) : std::string_view(source_it->second);
    const std::uint64_t seed = hash_string(source);
    const Rgba8 first{
        static_cast<std::uint8_t>(48U + (seed & 0x7FU)),
        static_cast<std::uint8_t>(48U + ((seed >> 8U) & 0x7FU)),
        static_cast<std::uint8_t>(48U + ((seed >> 16U) & 0x7FU)), 255};
    const Rgba8 second{
        static_cast<std::uint8_t>(48U + ((seed >> 24U) & 0x7FU)),
        static_cast<std::uint8_t>(48U + ((seed >> 32U) & 0x7FU)),
        static_cast<std::uint8_t>(48U + ((seed >> 40U) & 0x7FU)), 255};
    constexpr float tile = 8.0F;
    const int columns = std::max(1, static_cast<int>(std::ceil(frame.bounds.width / tile)));
    const int rows = std::max(1, static_cast<int>(std::ceil(frame.bounds.height / tile)));
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            fill_rect(render::Rect{
                frame.bounds.x + static_cast<float>(column) * tile,
                frame.bounds.y + static_cast<float>(row) * tile,
                tile,
                tile}, frame.clip, ((row + column) % 2 == 0) ? first : second);
        }
    }
}

void SoftwareRenderer::draw_component(
    const units::Component& component,
    const render::FrameNode& frame) {
    using units::ComponentKind;
    const Rgba8 background = property_color(component, "background", theme_.panel);
    const Rgba8 foreground = property_color(component, "color", theme_.text);
    switch (component.kind) {
        case ComponentKind::Container:
        case ComponentKind::Grid:
        case ComponentKind::Canvas:
            fill_rect(frame.bounds, frame.clip, background);
            break;
        case ComponentKind::Button: {
            fill_rect(frame.bounds, frame.clip, property_color(component, "background", theme_.accent));
            stroke_rect(frame.bounds, frame.clip, theme_.border);
            const auto label = component.properties.find("label");
            draw_text(label == component.properties.end() ? component.id : label->second, frame.bounds, frame.clip, foreground);
            break;
        }
        case ComponentKind::Text: {
            const auto text = component.properties.find("text");
            draw_text(text == component.properties.end() ? component.id : text->second, frame.bounds, frame.clip, foreground);
            break;
        }
        case ComponentKind::ImageSurface:
        case ComponentKind::VideoSurface:
        case ComponentKind::Avatar:
            draw_image_placeholder(component, frame);
            stroke_rect(frame.bounds, frame.clip, theme_.border);
            break;
        case ComponentKind::AudioSurface:
            fill_rect(frame.bounds, frame.clip, theme_.panel_alt);
            stroke_rect(frame.bounds, frame.clip, theme_.accent);
            break;
        case ComponentKind::WaveformSurface: {
            fill_rect(frame.bounds, frame.clip, theme_.panel_alt);
            const auto samples = component.properties.find("samples");
            draw_waveform(samples == component.properties.end() ? std::string_view{} : std::string_view(samples->second), frame.bounds, frame.clip, theme_.waveform);
            stroke_rect(frame.bounds, frame.clip, theme_.border);
            break;
        }
        case ComponentKind::Custom:
            fill_rect(frame.bounds, frame.clip, theme_.panel_alt);
            stroke_rect(frame.bounds, frame.clip, theme_.border);
            break;
    }
}

void SoftwareRenderer::end_frame() {
    ++frame_count_;
}

} // namespace spiral::raster
