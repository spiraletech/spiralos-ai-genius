#pragma once

#include "spiral/renderer.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::raster {

struct Rgba8 {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
    std::uint8_t a = 255;

    [[nodiscard]] bool operator==(const Rgba8&) const noexcept = default;
};

class Framebuffer final {
public:
    Framebuffer() = default;
    Framebuffer(std::size_t width, std::size_t height, Rgba8 clear = {});

    void resize(std::size_t width, std::size_t height, Rgba8 clear = {});
    void clear(Rgba8 color = {});
    void clear_rect(render::Rect rect, Rgba8 color = {});
    void set_pixel(std::size_t x, std::size_t y, Rgba8 color);

    [[nodiscard]] Rgba8 pixel(std::size_t x, std::size_t y) const;
    [[nodiscard]] std::size_t width() const noexcept { return width_; }
    [[nodiscard]] std::size_t height() const noexcept { return height_; }
    [[nodiscard]] const std::vector<Rgba8>& pixels() const noexcept { return pixels_; }
    [[nodiscard]] std::uint64_t hash64() const noexcept;
    [[nodiscard]] std::size_t count_non_background(Rgba8 background) const noexcept;

    void save_ppm(const std::string& path) const;

private:
    [[nodiscard]] std::size_t index(std::size_t x, std::size_t y) const;

    std::size_t width_ = 0;
    std::size_t height_ = 0;
    std::vector<Rgba8> pixels_;
};

struct RasterTheme {
    Rgba8 background{10, 12, 16, 255};
    Rgba8 panel{24, 28, 36, 255};
    Rgba8 panel_alt{34, 40, 50, 255};
    Rgba8 text{232, 236, 244, 255};
    Rgba8 accent{112, 164, 255, 255};
    Rgba8 waveform{140, 220, 196, 255};
    Rgba8 border{74, 82, 98, 255};
};

class SoftwareRenderer final : public render::UnitRenderer {
public:
    explicit SoftwareRenderer(RasterTheme theme = {});

    void begin_frame(
        const render::FrameTree& tree,
        std::span<const render::Rect> dirty_regions,
        device::Device* device) override;
    void draw_component(const units::Component& component, const render::FrameNode& frame) override;
    void end_frame() override;

    [[nodiscard]] const Framebuffer& framebuffer() const noexcept { return framebuffer_; }
    [[nodiscard]] Framebuffer& framebuffer() noexcept { return framebuffer_; }
    [[nodiscard]] Rgba8 background_color() const noexcept { return theme_.background; }
    [[nodiscard]] std::size_t frame_count() const noexcept { return frame_count_; }
    [[nodiscard]] std::span<const render::Rect> last_dirty_regions() const noexcept {
        return std::span<const render::Rect>(last_dirty_regions_.data(), last_dirty_regions_.size());
    }

private:
    void fill_rect(render::Rect rect, render::Rect clip, Rgba8 color);
    void stroke_rect(render::Rect rect, render::Rect clip, Rgba8 color);
    void draw_text(std::string_view text, render::Rect rect, render::Rect clip, Rgba8 color);
    void draw_waveform(std::string_view samples, render::Rect rect, render::Rect clip, Rgba8 color);
    void draw_image_placeholder(const units::Component& component, const render::FrameNode& frame);

    RasterTheme theme_;
    Framebuffer framebuffer_;
    render::Rect viewport_;
    std::vector<render::Rect> last_dirty_regions_;
    std::size_t frame_count_ = 0;
};

} // namespace spiral::raster
