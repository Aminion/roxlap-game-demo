use fontdue::{Font, FontSettings};
use sdl2::{
    pixels::{Color, PixelFormatEnum},
    rect::Rect,
    render::BlendMode,
};

use crate::CanvasResources;

pub struct FontRenderer {
    font: Font,
}

impl FontRenderer {
    pub fn new() -> Self {
        let font_data = include_bytes!("../truetype/OpenSans-Regular.ttf") as &[u8];
        let font =
            Font::from_bytes(font_data, FontSettings::default()).expect("Failed to load font");
        Self { font }
    }

    /// Render `text` at pixel position `(x, y)` (top-left anchor).
    /// `size` is the font size in pixels.  Glyph textures are allocated
    /// per-call; this is fine for infrequent UI text like an FPS counter.
    pub fn draw_text(
        &self,
        cr: &mut CanvasResources,
        text: &str,
        x: i32,
        y: i32,
        size: f32,
        color: Color,
    ) {
        let line_height = (size * 1.25) as i32;
        let mut cursor_x = x;
        let mut cursor_y = y;
        for ch in text.chars() {
            if ch == '\n' {
                cursor_x = x;
                cursor_y += line_height;
                continue;
            }
            if ch == ' ' {
                cursor_x += (size * 0.3) as i32;
                continue;
            }
            let (metrics, bitmap) = self.font.rasterize(ch, size);
            if bitmap.is_empty() || metrics.width == 0 || metrics.height == 0 {
                cursor_x += metrics.advance_width as i32;
                continue;
            }

            let mut texture = match cr.texture_creator.create_texture_streaming(
                PixelFormatEnum::RGBA8888,
                metrics.width as u32,
                metrics.height as u32,
            ) {
                Ok(t) => t,
                Err(_) => continue,
            };
            texture.set_blend_mode(BlendMode::Blend);

            // RGBA8888: bytes in memory are [R, G, B, A] per pixel on SDL2.
            let _ = texture.with_lock(None, |buf: &mut [u8], pitch: usize| {
                for gy in 0..metrics.height {
                    for gx in 0..metrics.width {
                        let dst = gy * pitch + gx * 4;
                        buf[dst] = color.r;
                        buf[dst + 1] = color.g;
                        buf[dst + 2] = color.b;
                        buf[dst + 3] = bitmap[gy * metrics.width + gx];
                    }
                }
            });

            let draw_x = cursor_x + metrics.xmin;
            let draw_y = cursor_y + size as i32 - metrics.height as i32 - metrics.ymin;
            let dst = Rect::new(draw_x, draw_y, metrics.width as u32, metrics.height as u32);
            let _ = cr.canvas.copy(&texture, None, dst);

            cursor_x += metrics.advance_width as i32;
        }
    }
}
