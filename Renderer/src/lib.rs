/*!
# Vello GPU Text Renderer - Parallel CPU Version

CPU-based renderer using cosmic-text for layout and rasterization.
Uses Rayon for parallel batch processing.

Features:
- cosmic-text for text layout, shaping, and rasterization
- Full CJK (Chinese, Japanese, Korean) support via Noto Sans CJK
- Binary search for optimal font size
- Rayon parallel processing (1000-2000+ img/s)
*/

use cosmic_text::{Attrs, Buffer, Color, FontSystem, Metrics, Shaping, SwashCache};
use image::{ImageBuffer, Rgb, RgbImage};
use numpy::{PyArray3, PyArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;
use std::fs;
use std::sync::Arc;

// Use Noto Sans CJK for comprehensive CJK + Latin support
const CJK_FONT_PATH: &str = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc";
const FALLBACK_FONT_DATA: &[u8] = include_bytes!("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");

// Thread-local font system and cache for parallel rendering
thread_local! {
    static FONT_SYSTEM: RefCell<Option<FontSystem>> = RefCell::new(None);
    static SWASH_CACHE: RefCell<Option<SwashCache>> = RefCell::new(None);
}

/// Initialize thread-local font system
fn init_thread_local_font_system() {
    FONT_SYSTEM.with(|fs| {
        if fs.borrow().is_none() {
            let mut font_system = FontSystem::new();

            // Try to load CJK font from file
            if let Ok(font_data) = fs::read(CJK_FONT_PATH) {
                font_system.db_mut().load_font_data(font_data);
            }

            // Load fallback font
            font_system.db_mut().load_font_data(FALLBACK_FONT_DATA.to_vec());

            *fs.borrow_mut() = Some(font_system);
        }
    });

    SWASH_CACHE.with(|sc| {
        if sc.borrow().is_none() {
            *sc.borrow_mut() = Some(SwashCache::new());
        }
    });
}

/// Render configuration (immutable, shareable across threads)
#[derive(Clone)]
struct RenderConfig {
    width: u32,
    height: u32,
    padding: u32,
    min_font_size: f32,
    max_font_size: f32,
}

/// Binary search for optimal font size
fn find_optimal_font_size(
    text: &str,
    config: &RenderConfig,
) -> f32 {
    let usable_width = (config.width - 2 * config.padding) as f32;
    let usable_height = (config.height - 2 * config.padding) as f32;

    let mut low = config.min_font_size;
    let mut high = config.max_font_size;
    let mut optimal = low;

    FONT_SYSTEM.with(|fs| {
        let mut fs_ref = fs.borrow_mut();
        let font_system = fs_ref.as_mut().unwrap();

        // Phase 1: Fast binary search
        while high - low > 0.5 {
            let mid = (low + high) / 2.0;

            let mut buffer = Buffer::new(
                font_system,
                Metrics::new(mid, mid * 1.2),
            );

            buffer.set_size(font_system, Some(usable_width), Some(usable_height));
            buffer.set_text(
                font_system,
                text,
                &Attrs::new(),
                Shaping::Advanced,
                None,
            );
            buffer.shape_until_scroll(font_system, false);

            let layout_height = buffer.layout_runs().count() as f32 * mid * 1.2;

            if layout_height <= usable_height {
                optimal = mid;
                low = mid;
            } else {
                high = mid;
            }
        }

        // Phase 2: Verification with actual layout
        let mut verified_font_size = optimal;
        for _ in 0..50 {
            let mut buffer = Buffer::new(
                font_system,
                Metrics::new(verified_font_size, verified_font_size * 1.2),
            );
            buffer.set_size(font_system, Some(usable_width), None);
            buffer.set_text(
                font_system,
                text,
                &Attrs::new(),
                Shaping::Advanced,
                None,
            );
            buffer.shape_until_scroll(font_system, false);

            let mut max_y = 0.0f32;
            for run in buffer.layout_runs() {
                let line_bottom = run.line_y + verified_font_size * 1.2;
                max_y = max_y.max(line_bottom);
            }

            if max_y <= usable_height {
                break;
            }

            verified_font_size -= 0.5;
            if verified_font_size < config.min_font_size {
                verified_font_size = config.min_font_size;
                break;
            }
        }

        optimal = verified_font_size;
    });

    optimal
}

/// Preprocess text for compact rendering (collapse multiple newlines)
fn preprocess_text(text: &str) -> String {
    // Replace multiple consecutive newlines with single newline
    let mut result = String::with_capacity(text.len());
    let mut prev_was_newline = false;

    for ch in text.chars() {
        if ch == '\n' {
            if !prev_was_newline {
                result.push(' ');  // Replace first newline with space for flow
            }
            // Skip additional newlines
            prev_was_newline = true;
        } else if ch == '\r' {
            // Skip carriage returns
            continue;
        } else {
            prev_was_newline = false;
            result.push(ch);
        }
    }

    result
}

/// Render single text to raw RGB bytes
fn render_single_to_bytes(text: &str, config: &RenderConfig) -> Vec<u8> {
    init_thread_local_font_system();

    // Preprocess text for compact rendering
    let text = preprocess_text(text);
    let text = text.as_str();

    let font_size = find_optimal_font_size(text, config);
    let usable_width = (config.width - 2 * config.padding) as f32;

    FONT_SYSTEM.with(|fs| {
        SWASH_CACHE.with(|sc| {
            let mut fs_ref = fs.borrow_mut();
            let mut sc_ref = sc.borrow_mut();
            let font_system = fs_ref.as_mut().unwrap();
            let swash_cache = sc_ref.as_mut().unwrap();

            // Create buffer
            let mut buffer = Buffer::new(
                font_system,
                Metrics::new(font_size, font_size * 1.2),
            );
            buffer.set_size(font_system, Some(usable_width), None);
            buffer.set_text(
                font_system,
                text,
                &Attrs::new(),
                Shaping::Advanced,
                None,
            );
            buffer.shape_until_scroll(font_system, false);

            // Create white image
            let mut img: RgbImage = ImageBuffer::from_pixel(
                config.width,
                config.height,
                Rgb([255, 255, 255]),
            );

            // Render glyphs
            buffer.draw(font_system, swash_cache, Color::rgb(0, 0, 0), |x, y, w, h, color| {
                let img_x = x + config.padding as i32;
                let img_y = y + config.padding as i32;

                for row in 0..h {
                    for col in 0..w {
                        let px = img_x + col as i32;
                        let py = img_y + row as i32;

                        if px >= 0 && px < config.width as i32 && py >= 0 && py < config.height as i32 {
                            let pixel = img.get_pixel_mut(px as u32, py as u32);
                            let alpha = color.a() as f32 / 255.0;
                            pixel[0] = (255.0 * (1.0 - alpha)) as u8;
                            pixel[1] = (255.0 * (1.0 - alpha)) as u8;
                            pixel[2] = (255.0 * (1.0 - alpha)) as u8;
                        }
                    }
                }
            });

            img.into_raw()
        })
    })
}

/// GPU-ready text renderer with parallel processing
#[pyclass]
struct VelloRenderer {
    config: Arc<RenderConfig>,
}

#[pymethods]
impl VelloRenderer {
    /// Create new renderer
    #[new]
    #[pyo3(signature = (width=640, height=640, padding=20, min_font_size=5.0, max_font_size=20.0))]
    fn new(
        width: u32,
        height: u32,
        padding: u32,
        min_font_size: f32,
        max_font_size: f32,
    ) -> PyResult<Self> {
        // Initialize thread-local font system for main thread
        init_thread_local_font_system();

        // Set rayon thread pool size based on available CPUs
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get().min(64))  // Cap at 64 threads
            .unwrap_or(16);

        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .ok();  // Ignore if already built

        Ok(Self {
            config: Arc::new(RenderConfig {
                width,
                height,
                padding,
                min_font_size,
                max_font_size,
            }),
        })
    }

    /// Render batch of texts to numpy arrays (parallel)
    fn render_batch<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
        let config = self.config.clone();

        // Release GIL and render in parallel
        let raw_images: Vec<Vec<u8>> = py.allow_threads(|| {
            texts.par_iter()
                .map(|text| render_single_to_bytes(text, &config))
                .collect()
        });

        // Convert to numpy arrays (must be done with GIL held)
        let mut results = Vec::with_capacity(raw_images.len());
        for img_data in raw_images {
            let array_1d = numpy::PyArray::from_vec(py, img_data);
            let array = array_1d.reshape([
                config.height as usize,
                config.width as usize,
                3,
            ])?;
            results.push(array);
        }

        Ok(results)
    }

    /// Get version
    #[getter]
    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    /// Get number of threads
    #[getter]
    fn num_threads(&self) -> usize {
        rayon::current_num_threads()
    }
}

/// Python module definition
#[pymodule]
fn vello_renderer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VelloRenderer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
