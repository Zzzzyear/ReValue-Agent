# Re-Value Agent

> Automated Visual Creative Agent for Secondhand Product Images

A VLM-powered multi-step reasoning agent that transforms seller-taken product photos and minimal descriptions into e-commerce-ready images with professional backgrounds, creative decorations, and marketing copy.

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Environment Setup](#4-environment-setup)
5. [API Configuration](#5-api-configuration)
6. [Output Files](#6-output-files)

---

## 1. Overview

### 1.1 Problem Statement

Individual sellers on C2C platforms (e.g., Xianyu, eBay) lack professional photography and copywriting skills. Poor-quality images with cluttered backgrounds and bad lighting significantly reduce product conversion rates and pricing power.

### 1.2 Project Goal

**Input:**
- Product photo (casually shot)
- Minimal description (e.g., "Used office chair, $100")

**Output:**
- E-commerce hero image (high-quality matting + professional background + shadow blending + creative decorations + rendered text)
- Marketing copy (eye-catching title + structured description + Emoji)

### 1.3 Technology Stack

| Module | Technology | Notes |
|--------|------------|-------|
| VLM Planner | **qwen3.5-plus** | Alibaba DashScope, multimodal image input |
| LLM Copywriter | **qwen-plus** | Alibaba DashScope, text generation |
| Matting | rembg (u2net) | Local, ~176MB |
| Image Generation | Stable Diffusion 1.5 | Local, ~6GB, RTX 4060 recommended |
| Async Framework | Python asyncio | Async concurrent DAG |
| Web UI | Gradio | Interactive interface |

---

## 2. System Architecture

### 2.1 Pipeline Overview

```
                              ┌─────────────────┐
                              │   User Input     │
                              │  (Image + Text)  │
                              └────────┬────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Node 1: VLM Routing                        │
│  Input: Base64 Image + Raw Description                       │
│  Output: JSON {subject, selling_points, background_prompt,  │
│        reference_box, creative_elements, text_placement}    │
│  Model: qwen3.5-plus (Alibaba DashScope)                    │
└─────────────────────────────┬────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐   ┌─────────────────────────────┐
│   Node 2A: Vision Flow  │   │     Node 2B: Text Flow       │
│  Step 1: rembg Matting  │   │  Step 1: Assemble Prompt     │
│  Step 2: SD Generation  │   │  Step 2: Call LLM API        │
│  Step 3: Image Fusion   │   │  Step 3: Parse JSON Output    │
└────────────┬────────────┘   └──────────────┬───────────────┘
             │                              │
             └──────────────┬─────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                Node 2C: Creative Composition                 │
│  • Render decorative elements (sparkle, badge, heart, etc.) │
│  • Render text onto image (bottom-center)                   │
│  • Output decorated_image (final composite)                 │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                      Final Output
```

### 2.2 Async Parallel DAG

The system uses **asyncio** for async concurrent DAG pipeline:
- Global `asyncio.Semaphore(2)` limits concurrent request peaks
- Vision flow (Node 2A) and Text flow (Node 2B) execute in true parallel
- SD generation uses `run_in_executor()` to avoid blocking the event loop

### 2.3 Retry Mechanism

All external network requests use **tenacity** for exponential backoff retry:
- Max 3 retries
- Wait time: 2s → 4s → 8s (exponential backoff)
- Final fallback: use fallback values

---

## 3. Project Structure

```
ReValue-Agent/
├── conf/
│   └── config.yaml              # Global configuration (models, paths, timeouts)
├── core/                        # Core business logic
│   ├── __init__.py
│   ├── engine.py               # DAG scheduling engine (ReValueEngine class)
│   ├── context.py             # Context dataclass (data passing between nodes)
│   ├── prompts.py             # Prompt template management
│   ├── node_router.py         # Node 1: VLM routing (image analysis)
│   ├── node_vision.py         # Node 2A: Vision flow (matting → generation → fusion)
│   ├── node_text.py           # Node 2B: Text flow (marketing copy generation)
│   └── node_creative.py       # Node 2C: Creative composition (decorations + text)
├── tools/                      # Atomic tool modules
│   ├── __init__.py
│   ├── api_clients.py        # Unified API client (VLM/LLM) with retry mechanism
│   ├── cv_utils.py           # Image processing tools
│   │   ├── MattingTool       # rembg matting wrapper
│   │   ├── ShadowRenderer    # Shadow rendering
│   │   ├── ImageFusion       # Image fusion
│   │   └── resize_if_needed  # Image resizing helper
│   ├── sd_local.py           # Local SD inference wrapper
│   │   ├── BaseSDGenerator   # SD generator abstract base class
│   │   ├── DummySDGenerator  # Placeholder generator (no GPU fallback)
│   │   └── DiffusersSDGenerator # Real SD generator (Diffusers library)
│   ├── text_renderer.py       # Text renderer
│   │   ├── TextRenderer      # Main renderer class
│   │   └── render_title_on_image # Convenience function
│   └── decorations.py         # Decorative element rendering
│       └── DecorationRenderer # Decoration renderer (sparkle, badge, heart, etc.)
├── web/
│   ├── __init__.py
│   └── gradio_app.py         # Gradio interactive interface
├── tests/                     # Test scripts
│   ├── __init__.py
│   ├── test_01_vlm.py        # VLM module test (logs full prompt I/O)
│   ├── test_02_copy.py       # Copy module test
│   ├── test_03_vision.py     # Vision module test (matting/SD/fusion)
│   ├── test_04_creative.py   # Creative module test
│   ├── run_full_pipeline.py  # Full pipeline batch test
│   ├── run_tests.py          # Quick component test
│   ├── test_panel.py         # Gradio test panel
│   ├── eval_batch.py         # Batch evaluation script
│   └── test_vision.py        # Vision module unit test
├── data/
│   ├── original/              # Original test images
│   ├── input/                # Test input images (paired with .json descriptions)
│   ├── intermediate/         # Intermediate outputs (for testing)
│   └── output/               # Final output directory
│       └── logs/             # Module test logs
│           ├── vlm/          # VLM complete logs
│           ├── copy/         # Copy complete logs
│           ├── vision/       # Vision complete logs
│           └── creative/     # Creative complete logs
├── scripts/                   # Utility scripts
│   ├── test_emoji.py         # Emoji rendering test
│   └── test_emoji_on_real.py # Emoji rendering test on real image
├── data/assets/               # Asset files
│   └── stickers/             # Decorative sticker assets
├── main.py                    # CLI main entry point
├── requirements.txt           # Python dependencies
└── .gitignore               # Git ignore configuration
```

### 3.1 Core Modules

#### engine.py - Main Scheduling Engine

`ReValueEngine` is the "brain" of the agent, responsible for DAG scheduling and result aggregation.

```python
class ReValueEngine:
    def __init__(self, config: dict, semaphore_limit: int = 2)
    async def process(self, image: Image.Image, text: str) -> Context
```

#### context.py - Context Dataclass

The `Context` dataclass is the core data structure flowing through the pipeline:

```python
@dataclass
class Context:
    original_image: Image.Image
    original_text: str
    subject: str
    selling_points: List[str]
    background_prompt: str
    reference_box: List[float]
    creative_elements: List[dict]
    text_placement: dict
    matted_image_rgba: Image.Image
    background_image: Image.Image
    final_image: Image.Image
    final_copy: dict
    decorated_image: Image.Image
    status: TaskStatus
    error_message: str
```

#### prompts.py - Prompt Management

Manages all VLM/LLM system prompts and user prompt templates:

| Constant | Purpose |
|----------|---------|
| `VLM_SYSTEM_PROMPT` | VLM image analysis system prompt |
| `VLM_USER_PROMPT_TEMPLATE` | VLM user template |
| `COPY_SYSTEM_PROMPT_TEMPLATE` | Copy generation system prompt |
| `COPY_USER_PROMPT_TEMPLATE` | Copy generation user template |
| `FALLBACK_JSON` | Fallback output on VLM parse failure |

#### node_router.py - Node 1: VLM Routing

Calls VLM API to extract product information.

#### node_vision.py - Node 2A: Vision Flow

Chains matting, background generation, and fusion.

#### node_text.py - Node 2B: Text Flow

Calls LLM to generate marketing copy.

#### node_creative.py - Node 2C: Creative Composition

Renders decorative elements and text onto the final image.

### 3.2 Tool Modules

#### api_clients.py - API Client

Unified VLM/LLM API wrapper with built-in retry mechanism:

```python
class QwenClient(BaseAPIClient):
    async def chat(self, messages: list, **kwargs) -> str
    async def chat_with_image(self, image_base64: str, system_prompt: str, user_prompt: str) -> str
```

**API Key Loading Priority:**
1. Environment variable (production)
2. `shared/api_keys_local.yaml` (local override, at project root level)
3. `shared/api_keys.yaml` (template)

#### cv_utils.py - Image Processing

```python
class MattingTool:
    def remove_background(self, image: Image.Image) -> Image.Image  # Returns RGBA

class ShadowRenderer:
    def apply_shadow(self, subject_rgba: Image.Image, background: Image.Image) -> Image.Image

class ImageFusion:
    def fuse(self, subject_rgba: Image.Image, background: Image.Image,
             reference_box=None, shadow_renderer=None) -> Image.Image
```

#### sd_local.py - Local SD Inference

```python
class DiffusersSDGenerator:
    async def generate(self, prompt: str, negative_prompt: str = "",
                       width: int = 1024, height: int = 1024,
                       num_inference_steps: int = 25,
                       guidance_scale: float = 7.5) -> Image.Image
```

#### text_renderer.py - Text Rendering

```python
class TextRenderer:
    def render(self, image: Image.Image, text: str,
               position: str = "bottom-center",
               max_width_ratio: float = 0.85,
               font_size: int = 48,
               background_color: tuple = (0, 0, 0, 180),
               text_color: tuple = (255, 255, 255)) -> Image.Image
```

#### decorations.py - Decorative Rendering

```python
class DecorationRenderer:
    @staticmethod
    def render_elements(image: Image.Image, elements: list) -> Image.Image
```

Decorative element types: `sparkle`, `star`, `heart`, `badge`, `price_tag`, `ribbon`, `circle_decoration`

---

## 4. Environment Setup

### 4.1 Requirements

- **Python**: 3.10+
- **CUDA**: 11.8+ (for SD local inference)
- **GPU**: RTX 3060+ (RTX 4060 recommended for best experience)
- **VRAM**: 8GB+ (SD 1.5 requires ~6GB)

### 4.2 Setup Steps

```bash
# 1. Navigate to project directory
cd ReValue-Agent

# 2. Create virtual environment (if not exists)
python -m venv venv

# 3. Activate virtual environment
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download SD model (if not exists)
# Model will auto-download from HuggingFace on first run
```

### 4.3 Configure API Key

**Important**: API keys are stored in shared directory at repo root level, not committed to Git.

```bash
# 1. Edit API configuration file (create if not exists)
# Path: shared/api_keys_local.yaml

# 2. Fill in your Alibaba DashScope API Key
# Free tier: https://help.aliyun.com/zh/dashscope/
dashscope:
  api_key: "sk-xxxxxxxxxxxxxxxxxxxxxxxx"  # Your actual API Key
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### 4.4 Running Commands

```bash
# ============ Main Entry (main.py) ============

# System check
python main.py check

# Single image processing (recommended)
python main.py single -i data/input/xianyu_bag_01.jpg -t "MiniKelly second-hand bag" -o data/output/my_test

# Single image with verbose output
python main.py single -i data/input/xianyu_bag_01.jpg -t "MiniKelly second-hand bag" -o data/output/my_test -v

# Batch processing
python main.py batch --input data/input --output data/output/batch

# Start web interface
python main.py web --port 7860

# ============ Module Test Scripts (tests/) ============

# Test VLM module
python tests/test_01_vlm.py -i data/input/xianyu_bag_01.jpg -t "MiniKelly second-hand bag"

# Test copy generation module
python tests/test_02_copy.py -s "Kelly bag" -p "09 Purple|MiniKelly|95 new" -o "Second-hand MiniKelly"

# Test vision module (matting + SD + fusion)
python tests/test_03_vision.py -i data/input/xianyu_bag_01.jpg --step all

# Test creative module (decorations + text)
python tests/test_04_creative.py -i data/output/logs/vision/fused.png -t "Hermes MiniKelly"

# Run full pipeline test (batch)
python tests/run_full_pipeline.py
```

---

## 5. API Configuration

### 5.1 config.yaml

`conf/config.yaml` contains all runtime configuration:

```yaml
# VLM model configuration
vlm:
  default_model: "qwen3.5-plus"
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  timeout: 120
  max_concurrent: 2

# LLM model configuration
llm:
  default_model: "qwen-plus"
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  timeout: 30

# Local SD configuration
stable_diffusion:
  model_type: "sd_15"
  model_path: "models/.../snapshots/.../"
  num_inference_steps: 25
  guidance_scale: 7.5
  width: 1024
  height: 1024

# Matting configuration
matting:
  model_name: "u2net"

# Image fusion configuration
fusion:
  mode: "rule"
  subject_height_ratio: 0.65
  shadow_blur_radius: 15
  shadow_alpha: 100

# Creative composition configuration
creative:
  font_path: "C:/Windows/Fonts/msyh.ttc"
  default_font_size: 48
  default_text_bg: [0, 0, 0, 180]
  default_text_color: [255, 255, 255]
```

### 5.2 Switching Models

To switch VLM/LLM models, modify `default_model` in `config.yaml`:

```yaml
vlm:
  default_model: "qwen3.5-plus"  # Options: qwen_vl_plus, glm_4v, kimi_vl
llm:
  default_model: "qwen-plus"     # Options: qwen_max, glm_4, kimi
```

---

## 6. Output Files

After each processing run, `data/output/` generates:

| File Suffix | Description |
|-------------|-------------|
| `*_final.png` | Fused product image |
| `*_decorated.png` | **Final output** - with decorations and title text |
| `*_matted.png` | rembg matting result |
| `*_background.png` | SD generated background |
| `*_copy.json` | Marketing copy |
| `*_context.json` | VLM analysis result |
| `*_full_report.json` | Detailed log report |

### 6.1 Copy Output Format

`*_copy.json` example:

```json
{
  "title": "Hermes MiniKelly II | 09 Dream Purple Crocodile ✨ 95 New",
  "content": "Personal collection, less than half a year old, only worn 3 times! 09 classic dream purple American square matte crocodile skin..."
}
```

---

## Additional Documentation

- [中文版](docs/zh/README.md) - Chinese version of this documentation
