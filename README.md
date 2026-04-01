# Re-Value Agent

> 二手商品图文创意自动化视觉 Agent

基于视觉大语言模型（VLM）的多步推理 Agent，输入卖家随手拍摄的商品原图及极简描述，自动完成"商品特征提取、高质量抠图、场景重绘、文案生成及图文合成"的全链路创意优化。

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [目录结构与代码说明](#3-目录结构与代码说明)
4. [环境配置与运行](#4-环境配置与运行)
5. [API 配置](#5-api-配置)
6. [测试与调试](#6-测试与调试)
7. [输出文件说明](#7-输出文件说明)
8. [团队分工与下一步工作](#8-团队分工与下一步工作)
9. [常见问题](#9-常见问题)

---

## 1. 项目概述

### 1.1 业务背景

C2C 交易平台（如闲鱼、eBay）的个人卖家通常缺乏专业的拍摄和文案能力。背景杂乱、光线昏暗的实拍图严重降低了商品的转化率和溢价空间。

### 1.2 项目目标

**输入**：
- 商品原图（随手拍摄）
- 极简描述（如"九成新办公椅，100元"）

**输出**：
- 电商级主图（高质量抠图 + 专业背景 + 阴影融合 + 创意装饰元素 + 文字渲染）
- 营销文案（吸睛标题 + 结构化描述 + Emoji）

### 1.3 技术选型

| 模块 | 技术 | 说明 |
|------|------|------|
| VLM 规划器 | **qwen3.5-plus** | 阿里百炼，多模态支持图片输入 |
| LLM 文案 | **qwen-plus** | 阿里百炼，文本生成 |
| 抠图 | rembg (u2net) | 本地运行，约 176MB |
| 生图 | Stable Diffusion 1.5 | 本地运行，约 6GB，RTX 4060 可流畅运行 |
| 异步框架 | Python asyncio | 异步并发 DAG |
| Web UI | Gradio | 交互界面 |

---

## 2. 系统架构

### 2.1 整体架构图

```
                              ┌─────────────────┐
                              │   用户输入       │
                              │  (图片 + 简述)   │
                              └────────┬────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────┐
│                     Node 1: VLM 联合感知                      │
│  输入: Base64图片 + 原始描述                                   │
│  输出: JSON {subject, selling_points, background_prompt,      │
│        reference_box, creative_elements, text_placement}    │
│  模型: qwen3.5-plus (阿里百炼)                                │
└─────────────────────────────┬────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐   ┌─────────────────────────────┐
│   Node 2A: 视觉流        │   │     Node 2B: 文案流        │
│  Step 1: rembg 抠图     │   │  Step 1: 组装文案 Prompt   │
│  Step 2: SD 生图        │   │  Step 2: 调用 LLM API      │
│  Step 3: 图像融合+阴影   │   │  Step 3: 解析 JSON 输出   │
└────────────┬────────────┘   └──────────────┬───────────────┘
             │                              │
             └──────────────┬─────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                Node 2C: 创意合成                               │
│  • 渲染装饰元素 (sparkle, badge, heart等)                     │
│  • 渲染文字到图像 (底部居中)                                  │
│  • 输出 decorated_image (带文字和装饰的最终图)                  │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                      最终输出
```

### 2.2 异步并行 DAG

系统采用 **asyncio** 实现异步并发 DAG 管线：
- 全局 `asyncio.Semaphore(2)` 限制并发请求峰值
- 视觉流（Node 2A）与文本流（Node 2B）**真正并行**执行
- SD 生图使用 `run_in_executor()` 避免阻塞事件循环

### 2.3 重试机制

所有外部网络请求均使用 **tenacity** 库实现指数退避重试：
- 最大重试 3 次
- 等待时间：2s → 4s → 8s（指数退避）
- 最终兜底：使用 fallback 值

---

## 3. 目录结构与代码说明

```
ReValue-Agent/
├── conf/
│   └── config.yaml              # 全局配置文件（模型参数、路径、超时等）
├── core/                        # 核心业务逻辑
│   ├── __init__.py
│   ├── engine.py               # DAG 调度引擎（ReValueEngine 类）
│   ├── context.py             # Context 数据类（在节点间传递的上下文）
│   ├── prompts.py             # Prompt 模板管理（VLM/LLM prompts）
│   ├── node_router.py         # Node 1: VLM 路由（图像分析）
│   ├── node_vision.py         # Node 2A: 视觉流（抠图→生图→融合）
│   ├── node_text.py           # Node 2B: 文案流（营销文案生成）
│   └── node_creative.py       # Node 2C: 创意合成（装饰+文字渲染）
├── tools/                      # 原子工具模块
│   ├── __init__.py
│   ├── api_clients.py        # 统一 API 客户端（VLM/LLM），内置重试机制
│   ├── cv_utils.py           # 图像处理工具
│   │   ├── MattingTool       # rembg 抠图封装
│   │   ├── ShadowRenderer    # 阴影渲染
│   │   ├── ImageFusion       # 图像融合
│   │   └── resize_if_needed  # 图像缩放辅助
│   ├── sd_local.py           # 本地 SD 推理封装
│   │   ├── BaseSDGenerator   # SD 生成器抽象基类
│   │   ├── DummySDGenerator  # 占位符生成器（无 GPU 时使用）
│   │   └── DiffusersSDGenerator # 真实 SD 生成器（Diffusers 库）
│   ├── text_renderer.py       # 文字渲染器
│   │   ├── TextRenderer      # 主渲染器类
│   │   └── render_title_on_image # 便捷函数
│   └── decorations.py         # 装饰元素绘制
│       └── DecorationRenderer # 装饰渲染器（sparkle, badge, heart 等）
├── web/
│   ├── __init__.py
│   └── gradio_app.py         # Gradio 交互界面
├── tests/                     # 测试脚本
│   ├── __init__.py
│   ├── test_01_vlm.py        # VLM 模块测试（记录完整 prompt 输入输出）
│   ├── test_02_copy.py       # Copy 文案模块测试
│   ├── test_03_vision.py     # Vision 视觉模块测试（抠图/SD/融合）
│   ├── test_04_creative.py   # Creative 创意模块测试
│   ├── run_full_pipeline.py  # 完整流程批量测试
│   ├── run_tests.py          # 组件快速测试
│   ├── test_panel.py         # Gradio 测试面板
│   ├── eval_batch.py         # 批量评估脚本
│   └── test_vision.py        # 视觉模块单元测试
├── data/
│   ├── original/              # 原始测试图片
│   ├── input/                # 测试输入图片（与 .json 描述配对）
│   ├── intermediate/         # 中间产物（测试用）
│   └── output/               # 最终输出目录
│       └── logs/             # 模块测试日志
│           ├── vlm/          # VLM 完整日志
│           ├── copy/         # Copy 完整日志
│           ├── vision/       # Vision 完整日志
│           └── creative/     # Creative 完整日志
├── main.py                   # CLI 主入口（正式运行用）
├── requirements.txt          # Python 依赖
└── .gitignore               # Git 忽略配置
```

### 3.1 core/ 核心模块详解

#### engine.py - 主调度引擎

`ReValueEngine` 是整个 Agent 的"大脑"，负责 DAG 调度和结果汇聚。

```python
class ReValueEngine:
    def __init__(self, config: dict, semaphore_limit: int = 2)
    async def process(self, image: Image.Image, text: str) -> Context
    async def _process_vision_branch(ctx: Context) -> Context  # Node 2A
    async def _process_text_branch(ctx: Context) -> Context    # Node 2B
```

**Context 数据类**是贯穿整个管线的核心数据结构：

```python
@dataclass
class Context:
    # 输入
    original_image: Image.Image
    original_text: str

    # Node 1 输出 (VLM)
    subject: str                    # 商品品类（如"凯莉包"）
    selling_points: List[str]       # 卖点列表
    background_prompt: str           # SD 背景描述
    reference_box: List[float]      # 商品边界框 [x_min, y_min, x_max, y_max]
    creative_elements: List[dict]    # 装饰元素配置
    text_placement: dict            # 文字放置配置

    # Node 2A 输出 (Vision)
    matted_image_rgba: Image.Image  # 抠图结果
    background_image: Image.Image     # SD 背景图
    final_image: Image.Image         # 融合结果

    # Node 2B 输出 (Text)
    final_copy: dict                # {"title": "...", "content": "..."}

    # Node 2C 输出 (Creative)
    decorated_image: Image.Image     # 最终合成图

    # 状态
    status: TaskStatus
    error_message: str
```

#### prompts.py - Prompt 管理

管理所有 VLM/LLM 的 System Prompt 和 User Prompt 模板：

| 常量 | 用途 |
|------|------|
| `VLM_SYSTEM_PROMPT` | VLM 分析图片时的系统提示 |
| `VLM_USER_PROMPT_TEMPLATE` | VLM 用户模板 |
| `COPY_SYSTEM_PROMPT_TEMPLATE` | 文案生成系统提示（支持闲鱼体/小红书体）|
| `COPY_USER_PROMPT_TEMPLATE` | 文案生成用户模板 |
| `FALLBACK_JSON` | VLM 解析失败时的兜底输出 |

关键函数：
- `get_vlm_prompts(original_text)` → 返回 VLM 的 system + user prompt
- `get_copy_prompts(subject, selling_points, original_text, platform_style)` → 返回文案 prompt
- `parse_json_from_response(response)` → 从 VLM 响应中提取 JSON

#### node_router.py - Node 1: VLM 路由

调用 VLM API 提取商品信息：

```python
class NodeRouter:
    async def process(self, ctx: Context) -> Context:
        # 1. 获取 prompt
        prompts = get_vlm_prompts(ctx.original_text)
        # 2. 图片转 base64
        image_base64 = ReValueEngine.image_to_base64(ctx.original_image)
        # 3. 调用 VLM
        response = await self._call_vlm(image_base64, prompts["system"], prompts["user"])
        # 4. 解析 JSON
        vlm_result = parse_json_from_response(response)
        # 5. 更新 ctx
        ctx.vlm_json_output = vlm_result
        ctx.subject = vlm_result.get("subject", "商品")
        # ... 其他字段
```

#### node_vision.py - Node 2A: 视觉流

串联抠图、生图、融合三大步骤：

```python
class NodeVision:
    async def process(self, ctx: Context) -> Context:
        # Step 1: 抠图 (rembg u2net)
        ctx.matted_image_rgba = await self._process_matting(ctx.original_image)

        # Step 2: 背景生成 (SD 1.5)
        background_prompt = ctx.background_prompt + get_sd_prompt_suffix()
        ctx.background_image = await self._process_background(background_prompt)

        # Step 3: 图像融合
        ctx.final_image = self._process_fusion(
            subject_rgba=ctx.matted_image_rgba,
            background=ctx.background_image,
            reference_box=ctx.reference_box,
        )
```

关键工具：
- `MattingTool.remove_background(image)` - 使用 rembg 抠图
- `DiffusersSDGenerator.generate()` - SD 生图
- `ImageFusion.fuse()` - 融合商品图和背景

#### node_text.py - Node 2B: 文案流

调用 LLM 生成营销文案：

```python
class NodeText:
    async def process(self, ctx: Context) -> Context:
        prompts = get_copy_prompts(
            subject=ctx.subject,
            selling_points=ctx.selling_points,
            original_text=ctx.original_text,
            platform_style=ctx.platform_style,  # "闲鱼体" 或 "小红书体"
        )
        response = await self._call_llm(prompts["system"], prompts["user"])
        ctx.final_copy = self._parse_copy_response(response)
```

#### node_creative.py - Node 2C: 创意合成

将装饰元素和文字渲染到最终图像上：

```python
class NodeCreative:
    async def process(self, ctx: Context) -> Context:
        # 渲染装饰元素
        decorated = await self._render_decorations(ctx.final_image, ctx.creative_elements)
        # 渲染文字
        title = self._extract_title(ctx.final_copy)
        ctx.decorated_image = self._render_text(decorated, title, ctx.text_placement)
```

装饰元素类型：`sparkle`, `star`, `heart`, `badge`, `price_tag`, `ribbon`, `circle_decoration`

### 3.2 tools/ 工具模块详解

#### api_clients.py - API 客户端

统一封装 VLM/LLM API 调用，内置重试机制：

```python
class QwenClient(BaseAPIClient):
    async def chat(self, messages: list, **kwargs) -> str
    async def chat_with_image(self, image_base64: str, system_prompt: str, user_prompt: str) -> str
```

工厂函数：
- `get_vlm_client(model, config)` → 获取 VLM 客户端
- `get_llm_client(model, config)` → 获取 LLM 客户端

**API Key 加载优先级**：
1. 环境变量（生产环境）
2. `D:/Code/AI_Playground/shared/api_keys_local.yaml`（本地覆盖）
3. `D:/Code/AI_Playground/shared/api_keys.yaml`（模板）

#### cv_utils.py - 图像处理

```python
class MattingTool:
    def __init__(self, model_name: str = "u2net")
    def remove_background(self, image: Image.Image) -> Image.Image  # 返回 RGBA

class ShadowRenderer:
    def apply_shadow(self, subject_rgba: Image.Image, background: Image.Image) -> Image.Image

class ImageFusion:
    def fuse(self, subject_rgba: Image.Image, background: Image.Image,
             reference_box=None, shadow_renderer=None) -> Image.Image
```

#### sd_local.py - 本地 SD 推理

```python
class DiffusersSDGenerator:
    def __init__(self, model_path: str, model_type: str = "sd_15", device: str = "cuda")
    async def generate(self, prompt: str, negative_prompt: str = "",
                       width: int = 1024, height: int = 1024,
                       num_inference_steps: int = 25,
                       guidance_scale: float = 7.5) -> Image.Image
```

#### text_renderer.py - 文字渲染

```python
class TextRenderer:
    def render(self, image: Image.Image, text: str,
               position: str = "bottom-center",
               max_width_ratio: float = 0.85,
               font_size: int = 48,
               background_color: tuple = (0, 0, 0, 180),
               text_color: tuple = (255, 255, 255)) -> Image.Image
```

#### decorations.py - 装饰绘制

```python
class DecorationRenderer:
    @staticmethod
    def render_elements(image: Image.Image, elements: list) -> Image.Image
```

装饰元素格式：
```python
{
    "type": "sparkle",  # 或 badge, heart, price_tag 等
    "position": "top-right",  # top-left, top-right, bottom-left, bottom-right, center
    "count": 3,  # 数量（sparkle/heart/star 有效）
    "size": "medium",  # small, medium, large
    "text": "SALE",  # 仅 badge/price_tag 有效
    "style": "rounded_rect"  # 仅 badge 有效
}
```

---

## 4. 环境配置与运行

### 4.1 环境要求

- **Python**: 3.10+
- **CUDA**: 11.8+ (用于 SD 本地推理)
- **GPU**: RTX 3060+ (建议 4060 以获得最佳体验)
- **显存**: 8GB+ (SD 1.5 需要约 6GB)

### 4.2 环境配置步骤

```bash
# 1. 进入项目目录
cd D:/Code/AI_Playground/projects/ReValue-Agent

# 2. 创建虚拟环境（如果还没有）
python -m venv venv

# 3. 激活虚拟环境
venv\Scripts\activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 下载 SD 模型（如果还没有）
# 模型会自动从 HuggingFace 下载，首次运行时会提示
```

### 4.3 配置 API Key

**重要**：API Key 存储在共享目录 `D:/Code/AI_Playground/shared/`，不会提交到 GitHub。

```bash
# 1. 编辑 API 配置文件
notepad D:/Code/AI_Playground/shared/api_keys_local.yaml

# 2. 填入你的阿里云百炼 API Key
# DashScope 免费额度: https://help.aliyun.com/zh/dashscope/
dashscope:
  api_key: "sk-xxxxxxxxxxxxxxxxxxxxxxxx"  # 填入你的真实 API Key
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### 4.4 运行命令

```bash
# ============ 正式运行入口 (main.py) ============

# 系统检查
python main.py check

# 单图处理（正式推荐）
python main.py single -i data/input/xianyu_bag_01.jpg -t "MiniKelly二手包" -o data/output/my_test

# 单图处理（详细输出）
python main.py single -i data/input/xianyu_bag_01.jpg -t "MiniKelly二手包" -o data/output/my_test -v

# 批量处理
python main.py batch --input data/input --output data/output/batch

# 启动 Web 界面
python main.py web --port 7860

# ============ 模块测试脚本 (tests/) ============

# 测试 VLM 模块
python tests/test_01_vlm.py -i data/input/xianyu_bag_01.jpg -t "MiniKelly二手包"

# 测试文案生成模块
python tests/test_02_copy.py -s "凯莉包" -p "09紫|MiniKelly|95新" -o "二手MiniKelly"

# 测试视觉模块（抠图+SD+融合）
python tests/test_03_vision.py -i data/input/xianyu_bag_01.jpg --step all

# 测试创意模块（装饰+文字）
python tests/test_04_creative.py -i data/output/logs/vision/fused.png -t "爱马仕MiniKelly"

# 运行完整流程测试（批量）
python tests/run_full_pipeline.py
```

---

## 5. API 配置

### 5.1 config.yaml 配置文件

`conf/config.yaml` 包含所有运行时配置：

```yaml
# VLM 模型配置
vlm:
  default_model: "qwen3.5-plus"           # 模型名称
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  timeout: 120                            # 超时时间（秒）
  max_concurrent: 2

# 文案生成模型配置
llm:
  default_model: "qwen-plus"
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  timeout: 30

# 本地SD配置
stable_diffusion:
  model_type: "sd_15"                    # sd_15 | sdxl_turbo | sdxl
  model_path: "models/.../snapshots/.../" # 本地模型路径
  num_inference_steps: 25                # 推理步数
  guidance_scale: 7.5
  width: 1024
  height: 1024

# 抠图配置
matting:
  model_name: "u2net"                    # u2net | u2netp | u2net_human_seg

# 图像融合配置
fusion:
  mode: "rule"                          # rule | vlm
  subject_height_ratio: 0.65            # 商品占背景高度比例
  shadow_blur_radius: 15                # 阴影模糊半径
  shadow_alpha: 100                    # 阴影透明度

# 创意合成配置
creative:
  font_path: "C:/Windows/Fonts/msyh.ttc"  # 字体路径
  default_font_size: 48
  default_text_bg: [0, 0, 0, 180]       # 半透明黑色背景
  default_text_color: [255, 255, 255]   # 白色文字
```

### 5.2 切换模型

如需切换 VLM/LLM 模型，修改 `config.yaml` 中的 `default_model`：

```yaml
vlm:
  default_model: "qwen3.5-plus"  # 可选: qwen_vl_plus, glm_4v, kimi_vl
llm:
  default_model: "qwen-plus"     # 可选: qwen_max, glm_4, kimi
```

---

## 6. 测试与调试

### 6.1 模块测试脚本

每个核心模块都有独立的测试脚本，记录完整输入输出：

| 脚本 | 测试内容 | 日志输出位置 |
|------|---------|-------------|
| `test_01_vlm.py` | VLM 图像分析 | `data/output/logs/vlm/` |
| `test_02_copy.py` | 文案生成 | `data/output/logs/copy/` |
| `test_03_vision.py` | 视觉处理 | `data/output/logs/vision/` |
| `test_04_creative.py` | 创意渲染 | `data/output/logs/creative/` |

### 6.2 日志格式

每个模块测试输出的 JSON 包含：

```json
{
  "test_info": {
    "module": "vlm",
    "timestamp": "2026-03-31 16:00:00",
    "image_path": "data/input/xianyu_bag_01.jpg",
    "text": "MiniKelly二手包"
  },
  "prompt": {
    "system": "完整 system prompt...",
    "user": "完整 user prompt..."
  },
  "output": {
    "raw_response": "API 原始响应",
    "parsed": { ... }
  },
  "status": "success"
}
```

### 6.3 查看日志

```bash
# 查看 VLM 测试日志
ls data/output/logs/vlm/

# 查看某次测试的完整输入输出
cat data/output/logs/vlm/20260331_xxxxx_vlm_log.json
```

---

## 7. 输出文件说明

每次处理后，`data/output/` 会生成以下文件：

| 文件后缀 | 说明 |
|----------|------|
| `*_final.png` | 融合后的商品图 |
| `*_decorated.png` | **最终输出** - 带装饰和文案标题 |
| `*_matted.png` | rembg 抠图结果 |
| `*_background.png` | SD 生成的背景 |
| `*_copy.json` | 营销文案 |
| `*_context.json` | VLM 分析结果 |
| `*_full_report.json` | 详细日志报告 |

### 7.1 文案输出格式

`*_copy.json` 内容示例：

```json
{
  "title": "爱马仕MiniKelly二代｜09梦幻紫鳄鱼皮✨95新",
  "content": "自用收藏款～入手不到半年，只背过3次！09年经典梦幻紫美洲方块雾面鳄鱼皮..."
}
```

---

## 8. 团队分工与下一步工作

### 8.1 团队分工

| 成员 | 职责 | 主要文件 |
|------|------|----------|
| 赵钎年 | Agent 架构、core/、Gradio UI | core/*.py, web/gradio_app.py |
| 李冠良 | 视觉处理、tools/ | tools/*.py, core/node_vision.py, core/node_creative.py |
| 马在飞 | 数据、测试、Prompt 优化 | tests/*.py, core/node_text.py, core/prompts.py |

### 8.2 下一步工作 (TECH_PLAN.md)

详细内容请参考 [TECH_PLAN.md](TECH_PLAN.md)。

#### 李冠良负责（视觉优化）

1. **P0 文字方框问题**：修复字体不支持 Emoji 的问题
   - 文件：`tools/text_renderer.py`
   - 方案：过滤非支持字符或引入复合字体

2. **P0 文字重叠问题**：修复 `_wrap_text` 行高计算 Bug
   - 文件：`tools/text_renderer.py`

3. **P1 场景-品类硬路由**：建立 `SCENE_TEMPLATES` 字典
   - 文件：`core/node_vision.py`
   - 防止足球鞋背景出现厨房等不匹配问题

4. **P2 光影一致性**：ShadowRenderer 根据光源方向计算阴影偏移
   - 文件：`tools/cv_utils.py`

5. **P2 PNG 素材库**：用高质量贴图替代 PIL 手绘
   - 文件：`tools/decorations.py`
   - 新建：`data/assets/` 目录

6. **P2 文字排版海报化**：阿里巴巴普惠体 + 渐变底板 + 描边
   - 文件：`tools/text_renderer.py`

#### 马在飞负责（Prompt 优化与测试）

1. **P1 VLM Prompt 场景约束**：在 `core/prompts.py` 中增加
   - 相机视角约束（俯拍/平视）
   - 物理承托面约束（草地/桌面/地面）

2. **测试验证**：使用 `tests/test_01_vlm.py` 等脚本验证 Prompt 优化效果

3. **样本测试**：测试 10-15 组样本，覆盖多种品类

### 8.3 使用测试脚本的完整流程

```bash
# 1. 先单独测试每个模块
python tests/test_01_vlm.py -i data/input/xianyu_bag_01.jpg -t "MiniKelly二手包"
python tests/test_02_copy.py -s "凯莉包" -p "09紫|MiniKelly|95新" -o "二手MiniKelly"
python tests/test_03_vision.py -i data/input/xianyu_bag_01.jpg --step all
python tests/test_04_creative.py -i {fused_path}.png -t "爱马仕MiniKelly"

# 2. 检查日志输出
cat data/output/logs/vlm/xxx_vlm_log.json
cat data/output/logs/copy/xxx_copy_log.json

# 3. 如果模块都正常，用 main.py 跑完整流程
python main.py single -i data/input/xianyu_bag_01.jpg -t "MiniKelly二手包" -o data/output/final_test -v
```

---

## 9. 常见问题

### Q: 运行时显示 "ModuleNotFoundError"

```bash
# 解决方案：安装依赖
venv\Scripts\activate
pip install -r requirements.txt
```

### Q: API 调用报 401 错误

```bash
# 解决方案：检查 API Key 配置
# 1. 确认 api_keys_local.yaml 中填入了真实 API Key
# 2. 确认 API Key 有免费额度
notepad D:/Code/AI_Playground/shared/api_keys_local.yaml
```

### Q: SD 生成显示 "CUDA out of memory"

```yaml
# 解决方案：降低推理步数或图像尺寸
stable_diffusion:
  num_inference_steps: 15  # 从 25 降到 15
  width: 768               # 从 1024 降到 768
  height: 768
```

### Q: 抠图边缘有残留背景

```yaml
# 解决方案：调整 alpha threshold
matting:
  alpha_threshold: 10  # 提高阈值
```

### Q: 如何查看完整日志？

每个输出目录都有 `*_full_report.json`，包含端到端的完整处理记录。

### Q: 如何添加新的测试样本？

```bash
# 1. 在 data/input/ 目录下添加图片
# 2. 创建同名的 .json 描述文件
# 例如：data/input/xianyu_bag_01.jpg 需要配对 data/input/xianyu_bag_01.json
```

`xianyu_bag_01.json` 格式：
```json
{
  "description": "MiniKelly二代 09梦幻紫...",
  "platform": "闲鱼"
}
```

---

## 附录：测试样本

| 图片 | 商品 | 平台 |
|------|------|------|
| xianyu_bag_01.jpg | 爱马仕 Kelly 包 | 闲鱼 |
| xianyu_shoes_01.jpg | 李宁球鞋 | 闲鱼 |
| shoes_02.jpg | Nike 足球鞋 | 闲鱼 |
| xiaohongshu_bag_01.jpg | LV 月亮包 | 小红书 |

---

## 更新日志

### v4.1 (2026-04-01)
- 🐛 修复 Gradio 前端 `TaskStatus` 引用错误 (`self.engine.TaskStatus` → `TaskStatus`)
- 🐛 修复 aiohttp `ClientSession` 未关闭导致的资源泄漏警告
- 添加 `ReValueEngine.close()` 方法确保资源正确释放

### v4.0 (2026-03-31)
- 完整 DAG 管线实现（VLM → 视觉流/文案流 → 创意合成）
- Gradio Web UI 正式可用
- 支持闲鱼/小红书双平台文案风格
- rembg 抠图 + SD 1.5 背景生成
- 装饰元素（闪光/徽章/价格标签）+ 文字渲染
- 批量测试与评估脚本

---

*文档版本: v4.1*
*最后更新: 2026-04-01*
*负责人: 赵钎年（架构）、李冠良（视觉）、马在飞（测试）*
