# Re-Value Agent

> 二手商品图文创意自动化视觉 Agent

基于视觉大语言模型（VLM）的多步推理 Agent，输入卖家随手拍摄的商品原图及极简描述，自动完成"商品特征提取、高质量抠图、场景重绘、文案生成及图文合成"的全链路创意优化。

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [目录结构](#3-目录结构)
4. [环境配置与运行](#4-环境配置与运行)
5. [API 配置](#5-api-配置)
6. [输出文件说明](#6-输出文件说明)

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

### 2.1 管线概览

```
                              ┌─────────────────┐
                              │   用户输入       │
                              │  (图片 + 简述)   │
                              └────────┬────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────┐
│                     Node 1: VLM 路由                         │
│  输入: Base64图片 + 原始描述                                   │
│  输出: JSON {subject, selling_points, background_prompt,    │
│        reference_box, creative_elements, text_placement}    │
│  模型: qwen3.5-plus (阿里百炼)                                │
└─────────────────────────────┬────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐   ┌─────────────────────────────┐
│   Node 2A: 视觉流        │   │     Node 2B: 文案流        │
│  Step 1: rembg 抠图     │   │  Step 1: 组装 Prompt       │
│  Step 2: SD 生图        │   │  Step 2: 调用 LLM API      │
│  Step 3: 图像融合       │   │  Step 3: 解析 JSON 输出   │
└────────────┬────────────┘   └──────────────┬───────────────┘
             │                              │
             └──────────────┬─────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                Node 2C: 创意合成                              │
│  • 渲染装饰元素 (sparkle, badge, heart等)                    │
│  • 渲染文字到图像 (底部居中)                                  │
│  • 输出 decorated_image (最终合成图)                          │
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

## 3. 目录结构

```
ReValue-Agent/
├── conf/
│   └── config.yaml              # 全局配置文件（模型参数、路径、超时等）
├── core/                        # 核心业务逻辑
│   ├── __init__.py
│   ├── engine.py               # DAG 调度引擎（ReValueEngine 类）
│   ├── context.py              # Context 数据类（在节点间传递的上下文）
│   ├── prompts.py              # Prompt 模板管理（VLM/LLM prompts）
│   ├── node_router.py          # Node 1: VLM 路由（图像分析）
│   ├── node_vision.py          # Node 2A: 视觉流（抠图→生图→融合）
│   ├── node_text.py           # Node 2B: 文案流（营销文案生成）
│   └── node_creative.py       # Node 2C: 创意合成（装饰+文字渲染）
├── tools/                      # 原子工具模块
│   ├── __init__.py
│   ├── api_clients.py         # 统一 API 客户端（VLM/LLM），内置重试机制
│   ├── cv_utils.py            # 图像处理工具
│   │   ├── MattingTool        # rembg 抠图封装
│   │   ├── ShadowRenderer     # 阴影渲染
│   │   ├── ImageFusion        # 图像融合
│   │   └── resize_if_needed   # 图像缩放辅助
│   ├── sd_local.py            # 本地 SD 推理封装
│   │   ├── BaseSDGenerator    # SD 生成器抽象基类
│   │   ├── DummySDGenerator   # 占位符生成器（无 GPU 时使用）
│   │   └── DiffusersSDGenerator # 真实 SD 生成器
│   ├── text_renderer.py       # 文字渲染器
│   │   ├── TextRenderer       # 主渲染器类
│   │   └── render_title_on_image # 便捷函数
│   └── decorations.py         # 装饰元素绘制
│       └── DecorationRenderer # 装饰渲染器（sparkle, badge, heart 等）
├── web/
│   ├── __init__.py
│   └── gradio_app.py          # Gradio 交互界面
├── tests/                     # 测试脚本
│   ├── test_01_vlm.py         # VLM 模块测试
│   ├── test_02_copy.py       # 文案模块测试
│   ├── test_03_vision.py     # 视觉模块测试
│   ├── test_04_creative.py  # 创意模块测试
│   ├── run_full_pipeline.py  # 完整流程批量测试
│   ├── run_tests.py          # 组件快速测试
│   ├── test_panel.py         # Gradio 测试面板
│   ├── eval_batch.py         # 批量评估脚本
│   └── test_vision.py         # 视觉模块单元测试
├── data/
│   ├── original/              # 原始测试图片
│   ├── input/                # 测试输入图片
│   ├── intermediate/         # 中间产物
│   └── output/               # 最终输出目录
│       └── logs/             # 模块测试日志
├── scripts/                   # 工具脚本
├── data/assets/              # 资源文件
│   └── stickers/            # 装饰贴纸资源
├── main.py                    # CLI 主入口
├── requirements.txt          # Python 依赖
└── .gitignore               # Git 忽略配置
```

### 3.1 核心模块

#### engine.py - 主调度引擎

`ReValueEngine` 是整个 Agent 的"大脑"，负责 DAG 调度和结果汇聚。

```python
class ReValueEngine:
    def __init__(self, config: dict, semaphore_limit: int = 2)
    async def process(self, image: Image.Image, text: str) -> Context
```

#### context.py - Context 数据类

`Context` 是贯穿整个管线的核心数据结构，包含输入、各节点输出和状态信息。

#### prompts.py - Prompt 管理

管理所有 VLM/LLM 的 System Prompt 和 User Prompt 模板。

#### node_router.py - Node 1: VLM 路由

调用 VLM API 提取商品信息。

#### node_vision.py - Node 2A: 视觉流

串联抠图、生图、融合三大步骤。

#### node_text.py - Node 2B: 文案流

调用 LLM 生成营销文案。

#### node_creative.py - Node 2C: 创意合成

将装饰元素和文字渲染到最终图像上。

### 3.2 工具模块

#### api_clients.py - API 客户端

统一封装 VLM/LLM API 调用，内置重试机制。

**API Key 加载优先级**：
1. 环境变量（生产环境）
2. `shared/api_keys_local.yaml`（本地覆盖，项目根目录层级）
3. `shared/api_keys.yaml`（模板）

#### cv_utils.py - 图像处理

`MattingTool.remove_background()` - rembg 抠图
`ShadowRenderer.apply_shadow()` - 阴影渲染
`ImageFusion.fuse()` - 图像融合

#### sd_local.py - 本地 SD 推理

`DiffusersSDGenerator.generate()` - Stable Diffusion 生图

#### text_renderer.py - 文字渲染

`TextRenderer.render()` - 文字渲染到图像

#### decorations.py - 装饰绘制

`DecorationRenderer.render_elements()` - 渲染装饰元素

装饰元素类型：`sparkle`, `star`, `heart`, `badge`, `price_tag`, `ribbon`, `circle_decoration`

---

## 4. 环境配置与运行

### 4.1 环境要求

- **Python**: 3.10+
- **CUDA**: 11.8+ (用于 SD 本地推理)
- **GPU**: RTX 3060+ (建议 RTX 4060)
- **显存**: 8GB+ (SD 1.5 需要约 6GB)

### 4.2 配置步骤

```bash
# 1. 进入项目目录
cd ReValue-Agent

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
venv\Scripts\activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 下载 SD 模型（首次运行会自动下载）
```

### 4.3 配置 API Key

API Key 存储在项目根目录的 `shared/` 目录下，不会提交到 Git。

```bash
# 编辑 API 配置文件（如不存在请创建）
# 路径：shared/api_keys_local.yaml

# 填入阿里云百炼 API Key
# 免费额度: https://help.aliyun.com/zh/dashscope/
dashscope:
  api_key: "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### 4.4 运行命令

```bash
# ============ 主入口 (main.py) ============

# 系统检查
python main.py check

# 单图处理
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

# 测试视觉模块
python tests/test_03_vision.py -i data/input/xianyu_bag_01.jpg --step all

# 测试创意模块
python tests/test_04_creative.py -i data/output/logs/vision/fused.png -t "爱马仕MiniKelly"

# 完整流程测试
python tests/run_full_pipeline.py
```

---

## 5. API 配置

### 5.1 config.yaml 配置文件

`conf/config.yaml` 包含所有运行时配置：

```yaml
# VLM 模型配置
vlm:
  default_model: "qwen3.5-plus"
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  timeout: 120
  max_concurrent: 2

# 文案生成模型配置
llm:
  default_model: "qwen-plus"
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  timeout: 30

# 本地SD配置
stable_diffusion:
  model_type: "sd_15"
  model_path: "models/.../snapshots/.../"
  num_inference_steps: 25
  guidance_scale: 7.5
  width: 1024
  height: 1024

# 抠图配置
matting:
  model_name: "u2net"

# 图像融合配置
fusion:
  mode: "rule"
  subject_height_ratio: 0.65
  shadow_blur_radius: 15
  shadow_alpha: 100

# 创意合成配置
creative:
  font_path: "C:/Windows/Fonts/msyh.ttc"
  default_font_size: 48
  default_text_bg: [0, 0, 0, 180]
  default_text_color: [255, 255, 255]
```

### 5.2 切换模型

修改 `config.yaml` 中的 `default_model` 即可切换模型：

```yaml
vlm:
  default_model: "qwen3.5-plus"  # 可选: qwen_vl_plus, glm_4v, kimi_vl
llm:
  default_model: "qwen-plus"     # 可选: qwen_max, glm_4, kimi
```

---

## 6. 输出文件说明

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

### 6.1 文案输出格式

`*_copy.json` 内容示例：

```json
{
  "title": "爱马仕MiniKelly二代｜09梦幻紫鳄鱼皮✨95新",
  "content": "自用收藏款～入手不到半年，只背过3次！09年经典梦幻紫美洲方块雾面鳄鱼皮..."
}
```

---

## 其他文档

- [English Version](../README.md) - English version of this documentation
