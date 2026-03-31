# Re-Value Agent 项目上下文

> 本文档包含项目的完整上下文，方便在新对话中直接复用。
> 如需详细文档，请参考 README.md。

---

## 1. 项目概述

**项目名称**: Re-Value Agent
**项目路径**: `D:\Code\AI_Playground\projects\ReValue-Agent\`
**课程**: CV期末大作业
**团队成员**: 赵钎年（架构）、李冠良（视觉）、马在飞（数据测试）

### 业务目标
构建一个基于VLM的二手商品图文创意自动化Agent。
- **输入**: 随手拍的商品原图 + 极简描述（如"九成新办公椅，100元"）
- **输出**: 电商级主图（抠图+专业背景+阴影融合+装饰+文字）+ 营销文案（标题+Emoji描述）

### 系统架构
异步并行 DAG 管线，四个节点：
```
Node 1 (VLM Router) ─┬─→ Node 2A (Vision) ─┐
                      │                      ├─→ Node 2C (Creative) → Output
                      └─→ Node 2B (Text)  ────┘
```

| Node | 职责 | 模型/工具 |
|------|------|----------|
| Node 1 | VLM联合感知 | qwen3.5-plus (阿里百炼) |
| Node 2A | 视觉流（抠图→生图→融合）| rembg u2net + SD 1.5 |
| Node 2B | 文案流 | qwen-plus (阿里百炼) |
| Node 2C | 创意合成（装饰+文字）| PIL 渲染 |

---

## 2. 技术方案

### 模型选型
| 模块 | 技术 | 说明 |
|------|------|------|
| VLM规划器 | **qwen3.5-plus** | 阿里百炼，多模态支持 |
| LLM文案 | **qwen-plus** | 阿里百炼 |
| 抠图 | rembg (u2net) | 本地运行，~176MB |
| 生图 | Stable Diffusion 1.5 | 本地运行，~6GB，RTX 4060可跑 |
| 框架 | Python asyncio | |
| Web UI | Gradio | |

### API配置
- **API Base URL**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **API Key**: `sk-a703bced5eff4a3db3202ab375d3d729`

### 并发控制
- `asyncio.Semaphore(2)` 限制并发峰值
- tenacity 指数退避重试（最大3次）

---

## 3. 目录结构

```
ReValue-Agent/
├── conf/
│   └── config.yaml          # 全局配置 (VLM/LLM模型、SD参数、抠图设置等)
├── core/                    # 核心业务（赵钎年主责）
│   ├── engine.py           # DAG调度引擎
│   ├── context.py          # Context数据类
│   ├── prompts.py          # Prompt模板管理
│   ├── node_router.py      # Node 1: VLM路由
│   ├── node_vision.py      # Node 2A: 视觉流
│   ├── node_text.py        # Node 2B: 文案流
│   └── node_creative.py     # Node 2C: 创意合成
├── tools/                   # 原子工具（李冠良主责）
│   ├── api_clients.py      # API客户端封装
│   ├── cv_utils.py         # 图像处理（抠图/阴影/融合）
│   ├── sd_local.py         # 本地SD推理
│   ├── text_renderer.py    # 文字渲染器
│   └── decorations.py       # 装饰绘制器
├── web/
│   └── gradio_app.py       # Gradio界面
├── data/
│   ├── original/           # 原始测试图片
│   ├── input/              # 测试图片
│   ├── intermediate/       # 中间产物
│   └── output/             # 最终输出 (含 full_report.json)
├── tests/
│   ├── test_vision.py      # 视觉单元测试
│   └── eval_batch.py       # 批量评估
├── requirements.txt
└── main.py                 # CLI入口
```

---

## 4. 当前进度

### ✅ 已完成
| 任务 | 状态 | 说明 |
|------|------|------|
| 环境配置 | ✅ | Python 3.10, CUDA, 所有依赖 |
| VLM API | ✅ | qwen3.5-plus 可用 |
| LLM API | ✅ | qwen-plus 可用 |
| rembg抠图 | ✅ | u2net模型已下载 |
| SD本地推理 | ✅ | SD 1.5 已下载并测试 |
| 完整Pipeline | ✅ | 端到端测试成功 |
| Node 2C创意合成 | ✅ | 装饰+文字渲染 |
| Gradio界面 | ✅ | 可用 |
| 详细日志报告 | ✅ | save_full_report() |

### 🔄 待优化
- 抠图精度（边缘处理）
- 背景融合自然度
- 文案生成质量

---

## 5. 快速开始

```bash
cd D:/Code/AI_Playground/projects/ReValue-Agent
venv/Scripts/activate

# 系统检查
python main.py check

# 单图处理
python main.py single -i data/input/xianyu_bag_01.jpg -t "爱马仕Kelly二手包"

# 批量处理
python tests/eval_batch.py --input data/input --output data/output

# Web界面
python main.py web --port 7860

# 单元测试
pytest tests/test_vision.py -v
```

---

## 6. 测试数据

| 图片 | 商品 | 平台 |
|------|------|------|
| xianyu_bag_01.jpg | 爱马仕Kelly包 | 闲鱼 |
| xianyu_shoes_01.jpg | 李宁球鞋 | 闲鱼 |
| shoes_02.jpg | Nike足球鞋 | 闲鱼 |
| xiaohongshu_bag_01.jpg | LV月亮包 | 小红书 |

---

## 7. 组员分工

| 成员 | 职责 | 文件 |
|------|------|------|
| 赵钎年 | Agent架构、core/、Gradio UI | core/*.py, web/gradio_app.py |
| 李冠良 | 视觉处理、tools/ | tools/*.py, core/node_vision.py |
| 马在飞 | 数据、测试、报告 | data/, tests/, core/node_text.py |

---

## 8. 输出文件说明

每次处理后 `data/output/` 会生成：

| 文件 | 说明 |
|------|------|
| `*_final.png` | 融合后的商品图 |
| `*_decorated.png` | **最终输出** - 带装饰和文案标题 |
| `*_matted.png` | rembg抠图结果 |
| `*_background.png` | SD生成的背景 |
| `*_copy.json` | 营销文案 |
| `*_context.json` | VLM分析结果 |
| `*_full_report.json` | **详细日志报告** - 包含每一步的输入输出 |

---

## 9. 后续工作

| 优先级 | 任务 | 负责人 |
|--------|------|--------|
| High | 批量测试与评估 | 马在飞 |
| High | 抠图精度优化 | 李冠良 |
| High | Gradio界面完善 | 赵钎年 |
| Medium | 测试数据扩充(30-50张) | 马在飞 |
| Low | CVPR格式报告撰写 | - |

---

*最后更新: 2026-03-31*
