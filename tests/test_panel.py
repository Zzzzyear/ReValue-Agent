"""
ReValue-Agent 组件测试面板
==========================

本模块提供独立的组件测试功能，方便逐个测试和优化各个模块。

## 使用方式

### 1. 启动测试面板（推荐）

```bash
cd D:/Code/AI_Playground/projects/ReValue-Agent
venv/Scripts/activate
python tests/test_panel.py
# 浏览器打开 http://localhost:7861
```

### 2. 命令行测试（快速验证）

```bash
# 测试 VLM
python -c "
from tests.test_panel import test_vlm
from PIL import Image
img = Image.open('data/input/xianyu_bag_01.jpg')
result = test_vlm(img, '爱马仕Kelly二手包')
print(result)
"

# 测试文案生成
python -c "
from tests.test_panel import test_copy_generation
result = test_copy_generation('运动鞋', ['九成新', '性价比高'], 'Nike足球鞋九成新')
print(result)
"

# 测试抠图
python -c "
from tests.test_panel import test_matting
from PIL import Image
img = Image.open('data/input/xianyu_bag_01.jpg')
result = test_matting(img)
result.save('test_matted.png')
"

# 测试装饰渲染
python -c "
from tests.test_panel import test_decorations
from PIL import Image
img = Image.open('data/output/test_final.png')
result = test_decorations(img, '闪光+心形')
result.save('test_decorated.png')
"
```

### 3. 在 Python 代码中导入使用

```python
from tests.test_panel import (
    test_vlm,
    test_copy_generation,
    test_matting,
    test_shadow,
    test_fusion,
    test_decorations,
    test_text_renderer,
    test_sd_generation,
)
```

---

## 组件测试函数清单

| 函数 | 功能 | 主要输入 | 输出 |
|------|------|----------|------|
| `test_vlm()` | VLM 商品分析 | PIL.Image, str(描述) | dict (subject, points, prompt, box, elements) |
| `test_copy_generation()` | 文案生成 | str(subject), list(points), str(original_text) | dict (title, content) |
| `test_matting()` | 抠图 | PIL.Image | PIL.Image (RGBA) |
| `test_shadow()` | 阴影生成 | PIL.Image (RGBA) | PIL.Image (RGB) |
| `test_fusion()` | 图像融合 | PIL.Image (主体), PIL.Image (背景) | PIL.Image (RGB) |
| `test_decorations()` | 装饰渲染 | PIL.Image, str(装饰类型) | PIL.Image (RGB) |
| `test_text_renderer()` | 文字渲染 | PIL.Image, str(文案) | PIL.Image (RGB) |
| `test_sd_generation()` | SD 背景生成 | str(背景描述) | PIL.Image (RGB) |

---

## 各组件详解

### 1. test_vlm(image, text)
**VLM 商品分析测试**

输入商品图片和描述，调用 qwen3.5-plus 分析并返回结构化信息。

```python
from PIL import Image
from tests.test_panel import test_vlm

img = Image.open('data/input/xianyu_bag_01.jpg')
result = test_vlm(img, '爱马仕Kelly二手包')

# result 格式:
# {
#     'subject': '手提包',
#     'selling_points': ['MiniKelly二代', '09梦幻紫', ...],
#     'background_prompt': 'luxury boutique interior, ...',
#     'reference_box': [x_min, y_min, x_max, y_max],
#     'creative_elements': [{'type': 'sparkle', ...}, ...],
#     'text_placement': {'region': 'bottom_center', ...}
# }
```

### 2. test_copy_generation(subject, selling_points, original_text, platform='闲鱼体')
**文案生成测试**

根据商品信息生成营销文案。

```python
from tests.test_panel import test_copy_generation

result = test_copy_generation(
    subject='运动鞋',
    selling_points=['九成新', '性价比高', 'Nike正品'],
    original_text='Nike足球鞋九成新',
    platform='闲鱼体'  # 或 '小红书体'
)

# result 格式:
# {
#     'title': '🔥 Nike足球鞋 九成新 超低价转让',
#     'content': '姐妹们！这双鞋真的太舍不得了...'
# }
```

### 3. test_matting(image, model='u2net')
**抠图测试**

移除图片背景，返回 RGBA 格式的透明背景图。

```python
from PIL import Image
from tests.test_panel import test_matting

img = Image.open('data/input/xianyu_bag_01.jpg')
matted = test_matting(img, model='u2net')
matted.save('output_matted.png')  # 保存抠图结果

# 可选模型: 'u2net', 'u2netp', 'u2net_human_seg', 'silueta'
```

### 4. test_shadow(matted_image, blur_radius=15, shadow_alpha=100, offset_y=10)
**阴影生成测试**

为抠图主体生成自然阴影。

```python
from PIL import Image
from tests.test_panel import test_shadow, test_matting

# 先抠图
img = Image.open('data/input/xianyu_bag_01.jpg')
matted = test_matting(img)

# 生成阴影
shadowed = test_shadow(matted, blur_radius=20, shadow_alpha=120, offset_y=15)
shadowed.save('output_shadow.png')
```

### 5. test_fusion(subject_rgba, background, reference_box=None, shadow=True)
**图像融合测试**

将抠图主体融合到背景图中。

```python
from PIL import Image
from tests.test_panel import test_fusion, test_matting, test_shadow

# 抠图
img = Image.open('data/input/xianyu_bag_01.jpg')
matted = test_matting(img)

# 生成背景 (需要先有 background_prompt)
# bg = test_sd_generation('luxury boutique interior')

# 融合
if 'bg' in dir():
    final = test_fusion(matted, bg, shadow=True)
    final.save('output_fusion.png')
```

### 6. test_decorations(image, decoration_types='全部', position='top-right')
**装饰渲染测试**

在图片上渲染装饰元素。

```python
from PIL import Image
from tests.test_panel import test_decorations

img = Image.open('data/output/final.png')

# 测试全部装饰
result = test_decorations(img, '全部')

# 测试指定装饰
result = test_decorations(img, '闪光+心形+徽章')

# 指定位置
result = test_decorations(img, '价格标签', position='bottom-left')
```

装饰类型选项：
- '全部' - 所有装饰类型
- '闪光' - 金色十字闪光
- '星星' - 五角星
- '心形' - 粉色心形
- '徽章' - 红色徽章
- '价格标签' - 价格标签
- '丝带' - 丝带装饰
- '圆形' - 圆形装饰

### 7. test_text_renderer(image, text, position='bottom-center', font_size=48)
**文字渲染测试**

将文案渲染到图片上。

```python
from PIL import Image
from tests.test_panel import test_text_renderer

img = Image.open('data/output/final.png')

result = test_text_renderer(
    img,
    text='✨ MiniKelly二代 09梦幻紫',
    position='bottom-center',  # top-left, top-center, top-right, bottom-left, bottom-center, bottom-right
    font_size=48
)
result.save('output_text.png')
```

### 8. test_sd_generation(prompt, num_steps=25, guidance_scale=7.5)
**Stable Diffusion 背景生成测试**

使用 SD 生成背景图。

```python
from tests.test_panel import test_sd_generation

# 生成背景
bg = test_sd_generation(
    'luxury boutique interior, soft warm lighting, elegant and sophisticated',
    num_steps=25  # 步数越多质量越高但越慢
)
bg.save('output_background.png')

# 快速测试（少步数）
bg_fast = test_sd_generation('minimalist white background', num_steps=10)
```

---

## 测试面板界面说明

启动测试面板后 (http://localhost:7861)，界面分为以下标签页：

### 1. VLM 测试
- 上传商品图片
- 输入商品描述
- 查看 VLM 分析结果（subject, selling_points, background_prompt 等）

### 2. 文案生成测试
- 输入商品信息
- 选择平台风格
- 查看生成的标题和正文

### 3. 视觉处理测试
- 测试抠图
- 测试阴影
- 测试融合
- 查看中间结果

### 4. 装饰与文字测试
- 测试装饰渲染
- 测试文字渲染
- 调整参数并预览效果

### 5. 完整流程测试
- 输入图片和描述
- 一键运行完整 pipeline
- 查看所有输出和报告

"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from PIL import Image

# 导入核心组件
from core.engine import ReValueEngine
from core.context import Context, TaskStatus
from core.prompts import get_vlm_prompts, parse_json_from_response
from tools.api_clients import get_vlm_client, get_llm_client
from tools.cv_utils import MattingTool, ShadowRenderer, ImageFusion
from tools.sd_local import get_sd_generator

logger = logging.getLogger(__name__)

# ============================================================================
# 组件测试函数（供命令行和代码直接调用）
# ============================================================================

def test_vlm(image: Image.Image, text: str, model: str = "qwen3.5-plus") -> Dict:
    """
    测试 VLM 商品分析

    Args:
        image: PIL.Image 商品图片
        text: str 商品描述
        model: str 模型名称 (默认 qwen3.5-plus)

    Returns:
        dict: {
            'subject': str,           # 商品品类
            'selling_points': list,  # 卖点列表
            'background_prompt': str, # SD 背景提示词
            'reference_box': list,    # 参考框 [x_min, y_min, x_max, y_max]
            'creative_elements': list,# 创意元素建议
            'text_placement': dict   # 文字位置建议
        }

    Example:
        from PIL import Image
        from tests.test_panel import test_vlm
        img = Image.open('data/input/xianyu_bag_01.jpg')
        result = test_vlm(img, '爱马仕Kelly二手包')
        print(result['subject'])  # '手提包'
    """
    from core.engine import ReValueEngine

    # 将图片转为 base64
    image_base64 = ReValueEngine.image_to_base64(image)

    # 获取 prompt
    prompts = get_vlm_prompts(text)

    # 调用 VLM（同步方式）
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        async def _call():
            config = load_config()
            client = get_vlm_client(model, config)
            return await client.chat_with_image(
                image_base64=image_base64,
                system_prompt=prompts["system"],
                user_prompt=prompts["user"],
            )

        response = loop.run_until_complete(_call())
        result = parse_json_from_response(response)
        return result
    finally:
        loop.close()


def test_copy_generation(
    subject: str,
    selling_points: List[str],
    original_text: str,
    platform: str = "闲鱼体",
    model: str = "qwen-plus"
) -> Dict:
    """
    测试文案生成

    Args:
        subject: str 商品名称/品类
        selling_points: list 卖点列表
        original_text: str 原始描述
        platform: str 平台风格 ('闲鱼体' 或 '小红书体')
        model: str 模型名称

    Returns:
        dict: {'title': str, 'content': str}

    Example:
        from tests.test_panel import test_copy_generation
        result = test_copy_generation(
            subject='运动鞋',
            selling_points=['九成新', '性价比高'],
            original_text='Nike足球鞋九成新',
            platform='闲鱼体'
        )
        print(result['title'])  # '🔥 Nike足球鞋 九成新...'
    """
    from core.node_text import NodeText

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        async def _call():
            config = NodeText({}).config
            node_text = NodeText(config)
            ctx = Context(
                subject=subject,
                selling_points=selling_points,
                original_text=original_text,
                platform_style=platform,
            )
            return await node_text.process(ctx)

        ctx = loop.run_until_complete(_call())
        return ctx.final_copy or {"title": subject, "content": original_text}
    finally:
        loop.close()


def test_matting(image: Image.Image, model: str = "u2net") -> Image.Image:
    """
    测试抠图

    Args:
        image: PIL.Image 输入图片
        model: str 抠图模型 ('u2net', 'u2netp', 'u2net_human_seg', 'silueta')

    Returns:
        PIL.Image: RGBA 格式，透明背景

    Example:
        from PIL import Image
        from tests.test_panel import test_matting
        img = Image.open('data/input/xianyu_bag_01.jpg')
        matted = test_matting(img)
        matted.save('output_matted.png')
    """
    matting_tool = MattingTool(model_name=model)
    return matting_tool.remove_background(image)


def test_shadow(
    matted_image: Image.Image,
    blur_radius: int = 15,
    shadow_alpha: int = 100,
    offset_y: int = 10
) -> Image.Image:
    """
    测试阴影生成

    Args:
        matted_image: PIL.Image RGBA 抠图图片
        blur_radius: int 模糊半径
        shadow_alpha: int 阴影透明度 (0-255)
        offset_y: int 向下偏移量

    Returns:
        PIL.Image: RGB 格式，带阴影

    Example:
        from tests.test_panel import test_matting, test_shadow
        img = Image.open('data/input/xianyu_bag_01.jpg')
        matted = test_matting(img)
        shadowed = test_shadow(matted, blur_radius=20)
        shadowed.save('output_shadow.png')
    """
    shadow_renderer = ShadowRenderer(
        blur_radius=blur_radius,
        shadow_alpha=shadow_alpha,
        offset_y=offset_y
    )
    return shadow_renderer.apply_shadow(matted_image)


def test_fusion(
    subject_rgba: Image.Image,
    background: Image.Image,
    reference_box: Optional[List[float]] = None,
    shadow: bool = True
) -> Image.Image:
    """
    测试图像融合

    Args:
        subject_rgba: PIL.Image RGBA 抠图主体
        background: PIL.Image 背景图
        reference_box: list [x_min, y_min, x_max, y_max] 可选参考框
        shadow: bool 是否添加阴影

    Returns:
        PIL.Image: RGB 融合结果

    Example:
        from tests.test_panel import test_fusion, test_matting, test_sd_generation
        img = Image.open('data/input/xianyu_bag_01.jpg')
        matted = test_matting(img)
        bg = test_sd_generation('luxury boutique interior')
        fused = test_fusion(matted, bg)
        fused.save('output_fusion.png')
    """
    fusion = ImageFusion()
    return fusion.fuse(
        subject_rgba=subject_rgba,
        background=background,
        reference_box=reference_box,
        shadow_renderer=ShadowRenderer() if shadow else None
    )


def test_decorations(
    image: Image.Image,
    decoration_types: str = "全部",
    position: str = "top-right"
) -> Image.Image:
    """
    测试装饰渲染

    Args:
        image: PIL.Image 输入图片
        decoration_types: str 装饰类型组合，用 '+' 分隔
                          选项: '全部', '闪光', '星星', '心形', '徽章', '价格标签', '丝带', '圆形'
                          例如: '闪光+心形+徽章'
        position: str 装饰位置
                  选项: 'top-left', 'top-right', 'bottom-left', 'bottom-right',
                        'top-center', 'bottom-center'

    Returns:
        PIL.Image: RGB 渲染结果

    Example:
        from PIL import Image
        from tests.test_panel import test_decorations
        img = Image.open('data/output/final.png')
        result = test_decorations(img, '闪光+心形', 'top-right')
        result.save('output_decorated.png')
    """
    from tools.decorations import DecorationRenderer

    # 解析装饰类型
    if decoration_types == "全部":
        elements = [
            {"type": "sparkle", "position": position, "count": 3, "size": "medium"},
            {"type": "heart", "position": position, "count": 2, "size": "medium"},
            {"type": "badge", "position": "bottom-left", "text": "HOT", "style": "rounded_rect"},
        ]
    else:
        type_map = {
            "闪光": {"type": "sparkle", "position": position, "count": 3, "size": "medium"},
            "星星": {"type": "star", "position": position, "count": 2, "size": "medium"},
            "心形": {"type": "heart", "position": position, "count": 2, "size": "medium"},
            "徽章": {"type": "badge", "position": position, "text": "SALE", "style": "rounded_rect"},
            "价格标签": {"type": "price_tag", "position": position, "text": "特惠"},
            "丝带": {"type": "ribbon", "position": position},
            "圆形": {"type": "circle_decoration", "position": position, "size": "medium"},
        }
        elements = []
        for dec_type in decoration_types.split('+'):
            dec_type = dec_type.strip()
            if dec_type in type_map:
                elements.append(type_map[dec_type])

    renderer = DecorationRenderer()
    return renderer.render_elements(image, elements)


def test_text_renderer(
    image: Image.Image,
    text: str,
    position: str = "bottom-center",
    font_size: int = 48,
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 180),
    text_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    测试文字渲染

    Args:
        image: PIL.Image 输入图片
        text: str 要渲染的文字
        position: str 文字位置
        font_size: int 字体大小
        bg_color: tuple 背景颜色 (R, G, B, A)
        text_color: tuple 文字颜色 (R, G, B)

    Returns:
        PIL.Image: RGB 渲染结果

    Example:
        from PIL import Image
        from tests.test_panel import test_text_renderer
        img = Image.open('data/output/final.png')
        result = test_text_renderer(
            img,
            text='✨ MiniKelly二代 09梦幻紫',
            position='bottom-center',
            font_size=48
        )
        result.save('output_text.png')
    """
    from tools.text_renderer import TextRenderer

    renderer = TextRenderer()
    return renderer.render(
        image=image,
        text=text,
        position=position,
        max_width_ratio=0.85,
        font_size=font_size,
        background_color=bg_color,
        text_color=text_color
    )


def test_sd_generation(
    prompt: str,
    num_steps: int = 25,
    guidance_scale: float = 7.5,
    width: int = 1024,
    height: int = 1024
) -> Image.Image:
    """
    测试 Stable Diffusion 背景生成

    Args:
        prompt: str 背景描述 (英文效果更好)
        num_steps: int 推理步数 (10-30，越多越慢质量越高)
        guidance_scale: float CFG 引导强度 (5-15)
        width: int 输出宽度
        height: int 输出高度

    Returns:
        PIL.Image: RGB 生成的背景图

    Example:
        from tests.test_panel import test_sd_generation
        bg = test_sd_generation(
            'luxury boutique interior, soft warm lighting',
            num_steps=25
        )
        bg.save('output_background.png')

        # 快速测试
        bg_fast = test_sd_generation('minimalist white background', num_steps=10)
    """
    from tools.sd_local import get_sd_generator

    config = ReValueEngine.load_config()
    sd_generator = get_sd_generator(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = loop.run_in_executor(
            None,
            lambda: sd_generator.generate(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            )
        )
        return loop.run_until_complete(result)
    finally:
        loop.close()


# ============================================================================
# Gradio 测试面板界面
# ============================================================================

def load_config():
    """加载配置"""
    import yaml
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class TestPanel:
    """组件测试面板"""

    def __init__(self):
        config = load_config()
        self.engine = ReValueEngine(config)

    # ----- VLM 测试 -----
    def test_vlm_ui(self, image: Image.Image, text: str, model: str):
        """VLM 测试界面函数"""
        if image is None:
            return None, "请上传商品图片"
        try:
            result = test_vlm(image, text, model)
            import json
            return json.dumps(result, ensure_ascii=False, indent=2), f"✅ VLM 分析成功\nsubject: {result.get('subject', 'N/A')}"
        except Exception as e:
            return None, f"❌ VLM 分析失败: {str(e)}"

    # ----- 文案生成测试 -----
    def test_copy_ui(
        self,
        subject: str,
        selling_points: str,
        original_text: str,
        platform: str,
        model: str
    ):
        """文案生成测试界面函数"""
        if not subject.strip():
            return "请输入商品名称", "请先输入商品信息"
        try:
            points = [p.strip() for p in selling_points.split(',') if p.strip()]
            result = test_copy_generation(subject, points, original_text, platform, model)
            return result.get('title', ''), result.get('content', '')
        except Exception as e:
            return f"❌ 文案生成失败: {str(e)}", ""

    # ----- 抠图测试 -----
    def test_matting_ui(self, image: Image.Image, model: str):
        """抠图测试界面函数"""
        if image is None:
            return None, "请上传图片"
        try:
            result = test_matting(image, model)
            return result, f"✅ 抠图完成\n尺寸: {result.size}\n模式: {result.mode}"
        except Exception as e:
            return None, f"❌ 抠图失败: {str(e)}"

    # ----- 阴影测试 -----
    def test_shadow_ui(
        self,
        matted_image: Image.Image,
        blur_radius: int,
        shadow_alpha: int,
        offset_y: int
    ):
        """阴影测试界面函数"""
        if matted_image is None:
            return None, "请先进行抠图"
        try:
            result = test_shadow(matted_image, blur_radius, shadow_alpha, offset_y)
            return result, f"✅ 阴影生成完成\n参数: blur={blur_radius}, alpha={shadow_alpha}, offset={offset_y}"
        except Exception as e:
            return None, f"❌ 阴影生成失败: {str(e)}"

    # ----- SD 生成测试 -----
    def test_sd_ui(self, prompt: str, num_steps: int):
        """SD 生成测试界面函数"""
        if not prompt.strip():
            return None, "请输入背景描述"
        try:
            result = test_sd_generation(prompt, num_steps)
            return result, f"✅ SD 生成完成\n尺寸: {result.size}"
        except Exception as e:
            return None, f"❌ SD 生成失败: {str(e)}"

    # ----- 装饰测试 -----
    def test_decorations_ui(
        self,
        image: Image.Image,
        decoration_types: str,
        position: str
    ):
        """装饰测试界面函数"""
        if image is None:
            return None, "请上传图片"
        try:
            result = test_decorations(image, decoration_types, position)
            return result, f"✅ 装饰渲染完成\n类型: {decoration_types}\n位置: {position}"
        except Exception as e:
            return None, f"❌ 装饰渲染失败: {str(e)}"

    # ----- 文字渲染测试 -----
    def test_text_ui(
        self,
        image: Image.Image,
        text: str,
        position: str,
        font_size: int
    ):
        """文字渲染测试界面函数"""
        if image is None:
            return None, "请上传图片"
        if not text.strip():
            return None, "请输入要渲染的文字"
        try:
            result = test_text_renderer(image, text, position, font_size)
            return result, f"✅ 文字渲染完成\n文字: {text[:20]}..."
        except Exception as e:
            return None, f"❌ 文字渲染失败: {str(e)}"

    # ----- 融合测试 -----
    def test_fusion_ui(
        self,
        matted_image: Image.Image,
        background_image: Image.Image,
        add_shadow: bool
    ):
        """融合测试界面函数"""
        if matted_image is None or background_image is None:
            return None, "请先提供抠图结果和背景图"
        try:
            result = test_fusion(matted_image, background_image, shadow=add_shadow)
            return result, f"✅ 融合完成\n尺寸: {result.size}"
        except Exception as e:
            return None, f"❌ 融合失败: {str(e)}"

    def build_ui(self):
        """构建 Gradio UI"""
        with gr.Blocks(
            title="ReValue-Agent 组件测试面板",
            theme=gr.themes.Soft(),
        ) as app:
            gr.Markdown(
                """
                # ReValue-Agent 组件测试面板
                ## 独立测试各个组件，方便优化和调试

                ### 使用说明
                1. 选择对应的标签页测试单个组件
                2. 每个组件都有独立的输入输出
                3. 中间结果可以保存到本地
                4. 查看最下方的函数清单和使用示例
                """
            )

            # ----- VLM 测试 -----
            with gr.Tab("VLM 测试"):
                with gr.Row():
                    with gr.Column():
                        vlm_image = gr.Image(label="商品图片", type="pil", height=250)
                        vlm_text = gr.Textbox(label="商品描述", placeholder="例如：爱马仕Kelly二手包")
                        vlm_model = gr.Dropdown(
                            label="VLM 模型",
                            choices=["qwen3.5-plus", "qwen-vl-plus"],
                            value="qwen3.5-plus"
                        )
                        vlm_btn = gr.Button("分析", variant="primary")
                    with gr.Column():
                        vlm_output = gr.JSON(label="VLM 分析结果")
                        vlm_status = gr.Textbox(label="状态", lines=2)

                vlm_btn.click(
                    self.test_vlm_ui,
                    inputs=[vlm_image, vlm_text, vlm_model],
                    outputs=[vlm_output, vlm_status]
                )

            # ----- 文案生成测试 -----
            with gr.Tab("文案生成测试"):
                with gr.Row():
                    with gr.Column():
                        copy_subject = gr.Textbox(label="商品名称", placeholder="例如：运动鞋")
                        copy_points = gr.Textbox(
                            label="卖点（逗号分隔）",
                            placeholder="例如：九成新,性价比高,Nike正品"
                        )
                        copy_original = gr.Textbox(
                            label="原始描述",
                            placeholder="例如：Nike足球鞋九成新"
                        )
                        copy_platform = gr.Dropdown(
                            label="平台风格",
                            choices=["闲鱼体", "小红书体"],
                            value="闲鱼体"
                        )
                        copy_btn = gr.Button("生成文案", variant="primary")
                    with gr.Column():
                        copy_title = gr.Textbox(label="生成标题", lines=1)
                        copy_content = gr.Textbox(label="生成正文", lines=5)

                copy_btn.click(
                    self.test_copy_ui,
                    inputs=[copy_subject, copy_points, copy_original, copy_platform, gr.Textbox(value="qwen-plus", visible=False)],
                    outputs=[copy_title, copy_content]
                )

            # ----- 视觉处理测试 -----
            with gr.Tab("视觉处理测试"):
                with gr.Tab("抠图"):
                    with gr.Row():
                        with gr.Column():
                            mat_image = gr.Image(label="原图", type="pil", height=250)
                            mat_model = gr.Dropdown(
                                label="抠图模型",
                                choices=["u2net", "u2netp", "u2net_human_seg"],
                                value="u2net"
                            )
                            mat_btn = gr.Button("抠图", variant="primary")
                        with gr.Column():
                            mat_output = gr.Image(label="抠图结果 (RGBA)", type="pil", height=250)
                            mat_status = gr.Textbox(label="状态")

                    mat_btn.click(
                        self.test_matting_ui,
                        inputs=[mat_image, mat_model],
                        outputs=[mat_output, mat_status]
                    )

                with gr.Tab("SD 背景生成"):
                    with gr.Row():
                        with gr.Column():
                            sd_prompt = gr.Textbox(
                                label="背景描述 (英文)",
                                placeholder="例如：luxury boutique interior, soft warm lighting"
                            )
                            sd_steps = gr.Slider(label="推理步数", minimum=5, maximum=50, value=25)
                            sd_btn = gr.Button("生成背景", variant="primary")
                        with gr.Column():
                            sd_output = gr.Image(label="生成背景", type="pil", height=250)
                            sd_status = gr.Textbox(label="状态")

                    sd_btn.click(
                        self.test_sd_ui,
                        inputs=[sd_prompt, sd_steps],
                        outputs=[sd_output, sd_status]
                    )

                with gr.Tab("图像融合"):
                    with gr.Row():
                        with gr.Column():
                            fus_subject = gr.Image(label="抠图主体 (RGBA)", type="pil", height=200)
                            fus_bg = gr.Image(label="背景图", type="pil", height=200)
                            fus_shadow = gr.Checkbox(label="添加阴影", value=True)
                            fus_btn = gr.Button("融合", variant="primary")
                        with gr.Column():
                            fus_output = gr.Image(label="融合结果", type="pil", height=300)
                            fus_status = gr.Textbox(label="状态")

                    fus_btn.click(
                        self.test_fusion_ui,
                        inputs=[fus_subject, fus_bg, fus_shadow],
                        outputs=[fus_output, fus_status]
                    )

            # ----- 装饰与文字测试 -----
            with gr.Tab("装饰与文字测试"):
                with gr.Tab("装饰渲染"):
                    with gr.Row():
                        with gr.Column():
                            dec_image = gr.Image(label="图片", type="pil", height=250)
                            dec_type = gr.Dropdown(
                                label="装饰类型",
                                choices=["全部", "闪光", "星星", "心形", "徽章", "价格标签", "丝带", "圆形"],
                                value="全部"
                            )
                            dec_position = gr.Dropdown(
                                label="位置",
                                choices=["top-left", "top-right", "bottom-left", "bottom-right", "top-center", "bottom-center"],
                                value="top-right"
                            )
                            dec_btn = gr.Button("渲染装饰", variant="primary")
                        with gr.Column():
                            dec_output = gr.Image(label="装饰结果", type="pil", height=250)
                            dec_status = gr.Textbox(label="状态")

                    dec_btn.click(
                        self.test_decorations_ui,
                        inputs=[dec_image, dec_type, dec_position],
                        outputs=[dec_output, dec_status]
                    )

                with gr.Tab("文字渲染"):
                    with gr.Row():
                        with gr.Column():
                            txt_image = gr.Image(label="图片", type="pil", height=250)
                            txt_text = gr.Textbox(
                                label="要渲染的文字",
                                placeholder="例如：✨ MiniKelly二代 09梦幻紫"
                            )
                            txt_position = gr.Dropdown(
                                label="位置",
                                choices=["top-left", "top-right", "bottom-left", "bottom-right", "top-center", "bottom-center"],
                                value="bottom-center"
                            )
                            txt_size = gr.Slider(label="字体大小", minimum=24, maximum=96, value=48)
                            txt_btn = gr.Button("渲染文字", variant="primary")
                        with gr.Column():
                            txt_output = gr.Image(label="文字渲染结果", type="pil", height=250)
                            txt_status = gr.Textbox(label="状态")

                    txt_btn.click(
                        self.test_text_ui,
                        inputs=[txt_image, txt_text, txt_position, txt_size],
                        outputs=[txt_output, txt_status]
                    )

            # ----- 使用说明 -----
            with gr.Tab("使用说明"):
                gr.Markdown(open(__file__, encoding='utf-8').read())

        return app

    def launch(self, server_port: int = 7861):
        """启动测试面板"""
        app = self.build_ui()
        logger.info(f"Starting Test Panel on port {server_port}")
        app.launch(server_name="0.0.0.0", server_port=server_port, share=False)


def main():
    """主入口"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    panel = TestPanel()
    panel.launch()


if __name__ == "__main__":
    main()
