"""
ReValue-Agent 快速测试脚本
===========================

直接运行此脚本测试各个组件：
```bash
cd D:/Code/AI_Playground/projects/ReValue-Agent
venv/Scripts/activate
python tests/run_tests.py
```

或指定图片测试：
```bash
python tests/run_tests.py --image data/input/xianyu_bag_01.jpg --text "爱马仕Kelly二手包"
```
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置加载函数
# ============================================================================

def load_config():
    """加载配置文件"""
    import yaml
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# ============================================================================
# 组件测试函数
# ============================================================================

def test_matting(image_path: str, output_dir: str = "data/intermediate") -> Path:
    """测试抠图"""
    from tools.cv_utils import MattingTool

    logger.info("Testing matting...")
    img = Image.open(image_path)
    matting_tool = MattingTool(model_name="u2net")
    matted = matting_tool.remove_background(img)

    output_path = Path(output_dir) / "test_matted.png"
    matted.save(output_path)
    logger.info(f"  -> Saved: {output_path}")
    return matted, output_path


def test_shadow(matted, background, output_dir: str = "data/intermediate") -> Path:
    """测试阴影生成"""
    from tools.cv_utils import ShadowRenderer

    logger.info("Testing shadow...")
    shadow_renderer = ShadowRenderer(blur_radius=15, shadow_alpha=100, offset_y=10)
    result = shadow_renderer.apply_shadow(matted, background)

    output_path = Path(output_dir) / "test_shadow.png"
    result.save(output_path)
    logger.info(f"  -> Saved: {output_path}")
    return output_path


def test_fusion(matted, background, output_dir: str = "data/intermediate") -> Path:
    """测试图像融合"""
    from tools.cv_utils import ImageFusion, ShadowRenderer

    logger.info("Testing fusion...")
    fusion = ImageFusion()
    result = fusion.fuse(matted, background, shadow_renderer=ShadowRenderer())

    output_path = Path(output_dir) / "test_fusion.png"
    result.save(output_path)
    logger.info(f"  -> Saved: {output_path}")
    return output_path


def test_decorations(image_path: str, output_dir: str = "data/intermediate") -> Path:
    """测试装饰渲染"""
    from tools.decorations import DecorationRenderer

    logger.info("Testing decorations...")

    elements = [
        {"type": "sparkle", "position": "top-right", "count": 3, "size": "medium"},
        {"type": "heart", "position": "top-right", "count": 2, "size": "medium"},
        {"type": "badge", "position": "bottom-left", "text": "HOT", "style": "rounded_rect"},
    ]

    img = Image.open(image_path)
    renderer = DecorationRenderer()
    result = renderer.render_elements(img, elements)

    output_path = Path(output_dir) / "test_decorated.png"
    result.save(output_path)
    logger.info(f"  -> Saved: {output_path}")
    return output_path


def test_text_renderer(image_path: str, text: str, output_dir: str = "data/intermediate") -> Path:
    """测试文字渲染"""
    from tools.text_renderer import TextRenderer

    logger.info("Testing text renderer...")

    img = Image.open(image_path)
    renderer = TextRenderer()
    result = renderer.render(
        image=img,
        text=text,
        position="bottom-center",
        max_width_ratio=0.85,
        font_size=48,
        background_color=(0, 0, 0, 180),
        text_color=(255, 255, 255)
    )

    output_path = Path(output_dir) / "test_text.png"
    result.save(output_path)
    logger.info(f"  -> Saved: {output_path}")
    return output_path


async def test_vlm_async(image_path: str, text: str) -> dict:
    """测试 VLM 分析（异步）"""
    from tools.api_clients import get_vlm_client
    from core.prompts import get_vlm_prompts, parse_json_from_response

    logger.info("Testing VLM...")

    # 读取图片并转为 base64
    img = Image.open(image_path)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    # 获取 prompt
    prompts = get_vlm_prompts(text)

    # 调用 VLM
    config = load_config()
    client = get_vlm_client("qwen3.5-plus", config)

    response = await client.chat_with_image(
        image_base64=image_base64,
        system_prompt=prompts["system"],
        user_prompt=prompts["user"],
    )

    result = parse_json_from_response(response)
    logger.info(f"  -> subject: {result.get('subject')}")
    logger.info(f"  -> selling_points: {result.get('selling_points')[:2]}...")
    return result


def test_vlm(image_path: str, text: str) -> dict:
    """测试 VLM 分析（同步包装）"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(test_vlm_async(image_path, text))
    finally:
        loop.close()


async def test_copy_async(subject: str, selling_points: list, original_text: str, platform: str = "闲鱼体") -> dict:
    """测试文案生成（异步）"""
    from core.node_text import NodeText
    from core.context import Context

    logger.info(f"Testing copy generation ({platform})...")

    node_text = NodeText(load_config())
    ctx = Context(
        subject=subject,
        selling_points=selling_points,
        original_text=original_text,
        platform_style=platform,
    )
    ctx = await node_text.process(ctx)

    result = ctx.final_copy or {"title": subject, "content": original_text}
    logger.info(f"  -> title: {result.get('title')}")
    return result


def test_copy(subject: str, selling_points: list, original_text: str, platform: str = "闲鱼体") -> dict:
    """测试文案生成（同步包装）"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(test_copy_async(subject, selling_points, original_text, platform))
    finally:
        loop.close()


# ============================================================================
# 主测试流程
# ============================================================================

def run_all_tests(image_path: str = "data/input/xianyu_bag_01.jpg", text: str = "爱马仕Kelly二手包"):
    """运行所有组件测试"""
    print("=" * 60)
    print("ReValue-Agent 组件测试")
    print("=" * 60)
    print()

    output_dir = Path("data/intermediate")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 测试抠图
    print("[1/6] 抠图测试")
    matted, matted_path = test_matting(image_path, output_dir)
    print()

    # 2. 创建测试背景
    print("[2/6] 创建测试背景")
    bg = Image.new('RGB', (1024, 1024), color=(240, 240, 245))
    bg_path = output_dir / "test_bg.png"
    bg.save(bg_path)
    logger.info(f"  -> Saved: {bg_path}")
    print()

    # 3. 测试融合
    print("[3/6] 图像融合测试")
    fusion_path = test_fusion(matted, bg, output_dir)
    print()

    # 4. 测试装饰
    print("[4/6] 装饰渲染测试")
    decor_path = test_decorations(str(fusion_path), output_dir)
    print()

    # 5. 测试文字
    print("[5/6] 文字渲染测试")
    text_path = test_text_renderer(str(decor_path), f"✨ {text}", output_dir)
    print()

    # 6. 测试 VLM（需要 API）
    print("[6/6] VLM 分析测试")
    try:
        vlm_result = test_vlm(image_path, text)
        print(f"  -> VLM result: {json.dumps(vlm_result, ensure_ascii=False, indent=4)[:200]}...")
    except Exception as e:
        print(f"  -> VLM failed: {e}")
    print()

    # 7. 测试文案生成
    print("[Bonus] 文案生成测试")
    try:
        copy_result = test_copy(
            subject="手提包",
            selling_points=["九成新", "爱马仕", "MiniKelly"],
            original_text=text,
            platform="闲鱼体"
        )
        print(f"  -> title: {copy_result.get('title')}")
        print(f"  -> content: {copy_result.get('content')[:100]}...")
    except Exception as e:
        print(f"  -> Copy failed: {e}")
    print()

    print("=" * 60)
    print("测试完成！输出文件在: data/intermediate/")
    print("=" * 60)

    # 输出文件列表
    print()
    print("生成的文件:")
    for f in sorted(output_dir.glob("test_*")):
        size = f.stat().st_size
        print(f"  - {f.name} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReValue-Agent 组件测试")
    parser.add_argument("--image", "-i", default="data/input/xianyu_bag_01.jpg",
                        help="测试图片路径")
    parser.add_argument("--text", "-t", default="爱马仕Kelly二手包",
                        help="商品描述")

    args = parser.parse_args()
    run_all_tests(args.image, args.text)
