"""
ReValue-Agent - Test 04: Creative 创意渲染模块测试
====================================================

测试创意渲染模块（装饰元素、文字渲染），记录完整输入输出。

## 运行命令

```bash
cd D:/Code/AI_Playground/projects/ReValue-Agent
venv\Scripts\activate

# 基本用法
python tests/test_04_creative.py -i data/output/logs/vision/fused.png -t "爱马仕MiniKelly二手包"

# 自定义装饰元素
python tests/test_04_creative.py -i data/output/logs/vision/fused.png -t "爱马仕MiniKelly" -de '{"type":"sparkle","position":"top-right","count":3}'

# 仅测试装饰渲染
python tests/test_04_creative.py -i data/output/logs/vision/fused.png --step decorations

# 仅测试文字渲染
python tests/test_04_creative.py -i data/output/logs/vision/fused.png -t "爱马仕MiniKelly" --step text

# 指定输出目录
python tests/test_04_creative.py -i data/output/logs/vision/fused.png -t "爱马仕MiniKelly" -o data/output/logs/creative
```

## 输出文件

在 data/output/logs/creative/ 目录下生成:
- {timestamp}_{imagename}_creative_log.json - 完整日志
- {timestamp}_{imagename}_decorated.png - 仅装饰结果
- {timestamp}_{imagename}_with_text.png - 带文字结果

## 日志内容

{
  "test_info": {
    "module": "creative",
    "timestamp": "...",
    "image_path": "...",
    "title": "...",
    "text_placement": {...},
    "step": "all/decorations/text"
  },
  "decorations": {
    "elements": [...],
    "elapsed_seconds": 0.5
  },
  "text": {
    "title": "...",
    "position": "...",
    "elapsed_seconds": 0.3
  },
  "output_images": {
    "decorated": "path/to/decorated.png",
    "with_text": "path/to/with_text.png"
  },
  "status": "success/failed",
  "total_elapsed_seconds": 1.2,
  "error": "错误信息（如果有）"
}
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.node_creative import NodeCreative

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config():
    """加载配置"""
    import yaml
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def parse_elements(elements_str: str) -> list:
    """解析装饰元素 JSON 字符串"""
    if not elements_str:
        return []
    try:
        import json as json_module
        return json_module.loads(elements_str)
    except:
        return []


async def test_creative_module(
    image_path: str,
    title: str = "",
    text_placement: dict = None,
    elements: list = None,
    step: str = "all",
    output_dir: str = "data/output/logs/creative",
) -> dict:
    """
    测试 Creative 创意渲染模块

    Args:
        image_path: 图像路径
        title: 标题文字
        text_placement: 文字放置配置
        elements: 装饰元素列表
        step: 测试步骤 all/decorations/text
        output_dir: 输出目录

    Returns:
        测试结果字典
    """
    start_time = time.time()

    # 默认文字放置配置
    if text_placement is None:
        text_placement = {"region": "bottom_center", "max_width_ratio": 0.8}

    # 默认装饰元素
    if elements is None:
        elements = [
            {"type": "sparkle", "position": "top-right", "count": 3, "size": "medium"},
            {"type": "badge", "position": "bottom-left", "text": "SALE", "style": "rounded_rect"},
        ]

    # 记录完整输入输出
    log_record = {
        "test_info": {
            "module": "creative",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": str(image_path),
            "title": title,
            "text_placement": text_placement,
            "elements": elements,
            "step": step,
        },
        "decorations": {},
        "text": {},
        "output_images": {},
        "status": "pending",
        "total_elapsed_seconds": 0,
        "error": None,
    }

    # 输出路径
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 生成文件名
    img_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # 1. 加载图片
        logger.info("Loading image: %s", image_path)
        img = Image.open(image_path).convert("RGB")
        log_record["test_info"]["image_size"] = list(img.size)

        # 2. 初始化 NodeCreative
        config = load_config()
        node_creative = NodeCreative(config)

        # 3. 执行测试
        if step in ("all", "decorations"):
            # 装饰渲染测试
            logger.info("Testing decorations...")
            d_start = time.time()
            decorated = await node_creative._render_decorations(img, elements)
            d_elapsed = time.time() - d_start

            log_record["decorations"] = {
                "elements": elements,
                "elapsed_seconds": round(d_elapsed, 2),
            }

            # 保存装饰结果
            decorated_path = output_path / f"{timestamp}_{img_name}_decorated.png"
            decorated.save(decorated_path)
            log_record["output_images"]["decorated"] = str(decorated_path)
            logger.info("Decorated saved: %s", decorated_path)

        if step in ("all", "text"):
            # 文字渲染测试
            if step == "text":
                decorated = img  # 文字测试用原图
            logger.info("Testing text rendering...")
            t_start = time.time()

            # 凝练标题
            clean_title = node_creative._extract_title({"title": title})
            rendered = node_creative._render_text(decorated, clean_title, text_placement)
            t_elapsed = time.time() - t_start

            log_record["text"] = {
                "original_title": title,
                "clean_title": clean_title,
                "position": text_placement.get("region", "bottom_center"),
                "max_width_ratio": text_placement.get("max_width_ratio", 0.8),
                "elapsed_seconds": round(t_elapsed, 2),
            }

            # 保存文字结果
            text_path = output_path / f"{timestamp}_{img_name}_with_text.png"
            rendered.save(text_path)
            log_record["output_images"]["with_text"] = str(text_path)
            logger.info("Text rendered saved: %s", text_path)

        log_record["status"] = "success"
        logger.info("Creative test completed: status=%s", log_record["status"])

    except Exception as e:
        log_record["status"] = "failed"
        log_record["error"] = str(e)
        logger.error("Creative test failed: %s", e)

    # 计算总耗时
    elapsed = time.time() - start_time
    log_record["total_elapsed_seconds"] = round(elapsed, 2)

    # 保存日志
    log_file = output_path / f"{timestamp}_{img_name}_creative_log.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_record, f, ensure_ascii=False, indent=2)

    logger.info("Log saved: %s", log_file)

    return log_record


def main():
    parser = argparse.ArgumentParser(description="Creative 创意渲染模块测试")
    parser.add_argument("-i", "--image", required=True, help="图像路径")
    parser.add_argument("-t", "--title", default="二手好物转让", help="标题文字")
    parser.add_argument("-de", "--decorations", default=None, help="装饰元素 JSON")
    parser.add_argument("-tp", "--text-placement", default=None, help="文字放置配置 JSON")
    parser.add_argument("-o", "--output", default="data/output/logs/creative", help="输出目录")
    parser.add_argument("--step", default="all", choices=["all", "decorations", "text"],
                       help="测试步骤: all/decorations/text")

    args = parser.parse_args()

    # 解析装饰元素
    elements = None
    if args.decorations:
        import json
        try:
            elements = json.loads(args.decorations)
        except:
            logger.warning("Invalid decorations JSON, using defaults")

    # 解析文字放置配置
    text_placement = None
    if args.text_placement:
        import json
        try:
            text_placement = json.loads(args.text_placement)
        except:
            logger.warning("Invalid text_placement JSON, using defaults")

    result = asyncio.run(test_creative_module(
        image_path=args.image,
        title=args.title,
        text_placement=text_placement,
        elements=elements,
        step=args.step,
        output_dir=args.output,
    ))

    # 输出摘要
    print("\n" + "=" * 60)
    print("Creative Test Summary")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Total Elapsed: {result['total_elapsed_seconds']}s")

    if result['decorations']:
        print(f"Decorations: {result['decorations'].get('elapsed_seconds', 'N/A')}s, {len(result['decorations'].get('elements', []))} elements")
    if result['text']:
        print(f"Text: {result['text'].get('elapsed_seconds', 'N/A')}s, title='{result['text'].get('clean_title', '')[:30]}...'")

    print(f"\nOutput images:")
    for key, path in result.get('output_images', {}).items():
        print(f"  {key}: {Path(path).name}")

    print(f"Log: data/output/logs/creative/")
    print("=" * 60)


if __name__ == "__main__":
    main()
