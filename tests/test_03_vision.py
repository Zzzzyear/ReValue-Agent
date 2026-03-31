"""
ReValue-Agent - Test 03: Vision 视觉处理模块测试
================================================

测试视觉处理模块（抠图、SD背景生成、图像融合），记录完整输入输出。

## 运行命令

```bash
cd D:/Code/AI_Playground/projects/ReValue-Agent
venv\Scripts\activate

# 基本用法
python tests/test_03_vision.py -i data/input/xianyu_bag_01.jpg

# 指定背景 prompt
python tests/test_03_vision.py -i data/input/xianyu_bag_01.jpg -bp "luxury boutique interior"

# 仅测试抠图
python tests/test_03_vision.py -i data/input/xianyu_bag_01.jpg --step matting

# 仅测试 SD 背景
python tests/test_03_vision.py -i data/input/xianyu_bag_01.jpg --step sd -bp "luxury boutique"

# 仅测试融合
python tests/test_03_vision.py -i data/input/xianyu_bag_01.jpg --step fusion

# 指定输出目录
python tests/test_03_vision.py -i data/input/xianyu_bag_01.jpg -o data/output/logs/vision
```

## 输出文件

在 data/output/logs/vision/ 目录下生成:
- {timestamp}_{imagename}_vision_log.json - 完整日志
- {timestamp}_{imagename}_matted.png - 抠图结果
- {timestamp}_{imagename}_background.png - SD 背景结果
- {timestamp}_{imagename}_fused.png - 融合结果

## 日志内容

{
  "test_info": {
    "module": "vision",
    "timestamp": "...",
    "image_path": "...",
    "image_size": [width, height],
    "background_prompt": "...",
    "step": "all/matting/sd/fusion"
  },
  "steps": {
    "matting": {
      "input_size": [...],
      "output_size": [...],
      "elapsed_seconds": 2.5
    },
    "sd": {
      "prompt": "...",
      "negative_prompt": "...",
      "elapsed_seconds": 30.0
    },
    "fusion": {
      "reference_box": [...],
      "elapsed_seconds": 1.0
    }
  },
  "output_images": {
    "matted": "path/to/matted.png",
    "background": "path/to/background.png",
    "fused": "path/to/fused.png"
  },
  "status": "success/failed",
  "total_elapsed_seconds": 35.0,
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

from core.context import Context
from core.node_vision import NodeVision
from core.prompts import get_sd_prompt_suffix

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


async def test_vision_module(
    image_path: str,
    background_prompt: str = None,
    step: str = "all",
    output_dir: str = "data/output/logs/vision",
) -> dict:
    """
    测试 Vision 视觉处理模块

    Args:
        image_path: 图片路径
        background_prompt: SD 背景 prompt（可选）
        step: 测试步骤 all/matting/sd/fusion
        output_dir: 输出目录

    Returns:
        测试结果字典
    """
    start_time = time.time()

    # 默认背景 prompt
    if not background_prompt:
        background_prompt = "luxury boutique interior, soft pink and grey gradient background, elegant spotlighting, high-end fashion ambiance, 8k, professional product photography"

    # 记录完整输入输出
    log_record = {
        "test_info": {
            "module": "vision",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": str(image_path),
            "background_prompt": background_prompt,
            "step": step,
        },
        "steps": {
            "matting": {},
            "sd": {},
            "fusion": {},
        },
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

        # 2. 初始化 NodeVision
        config = load_config()
        node_vision = NodeVision(config)

        # 3. 执行测试
        if step in ("all", "matting"):
            # Matting 测试
            logger.info("Testing matting...")
            m_start = time.time()
            matted = await node_vision.process_matting_only(img)
            m_elapsed = time.time() - m_start

            log_record["steps"]["matting"] = {
                "input_size": list(img.size),
                "output_size": list(matted.size),
                "elapsed_seconds": round(m_elapsed, 2),
            }

            # 保存抠图结果
            matted_path = output_path / f"{timestamp}_{img_name}_matted.png"
            matted.save(matted_path)
            log_record["output_images"]["matted"] = str(matted_path)
            logger.info("Matting saved: %s", matted_path)

        if step in ("all", "sd"):
            # SD 背景生成测试
            logger.info("Testing SD background generation...")
            sd_start = time.time()
            sd_prompt = background_prompt + get_sd_prompt_suffix()
            bg = await node_vision.process_background_only(sd_prompt)
            sd_elapsed = time.time() - sd_start

            log_record["steps"]["sd"] = {
                "prompt": sd_prompt,
                "negative_prompt": "product on ground, messy background, low quality",
                "elapsed_seconds": round(sd_elapsed, 2),
            }

            # 保存背景结果
            bg_path = output_path / f"{timestamp}_{img_name}_background.png"
            bg.save(bg_path)
            log_record["output_images"]["background"] = str(bg_path)
            logger.info("Background saved: %s", bg_path)

        if step == "fusion":
            # 单独测试融合需要手动加载 matted 和 bg
            logger.info("Fusion test requires all steps, use --step all")

        if step == "all" and log_record["steps"].get("matting") and log_record["steps"].get("sd"):
            # Fusion 测试（需要 matted 和 bg）
            logger.info("Testing fusion...")

            # 如果还没生成 matted 和 bg，先生成
            if "matted" not in log_record["output_images"]:
                matted = await node_vision.process_matting_only(img)
            if "background" not in log_record["output_images"]:
                sd_prompt = background_prompt + get_sd_prompt_suffix()
                bg = await node_vision.process_background_only(sd_prompt)

            # 加载已保存的图像
            matted = Image.open(log_record["output_images"]["matted"])
            bg = Image.open(log_record["output_images"]["background"])

            # 执行融合
            f_start = time.time()
            fused = node_vision._process_fusion(
                subject_rgba=matted,
                background=bg,
                reference_box=None,
            )
            f_elapsed = time.time() - f_start

            log_record["steps"]["fusion"] = {
                "reference_box": None,
                "elapsed_seconds": round(f_elapsed, 2),
            }

            # 保存融合结果
            fused_path = output_path / f"{timestamp}_{img_name}_fused.png"
            fused.save(fused_path)
            log_record["output_images"]["fused"] = str(fused_path)
            logger.info("Fusion saved: %s", fused_path)

        log_record["status"] = "success"
        logger.info("Vision test completed: status=%s", log_record["status"])

    except Exception as e:
        log_record["status"] = "failed"
        log_record["error"] = str(e)
        logger.error("Vision test failed: %s", e)

    # 计算总耗时
    elapsed = time.time() - start_time
    log_record["total_elapsed_seconds"] = round(elapsed, 2)

    # 保存日志
    log_file = output_path / f"{timestamp}_{img_name}_vision_log.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_record, f, ensure_ascii=False, indent=2)

    logger.info("Log saved: %s", log_file)

    return log_record


def main():
    parser = argparse.ArgumentParser(description="Vision 视觉处理模块测试")
    parser.add_argument("-i", "--image", required=True, help="图片路径")
    parser.add_argument("-bp", "--background-prompt", default=None, help="SD 背景 prompt")
    parser.add_argument("-o", "--output", default="data/output/logs/vision", help="输出目录")
    parser.add_argument("--step", default="all", choices=["all", "matting", "sd", "fusion"],
                       help="测试步骤: all/matting/sd/fusion")

    args = parser.parse_args()

    result = asyncio.run(test_vision_module(
        image_path=args.image,
        background_prompt=args.background_prompt,
        step=args.step,
        output_dir=args.output,
    ))

    # 输出摘要
    print("\n" + "=" * 60)
    print("Vision Test Summary")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Total Elapsed: {result['total_elapsed_seconds']}s")

    if result['steps'].get('matting'):
        print(f"Matting: {result['steps']['matting'].get('elapsed_seconds', 'N/A')}s")
    if result['steps'].get('sd'):
        print(f"SD: {result['steps']['sd'].get('elapsed_seconds', 'N/A')}s")
    if result['steps'].get('fusion'):
        print(f"Fusion: {result['steps']['fusion'].get('elapsed_seconds', 'N/A')}s")

    print(f"\nOutput images:")
    for key, path in result.get('output_images', {}).items():
        print(f"  {key}: {Path(path).name}")

    print(f"Log: data/output/logs/vision/")
    print("=" * 60)


if __name__ == "__main__":
    main()
