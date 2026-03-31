#!/usr/bin/env python3
"""
Re-Value Agent - CLI 入口
支持单图处理和批量处理
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

from PIL import Image

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent))

from core.engine import ReValueEngine
from tools.sd_local import check_cuda_available, get_gpu_memory_info

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """加载配置"""
    import yaml

    if config_path is None:
        config_path = Path(__file__).parent / "conf" / "config.yaml"

    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    logger.warning("Config file not found: %s, using defaults", config_path)
    return {}


async def process_single(
    engine: ReValueEngine,
    image_path: str,
    text: str,
    output_dir: str = None,
    verbose: bool = False,
) -> dict:
    """
    处理单张图片

    Args:
        engine: ReValueEngine 实例
        image_path: 图片路径
        text: 商品描述
        output_dir: 输出目录
        verbose: 是否输出详细信息

    Returns:
        结果字典
    """
    start_time = time.time()

    try:
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        logger.info("Processing: %s", image_path)
        logger.info("Description: %s", text)

        # 处理
        ctx = await engine.process(image, text)

        elapsed = time.time() - start_time

        result = {
            "image_path": image_path,
            "text": text,
            "status": ctx.status.value,
            "elapsed_seconds": round(elapsed, 2),
            "subject": ctx.subject,
            "error": ctx.error_message,
        }

        # 保存结果
        if output_dir and ctx.status.value == "completed":
            result_paths = ReValueEngine.save_output(
                ctx,
                output_dir=output_dir,
                prefix=Path(image_path).stem,
            )
            result["output_paths"] = result_paths

        if verbose:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"[OK] {image_path} - {ctx.status.value} ({elapsed:.1f}s)")

            if ctx.final_copy:
                print(f"  Title: {ctx.final_copy.get('title', 'N/A')}")

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Failed to process %s: %s", image_path, e)
        return {
            "image_path": image_path,
            "text": text,
            "status": "failed",
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e),
        }


async def process_batch(
    engine: ReValueEngine,
    input_dir: str,
    output_dir: str,
    pattern: str = "*.jpg",
    verbose: bool = False,
) -> List[dict]:
    """
    批量处理

    Args:
        engine: ReValueEngine 实例
        input_dir: 输入目录
        output_dir: 输出目录
        pattern: 文件匹配模式
        verbose: 是否输出详细信息

    Returns:
        结果列表
    """
    input_path = Path(input_dir)
    image_files = list(input_path.glob(pattern))

    if not image_files:
        image_files = list(input_path.glob("*.png"))
    if not image_files:
        image_files = list(input_path.glob("*.jpeg"))

    logger.info("Found %d images in %s", len(image_files), input_dir)

    if not image_files:
        print(f"No images found in {input_dir}")
        return []

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 处理 (顺序执行，避免并发压力)
    results = []
    for i, img_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] ", end="", flush=True)

        # 使用文件名作为默认描述
        text = f"二手商品，{img_file.stem}"
        result = await process_single(
            engine=engine,
            image_path=str(img_file),
            text=text,
            output_dir=output_dir,
            verbose=False,
        )
        results.append(result)

    # 统计
    success = sum(1 for r in results if r["status"] == "completed")
    failed = len(results) - success

    print("\n" + "=" * 50)
    print(f"Batch complete: {success} succeeded, {failed} failed")

    # 保存汇总
    summary_path = Path(output_dir) / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Re-Value Agent - 二手商品图文创意自动化"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (默认: conf/config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # 单图处理
    single_parser = subparsers.add_parser("single", help="处理单张图片")
    single_parser.add_argument("--image", "-i", required=True, help="图片路径")
    single_parser.add_argument("--text", "-t", required=True, help="商品描述")
    single_parser.add_argument("--output", "-o", default="data/output", help="输出目录")
    single_parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    # 批量处理
    batch_parser = subparsers.add_parser("batch", help="批量处理")
    batch_parser.add_argument(
        "--input", default="data/input", help="输入目录 (默认: data/input)"
    )
    batch_parser.add_argument(
        "--output", "-o", default="data/output", help="输出目录 (默认: data/output)"
    )
    batch_parser.add_argument(
        "--pattern", default="*.jpg", help="文件匹配模式 (默认: *.jpg)"
    )
    batch_parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    # Gradio web
    web_parser = subparsers.add_parser("web", help="启动 Gradio Web 界面")
    web_parser.add_argument(
        "--port", "-p", type=int, default=7860, help="端口 (默认: 7860)"
    )
    web_parser.add_argument(
        "--host", default="0.0.0.0", help="主机 (默认: 0.0.0.0)"
    )

    # 系统检查
    subparsers.add_parser("check", help="检查系统环境")

    args = parser.parse_args()

    # 日志配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "check":
        # 系统检查
        print("=" * 50)
        print("Re-Value Agent - System Check")
        print("=" * 50)

        # Python
        print(f"Python: {sys.version}")

        # CUDA
        has_cuda = check_cuda_available()
        print(f"CUDA: {'[OK]' if has_cuda else '[FAIL]'}")

        if has_cuda:
            gpu_info = get_gpu_memory_info()
            print(f"GPU Memory: {gpu_info['total']:.1f} GB total, {gpu_info['allocated']:.1f} GB allocated")

        # 配置文件
        config = load_config(args.config)
        print(f"Config: {'[OK]' if config else '[FAIL]'}")

        if config:
            print(f"VLM Model: {config.get('vlm', {}).get('default_model', 'N/A')}")
            print(f"SD Model: {config.get('stable_diffusion', {}).get('model_type', 'N/A')}")

        # 目录
        print("\nDirectories:")
        for name in ["data/input", "data/intermediate", "data/output"]:
            path = Path(__file__).parent / name
            exists = "[OK]" if path.exists() else "[FAIL]"
            print(f"  {exists} {path}")

        return

    if args.command == "single":
        config = load_config(args.config)
        engine = ReValueEngine(config)

        result = asyncio.run(process_single(
            engine=engine,
            image_path=args.image,
            text=args.text,
            output_dir=args.output,
            verbose=args.verbose,
        ))

        sys.exit(0 if result["status"] == "completed" else 1)

    if args.command == "batch":
        config = load_config(args.config)
        engine = ReValueEngine(config)

        results = asyncio.run(process_batch(
            engine=engine,
            input_dir=args.input,
            output_dir=args.output,
            pattern=args.pattern,
            verbose=args.verbose,
        ))

        success = sum(1 for r in results if r["status"] == "completed")
        sys.exit(0 if success == len(results) else 1)

    if args.command == "web":
        from web.gradio_app import ReValueGradioApp

        config = load_config(args.config)
        app = ReValueGradioApp(config)
        app.launch(server_name=args.host, server_port=args.port)
        return

    # 默认显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()
