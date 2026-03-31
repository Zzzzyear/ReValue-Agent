"""
Re-Value Agent - 批量评估测试脚本
用于马在飞构建的 30-50 张测试集批量运行
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import ReValueEngine
from core.prompts import FALLBACK_JSON

logger = logging.getLogger(__name__)


class BatchEvaluator:
    """
    批量评估器

    功能:
    1. 遍历 data/input 目录下的所有图片
    2. 运行完整处理流程
    3. 将失败案例隔离到 data/intermediate/
    4. 生成评估报告
    """

    def __init__(self, config: dict, input_dir: str, output_dir: str, intermediate_dir: str):
        self.config = config
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.intermediate_dir = Path(intermediate_dir)

        # 创建目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        # 初始化引擎
        engine_config = config.get("concurrency", {})
        semaphore_limit = engine_config.get("semaphore_limit", 2)
        self.engine = ReValueEngine(config, semaphore_limit=semaphore_limit)

        # 统计
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "partial": 0,  # 部分成功
        }

        # 失败案例
        self.failures: List[Dict] = []

    def get_image_files(self) -> List[Path]:
        """获取所有图片文件"""
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        files = []

        for pattern in patterns:
            files.extend(self.input_dir.glob(pattern))

        # 也检查子目录
        for pattern in patterns:
            files.extend(self.input_dir.glob(f"**/{pattern}"))

        return sorted(set(files))

    async def process_single(
        self,
        image_path: Path,
        description: Optional[str] = None,
    ) -> Dict:
        """
        处理单张图片

        Args:
            image_path: 图片路径
            description: 商品描述 (如果为None，使用文件名)

        Returns:
            结果字典
        """
        start_time = time.time()

        try:
            # 加载图片
            image = Image.open(image_path).convert("RGB")

            # 描述
            if description is None:
                description = f"二手商品: {image_path.stem}"

            # 调用引擎
            ctx = await self.engine.process(image, description)

            elapsed = time.time() - start_time

            result = {
                "image_path": str(image_path),
                "description": description,
                "status": ctx.status.value,
                "elapsed_seconds": round(elapsed, 2),
                "subject": ctx.subject,
                "has_vlm_output": ctx.vlm_json_output is not None,
                "has_matted_image": ctx.matted_image_rgba is not None,
                "has_background": ctx.background_image is not None,
                "has_final_image": ctx.final_image is not None,
                "has_copy": ctx.final_copy is not None,
                "error": ctx.error_message,
            }

            # 保存结果
            if ctx.status.value == "completed":
                result["output_paths"] = ReValueEngine.save_output(
                    ctx,
                    output_dir=str(self.output_dir),
                    prefix=image_path.stem,
                )

                # 保存中间产物 (用于调试)
                if ctx.matted_image_rgba:
                    mat_path = self.intermediate_dir / f"{image_path.stem}_matted.png"
                    ctx.matted_image_rgba.save(mat_path)

                if ctx.background_image:
                    bg_path = self.intermediate_dir / f"{image_path.stem}_background.png"
                    ctx.background_image.save(bg_path)

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("Failed to process %s: %s", image_path, e)

            # 保存失败图片到 intermediate
            failed_copy = self.intermediate_dir / f"FAILED_{image_path.name}"
            if image_path.exists():
                import shutil
                shutil.copy(image_path, failed_copy)

            return {
                "image_path": str(image_path),
                "description": description,
                "status": "error",
                "elapsed_seconds": round(elapsed, 2),
                "error": str(e),
            }

    async def run(self) -> List[Dict]:
        """
        运行批量评估

        Returns:
            所有结果列表
        """
        files = self.get_image_files()
        self.stats["total"] = len(files)

        logger.info("Found %d images to process", len(files))

        if not files:
            print(f"No images found in {self.input_dir}")
            return []

        results = []

        for i, img_file in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Processing {img_file.name}...", end=" ", flush=True)

            result = await self.process_single(img_file)
            results.append(result)

            # 更新统计
            status = result["status"]
            if status == "completed":
                self.stats["success"] += 1
                print("✓")
            elif status == "failed":
                self.stats["failed"] += 1
                self.failures.append(result)
                print("✗")
            else:
                self.stats["partial"] += 1
                self.failures.append(result)
                print("⚠")

        return results

    def generate_report(self, results: List[Dict]) -> Dict:
        """
        生成评估报告

        Returns:
            报告字典
        """
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": self.stats["total"],
            "success": self.stats["success"],
            "failed": self.stats["failed"],
            "partial": self.stats["partial"],
            "success_rate": round(
                self.stats["success"] / self.stats["total"] * 100, 2
            ) if self.stats["total"] > 0 else 0,
            "failures": self.failures,
            "results": results,
        }

        # 保存报告
        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 50)
        print("EVALUATION REPORT")
        print("=" * 50)
        print(f"Total: {report['total_images']}")
        print(f"Success: {report['success']} ({report['success_rate']}%)")
        print(f"Failed: {report['failed']}")
        print(f"Partial: {report['partial']}")
        print(f"\nReport saved to: {report_path}")

        if self.failures:
            print("\nFailed cases:")
            for f in self.failures:
                print(f"  - {f['image_path']}: {f.get('error', 'Unknown error')}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Re-Value Agent Batch Evaluator")

    parser.add_argument(
        "--input",
        "-i",
        default="data/input",
        help="输入目录 (默认: data/input)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/output",
        help="输出目录 (默认: data/output)",
    )
    parser.add_argument(
        "--intermediate",
        "-m",
        default="data/intermediate",
        help="中间产物目录 (默认: data/intermediate)",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="配置文件路径",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="运行前清空中间产物目录",
    )

    args = parser.parse_args()

    # 日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 加载配置
    config_path = args.config
    if config_path is None:
        config_path = Path(__file__).parent.parent / "conf" / "config.yaml"

    import yaml
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        logger.warning("Config not found: %s, using defaults", config_path)
        config = {}

    # 清空中间目录
    if args.clean:
        intermediate_dir = Path(args.intermediate)
        if intermediate_dir.exists():
            import shutil
            for f in intermediate_dir.glob("*"):
                if f.is_file():
                    f.unlink()
            print(f"Cleaned intermediate directory: {intermediate_dir}")

    # 运行评估
    evaluator = BatchEvaluator(
        config=config,
        input_dir=args.input,
        output_dir=args.output,
        intermediate_dir=args.intermediate,
    )

    results = asyncio.run(evaluator.run())
    report = evaluator.generate_report(results)


if __name__ == "__main__":
    main()
