"""
ReValue-Agent - Test 01: VLM 模块测试
=====================================

测试 VLM 图像分析模块，记录完整输入输出。

## 运行命令

```bash
cd D:/Code/AI_Playground/projects/ReValue-Agent
venv\Scripts\activate

# 基本用法
python tests/test_01_vlm.py -i data/input/xianyu_bag_01.jpg -t "MiniKelly二手包"

# 指定输出目录
python tests/test_01_vlm.py -i data/input/xianyu_bag_01.jpg -t "MiniKelly二手包" -o data/output/logs/vlm
```

## 输出文件

在 data/output/logs/vlm/ 目录下生成:
- {timestamp}_{imagename}_vlm_log.json - 完整日志

## 日志内容

{
  "test_info": {
    "module": "vlm",
    "timestamp": "...",
    "image_path": "...",
    "image_size": [width, height],
    "text": "..."
  },
  "prompt": {
    "system": "完整 system prompt",
    "user": "完整 user prompt"
  },
  "api_response": {
    "raw": "API 原始响应文本",
    "parsed": { 解析后的 JSON }
  },
  "context_output": {
    "subject": "...",
    "selling_points": [...],
    "background_prompt": "...",
    "reference_box": [...],
    "creative_elements": [...],
    "text_placement": {...}
  },
  "status": "success/failed",
  "elapsed_seconds": 12.5,
  "error": "错误信息（如果有）"
}
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.context import Context, TaskStatus
from core.node_router import NodeRouter
from core.prompts import get_vlm_prompts, parse_json_from_response

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


def image_to_base64(image: Image.Image) -> str:
    """将 PIL Image 转换为 base64"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode()


async def test_vlm_module(
    image_path: str,
    text: str,
    output_dir: str = "data/output/logs/vlm",
) -> dict:
    """
    测试 VLM 模块

    Args:
        image_path: 图片路径
        text: 商品描述
        output_dir: 输出目录

    Returns:
        测试结果字典
    """
    start_time = time.time()

    # 记录完整输入输出
    log_record = {
        "test_info": {
            "module": "vlm",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": str(image_path),
            "text": text,
        },
        "prompt": {
            "system": "",
            "user": "",
        },
        "api_response": {
            "raw": "",
            "parsed": {},
        },
        "context_output": {},
        "status": "pending",
        "elapsed_seconds": 0,
        "error": None,
    }

    try:
        # 1. 加载图片
        logger.info("Loading image: %s", image_path)
        img = Image.open(image_path).convert("RGB")
        log_record["test_info"]["image_size"] = list(img.size)

        # 2. 获取 Prompt（记录完整内容）
        prompts = get_vlm_prompts(text)
        log_record["prompt"]["system"] = prompts["system"]
        log_record["prompt"]["user"] = prompts["user"]
        logger.info("Prompts loaded: system=%d chars, user=%d chars",
                    len(prompts["system"]), len(prompts["user"]))

        # 3. 初始化 NodeRouter
        config = load_config()
        node_router = NodeRouter(config)

        # 4. 调用 VLM API（直接调用，记录原始响应）
        logger.info("Calling VLM API...")
        image_base64 = image_to_base64(img)

        # 获取 API client 以便直接调用
        api_client = node_router.get_api_client()

        # 记录原始 API 响应
        raw_response = await api_client.chat_with_image(
            image_base64=image_base64,
            system_prompt=prompts["system"],
            user_prompt=prompts["user"],
        )
        log_record["api_response"]["raw"] = raw_response
        logger.info("API response received: %d chars", len(raw_response))

        # 5. 解析响应
        parsed_result = parse_json_from_response(raw_response)
        log_record["api_response"]["parsed"] = parsed_result
        logger.info("Parsed result: subject=%s", parsed_result.get("subject"))

        # 6. 创建 Context 并处理（用于获取完整 context 输出）
        ctx = Context(
            original_image=img,
            original_text=text,
            status=TaskStatus.RUNNING,
        )
        ctx = await node_router.process(ctx)

        # 记录 Context 输出
        log_record["context_output"] = {
            "subject": ctx.subject,
            "selling_points": ctx.selling_points,
            "background_prompt": ctx.background_prompt,
            "reference_box": ctx.reference_box,
            "creative_elements": ctx.creative_elements,
            "text_placement": ctx.text_placement,
            "vlm_json_output": ctx.vlm_json_output,
        }

        log_record["status"] = "success" if ctx.subject != "商品" else "partial_success"
        logger.info("VLM test completed: subject=%s, status=%s",
                    ctx.subject, log_record["status"])

    except Exception as e:
        log_record["status"] = "failed"
        log_record["error"] = str(e)
        logger.error("VLM test failed: %s", e)

    # 计算耗时
    elapsed = time.time() - start_time
    log_record["elapsed_seconds"] = round(elapsed, 2)

    # 保存日志
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 生成文件名
    img_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"{timestamp}_{img_name}_vlm_log.json"

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_record, f, ensure_ascii=False, indent=2)

    logger.info("Log saved: %s", log_file)

    return log_record


def main():
    parser = argparse.ArgumentParser(description="VLM 模块测试")
    parser.add_argument("-i", "--image", required=True, help="图片路径")
    parser.add_argument("-t", "--text", required=True, help="商品描述")
    parser.add_argument("-o", "--output", default="data/output/logs/vlm", help="输出目录")

    args = parser.parse_args()

    result = asyncio.run(test_vlm_module(
        image_path=args.image,
        text=args.text,
        output_dir=args.output,
    ))

    # 输出摘要
    print("\n" + "=" * 60)
    print("VLM Test Summary")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Elapsed: {result['elapsed_seconds']}s")
    if result['context_output']:
        print(f"Subject: {result['context_output'].get('subject', 'N/A')}")
        print(f"Selling Points: {result['context_output'].get('selling_points', [])[:2]}...")
    print(f"Log File: data/output/logs/vlm/")
    print("=" * 60)


if __name__ == "__main__":
    main()
