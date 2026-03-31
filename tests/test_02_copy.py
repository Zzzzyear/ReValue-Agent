"""
ReValue-Agent - Test 02: Copy 文案生成模块测试
=============================================

测试文案生成模块，记录完整输入输出。

## 运行命令

```bash
cd D:/Code/AI_Playground/projects/ReValue-Agent
venv\Scripts\activate

# 基本用法
python tests/test_02_copy.py -s "凯莉包" -p "09梦幻紫|MiniKelly|95新" -o "二手MiniKelly包"

# 指定平台
python tests/test_02_copy.py -s "凯莉包" -p "09梦幻紫|MiniKelly|95新" -o "二手MiniKelly" -pt "小红书体"

# 指定输出目录
python tests/test_02_copy.py -s "凯莉包" -p "09梦幻紫|MiniKelly|95新" -o "二手MiniKelly" -o data/output/logs/copy
```

## 输出文件

在 data/output/logs/copy/ 目录下生成:
- {timestamp}_{subject}_copy_log.json - 完整日志

## 日志内容

{
  "test_info": {
    "module": "copy",
    "timestamp": "...",
    "subject": "...",
    "selling_points": [...],
    "original_text": "...",
    "platform": "闲鱼体/小红书体"
  },
  "prompt": {
    "system": "完整 system prompt",
    "user": "完整 user prompt"
  },
  "api_response": {
    "raw": "API 原始响应文本",
    "parsed": { title, content }
  },
  "copy_output": {
    "title": "...",
    "content": "..."
  },
  "status": "success/failed",
  "elapsed_seconds": 5.2,
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

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.context import Context
from core.node_text import NodeText
from core.prompts import get_copy_prompts

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


async def test_copy_module(
    subject: str,
    selling_points: str,
    original_text: str,
    platform: str = "闲鱼体",
    output_dir: str = "data/output/logs/copy",
) -> dict:
    """
    测试 Copy 文案生成模块

    Args:
        subject: 商品名称
        selling_points: 卖点（用 | 分隔）
        original_text: 原始描述
        platform: 平台风格
        output_dir: 输出目录

    Returns:
        测试结果字典
    """
    start_time = time.time()

    # 解析卖点列表
    sp_list = [s.strip() for s in selling_points.split('|') if s.strip()]
    if not sp_list:
        sp_list = ["二手好物", "超低价", "实用性强"]

    # 记录完整输入输出
    log_record = {
        "test_info": {
            "module": "copy",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "subject": subject,
            "selling_points": sp_list,
            "original_text": original_text,
            "platform": platform,
        },
        "prompt": {
            "system": "",
            "user": "",
        },
        "api_response": {
            "raw": "",
            "parsed": {},
        },
        "copy_output": {},
        "status": "pending",
        "elapsed_seconds": 0,
        "error": None,
    }

    try:
        # 1. 获取 Prompt（记录完整内容）
        prompts = get_copy_prompts(
            subject=subject,
            selling_points=sp_list,
            original_text=original_text,
            platform_style=platform,
        )
        log_record["prompt"]["system"] = prompts["system"]
        log_record["prompt"]["user"] = prompts["user"]
        logger.info("Prompts loaded: system=%d chars, user=%d chars",
                    len(prompts["system"]), len(prompts["user"]))

        # 2. 初始化 NodeText
        config = load_config()
        node_text = NodeText(config)

        # 3. 直接调用 LLM API（记录原始响应）
        logger.info("Calling LLM API...")
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ]
        raw_response = await node_text.llm_client.chat(messages)
        log_record["api_response"]["raw"] = raw_response
        logger.info("API response received: %d chars", len(raw_response))

        # 4. 解析响应
        parsed_copy = node_text._parse_copy_response(raw_response)
        log_record["api_response"]["parsed"] = parsed_copy
        log_record["copy_output"] = parsed_copy
        logger.info("Parsed copy: title=%s", parsed_copy.get("title", "N/A")[:30])

        # 5. 完整流程处理
        ctx = Context(
            subject=subject,
            selling_points=sp_list,
            original_text=original_text,
            platform_style=platform,
        )
        ctx = await node_text.process(ctx)

        log_record["copy_output"] = ctx.final_copy or parsed_copy
        log_record["status"] = "success"

    except Exception as e:
        log_record["status"] = "failed"
        log_record["error"] = str(e)
        logger.error("Copy test failed: %s", e)

    # 计算耗时
    elapsed = time.time() - start_time
    log_record["elapsed_seconds"] = round(elapsed, 2)

    # 保存日志
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 生成文件名
    safe_subject = re.sub(r'[^\w\u4e00-\u9fff]', '_', subject)[:20]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"{timestamp}_{safe_subject}_copy_log.json"

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_record, f, ensure_ascii=False, indent=2)

    logger.info("Log saved: %s", log_file)

    return log_record


def main():
    parser = argparse.ArgumentParser(description="Copy 文案生成模块测试")
    parser.add_argument("-s", "--subject", required=True, help="商品名称")
    parser.add_argument("-p", "--selling-points", required=True,
                        help="卖点（用 | 分隔），例如: 09紫|MiniKelly|95新")
    parser.add_argument("-o", "--original-text", required=True, help="原始描述")
    parser.add_argument("-pt", "--platform", default="闲鱼体",
                        choices=["闲鱼体", "小红书体"], help="平台风格")
    parser.add_argument("-out", "--output", default="data/output/logs/copy", help="输出目录")

    args = parser.parse_args()

    result = asyncio.run(test_copy_module(
        subject=args.subject,
        selling_points=args.selling_points,
        original_text=args.original_text,
        platform=args.platform,
        output_dir=args.output,
    ))

    # 输出摘要
    print("\n" + "=" * 60)
    print("Copy Test Summary")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Elapsed: {result['elapsed_seconds']}s")
    if result['copy_output']:
        print(f"Title: {result['copy_output'].get('title', 'N/A')}")
    print(f"Log File: {result['test_info']['module']}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
