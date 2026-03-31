"""
Re-Value Agent - Context 数据类
在节点间传递的上下文数据
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from PIL import Image


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Context:
    """在节点间传递的上下文数据"""
    # 输入
    original_image: Optional[Image.Image] = None
    original_image_path: Optional[str] = None
    original_text: str = ""

    # Node 1 输出
    vlm_json_output: Optional[Dict[str, Any]] = None
    subject: str = ""
    selling_points: List[str] = field(default_factory=list)
    background_prompt: str = ""
    reference_box: Optional[List[float]] = None  # [x_min, y_min, x_max, y_max]
    creative_elements: Optional[List[Dict[str, Any]]] = None  # VLM生成的装饰元素
    text_placement: Optional[Dict[str, Any]] = None  # VLM建议的文字位置

    # Node 2A 输出 (视觉流)
    matted_image_rgba: Optional[Image.Image] = None
    background_image: Optional[Image.Image] = None
    final_image: Optional[Image.Image] = None

    # Node 2B 输出 (文本流)
    final_copy: Optional[Dict[str, str]] = None  # {title, content}

    # Node 2C 输出 (创意合成)
    decorated_image: Optional[Image.Image] = None  # 带文字和装饰的最终图

    # 状态
    status: TaskStatus = TaskStatus.PENDING
    error_message: Optional[str] = None

    # 可配置选项
    platform_style: str = "闲鱼体"  # 闲鱼体 / 小红书体

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于序列化"""
        return {
            "subject": self.subject,
            "selling_points": self.selling_points,
            "background_prompt": self.background_prompt,
            "reference_box": self.reference_box,
            "creative_elements": self.creative_elements,
            "text_placement": self.text_placement,
            "final_copy": self.final_copy,
            "status": self.status.value,
            "error_message": self.error_message,
        }
