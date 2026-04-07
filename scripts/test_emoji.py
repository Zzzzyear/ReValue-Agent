#!/usr/bin/env python3
"""测试 emoji 贴纸渲染效果"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from tools.decorations import DecorationRenderer

def main():
    print("Starting emoji sticker test...")

    os.makedirs("data/output", exist_ok=True)
    test_img = Image.new("RGB", (800, 800), (240, 240, 240))
    renderer = DecorationRenderer()

    elements = [
        {"emoji": "✨", "position": "top-right", "count": 2, "size": "medium"},
        {"emoji": "🔥", "position": "top-left", "count": 1, "size": "large"},
        {"emoji": "💰", "position": "top-center", "count": 1, "size": "medium"},
        {"emoji": "💎", "position": "center", "count": 1, "size": "large"},
    ]

    print(f"Rendering {len(elements)} emoji elements...")
    result = renderer.render_elements(test_img, elements)
    out_path = "data/output/test_emoji_stickers.png"
    result.save(out_path)
    print(f"Saved: {out_path} ({result.size})")

    # backward compat
    elements_old = [
        {"type": "sparkle", "position": "top-right", "count": 3, "size": "small"},
    ]
    result2 = renderer.render_elements(test_img, elements_old)
    out_path2 = "data/output/test_old_type_compat.png"
    result2.save(out_path2)
    print(f"Saved: {out_path2} ({result2.size})")

    print("All tests passed!")

if __name__ == "__main__":
    main()
