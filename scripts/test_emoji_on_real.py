#!/usr/bin/env python3
"""在真实产品图上测试 emoji 贴纸效果"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from tools.decorations import DecorationRenderer

def main():
    candidates = [
        "data/output/test_bag/xianyu_bag_01_1775554347_final.png",
        "data/input/xianyu_bag_01.jpg",
    ]
    src = None
    for c in candidates:
        if os.path.exists(c):
            src = c
            break
    if src is None:
        print("No test image found, using white canvas")
        img = Image.new("RGB", (1024, 1024), (245, 245, 245))
    else:
        img = Image.open(src).convert("RGB")
        print(f"Using: {src}  size={img.size}")

    renderer = DecorationRenderer()

    elements = [
        {"emoji": "✨", "position": "top-right", "count": 2, "size": "small"},
        {"emoji": "💎", "position": "top-left", "count": 1, "size": "large"},
        {"emoji": "🔥", "position": "top-center", "count": 1, "size": "medium"},
    ]

    print(f"Rendering {len(elements)} emoji stickers...")
    result = renderer.render_elements(img, elements)

    out = "data/output/test_emoji_on_real.png"
    result.save(out)
    print(f"Saved: {out}  size={result.size}")

if __name__ == "__main__":
    main()
