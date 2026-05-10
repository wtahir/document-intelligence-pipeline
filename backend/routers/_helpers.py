"""
shared helper — loads any JSON file from data/output.
"""

import json
import os
from datetime import datetime
from typing import Optional

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(_root, "data", "output")
PDF_DIR    = os.path.join(_root, "data", "pdfs")


def load_json(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_mod_time(filename: str) -> Optional[str]:
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return None
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def pdf_count() -> int:
    if not os.path.isdir(PDF_DIR):
        return 0
    return len([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
