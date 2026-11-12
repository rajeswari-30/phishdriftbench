"""Vercel entry point for the PhishLens API (a Python Serverless Function).

vercel.json sends every /api/* request here; the Next.js site serves
everything else from the same project and domain.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishdriftbench.server import app  # noqa: E402,F401
