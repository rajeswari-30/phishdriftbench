"""One-time training of the interactive demo's model. Run this once (or
whenever you want to retrain on more data); the CLI and web app both just
load the saved result.

Run: PYTHONPATH=src python scripts/train_demo_model.py
"""
from phishdriftbench.demo import model

if __name__ == "__main__":
    model.train_and_save()
