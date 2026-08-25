"""Interactive command-line demo: paste a URL, get a verdict + explanation.

Run: PYTHONPATH=src python scripts/demo_cli.py
(Train the model first if you haven't: PYTHONPATH=src python scripts/train_demo_model.py)
"""
from phishdriftbench.demo import model


def print_result(result: dict):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"URL:        {result['url']}")
    print(f"VERDICT:    {result['verdict']}  (confidence {result['confidence']*100:.1f}%)")
    print(bar)

    if result["squatting_reasons"]:
        print("Brand-jacking / squatting screen flagged:")
        for r in result["squatting_reasons"]:
            print(f"  - {r}")
        if result["squatting_flagged"]:
            print("  -> This alone was enough to call it PHISHING, without needing the ML layer.")
        print()

    print(f"ML layer phishing score: {result['ml_score']*100:.1f}%")
    print("Top contributing factors for this URL:")
    if result["top_reasons"]:
        for r in result["top_reasons"]:
            print(f"  - {r}")
    else:
        print("  (no single feature stood out strongly)")
    print(bar)


def main():
    try:
        m = model.load()
    except FileNotFoundError:
        print("No trained demo model found. Run this first:")
        print("  PYTHONPATH=src python scripts/train_demo_model.py")
        return

    print("Phishing URL checker (demo). Type a URL, or 'quit' to exit.")
    while True:
        try:
            url = input("\nURL> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not url or url.lower() in {"quit", "exit"}:
            break
        result = model.predict_and_explain(url, m)
        print_result(result)


if __name__ == "__main__":
    main()
