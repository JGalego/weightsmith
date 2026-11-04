"""Build the synthetic fixtures the examples load (same as `weightsmith fixtures`)."""

from pathlib import Path

from weightsmith.synthetic import build_all

if __name__ == "__main__":
    for name, accuracy in build_all(Path(__file__).parent / "fixtures").items():
        print(name, accuracy)
