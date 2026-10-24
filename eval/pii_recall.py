"""Measure how much planted PII the local detectors catch (for the README's honest numbers).

    python eval/pii_recall.py

Uses samples/sensitive/ (synthetic data only). A planted value counts as caught if it no longer
appears anywhere in the redacted text.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.privacy.redactor import Redactor  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sensitive"


def main() -> None:
    text = (SAMPLE / "rental_agreement_synthetic.txt").read_text(encoding="utf-8")
    planted: dict[str, list[str]] = json.loads((SAMPLE / "planted_pii.json").read_text(encoding="utf-8"))
    for use_ner in (False, True):
        redactor = Redactor()
        redactor.learn(text, use_ner=use_ner)
        redacted = redactor.redact(text, use_ner=False)
        print(f"\nNER {'on' if use_ner else 'off'}:")
        print("| type | planted | caught |")
        print("|---|---|---|")
        for pii_type, values in planted.items():
            caught = sum(v not in redacted for v in values)
            print(f"| {pii_type} | {len(values)} | {caught} |")
        found_names = sorted(v for _, (t, v) in redactor.values.items() if t in {"PERSON", "ORG", "LOCATION"})
        if use_ner:
            print(f"names/orgs/places hidden: {found_names}")


if __name__ == "__main__":
    main()
