"""Reproduce manuscript aggregate calculations without loading private records."""
from decimal import Decimal, ROUND_HALF_UP
from math import sqrt


def percent(count, total):
    return (Decimal(count) * 100 / Decimal(total)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )


def wilson(count, total):
    z = 1.959963984540054
    p = count / total
    divisor = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / divisor
    radius = z * sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / divisor
    return 100 * (centre - radius), 100 * (centre + radius)


if __name__ == "__main__":
    print("160-case Top-1 classification")
    for model, correct in [("PanDerm", 90), ("SkinGPT-R1", 81)]:
        print(f"{model}: {correct}/160, {percent(correct, 160)}%")
    print(f"Difference: 9 cases, {percent(9, 160)} percentage points")
    assert percent(90, 160) == Decimal("56.25")
    assert percent(81, 160) == Decimal("50.63")
    assert percent(9, 160) == Decimal("5.63")

    print("\nSingle-clinician preferences; 158 completed cases")
    counts = {
        "Accuracy": 72,
        "Safety": 88,
        "Medical Groundedness": 91,
        "Clinical Coverage": 114,
        "Reasoning Coherence": 103,
        "Description Precision": 77,
    }
    for dimension, count in counts.items():
        lower, upper = wilson(count, 158)
        print(
            f"{dimension}: R1 {count}/158, {percent(count, 158)}%; "
            f"SkinGPT-4 {158-count}/158, {percent(158-count, 158)}%; "
            f"R1 Wilson 95% CI {lower:.2f} to {upper:.2f}%"
        )
