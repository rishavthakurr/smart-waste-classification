"""Centralised disposal guidance and the (software-only) smart-bin simulation."""
from __future__ import annotations

from src.utils import class_names

DISCLAIMER = (
    "Guidance is general. Actual recycling and disposal requirements depend on your "
    "local facilities and municipal regulations."
)

RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "plastic": {
        "action": "Check your local plastic recycling rules. Empty and rinse recyclable containers, "
                  "and keep them separate from heavily contaminated plastic.",
        "tip": "Not every plastic type is accepted everywhere; look for local resin-code guidance.",
    },
    "paper": {
        "action": "Keep recyclable paper clean and dry. Where local rules require it, leave out "
                  "heavily soiled or greasy paper.",
        "tip": "Wet or food-stained paper often cannot be recycled.",
    },
    "cardboard": {
        "action": "Flatten clean cardboard and keep it dry. Separate it from wet or heavily "
                  "contaminated waste.",
        "tip": "Greasy items such as pizza boxes may be refused by some facilities.",
    },
    "glass": {
        "action": "Use a designated glass collection point or bin where available, and handle "
                  "broken glass carefully.",
        "tip": "Some programmes sort glass by colour or exclude ceramics and mirrors.",
    },
    "metal": {
        "action": "Separate recyclable metal containers (empty and rinsed) and follow your local "
                  "collection rules.",
        "tip": "Aerosol cans and mixed-material items may have special rules.",
    },
    "general_waste": {
        "action": "Check your local municipal instructions. Items here are generally not "
                  "recyclable through standard streams.",
        "tip": "Hazardous items (batteries, electronics, chemicals) usually need special drop-off.",
    },
}

# Simulated bin scheme (illustrative only; no physical mechanism exists in this project).
SIMULATED_BINS: dict[str, dict[str, str]] = {
    "plastic": {"bin": "Bin 1 - Plastics", "color": "#F2B705"},
    "paper": {"bin": "Bin 2 - Paper", "color": "#2F80ED"},
    "cardboard": {"bin": "Bin 3 - Cardboard", "color": "#A0693B"},
    "glass": {"bin": "Bin 4 - Glass", "color": "#27AE60"},
    "metal": {"bin": "Bin 5 - Metal", "color": "#7F8C8D"},
    "general_waste": {"bin": "Bin 6 - General Waste", "color": "#333F48"},
}


def get_recommendation(category: str) -> dict[str, str]:
    """Return {'action', 'tip', 'disclaimer'} for a class name; raises KeyError if unknown."""
    if category not in RECOMMENDATIONS:
        raise KeyError(f"No recommendation configured for category '{category}'")
    return {**RECOMMENDATIONS[category], "disclaimer": DISCLAIMER}


def get_bin(category: str) -> dict[str, str]:
    if category not in SIMULATED_BINS:
        raise KeyError(f"No simulated bin configured for category '{category}'")
    return SIMULATED_BINS[category]


def validate_coverage() -> list[str]:
    """Return configured classes missing a recommendation or bin (empty list = OK)."""
    return [c for c in class_names() if c not in RECOMMENDATIONS or c not in SIMULATED_BINS]
