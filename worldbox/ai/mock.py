"""An offline stand-in for a tribe mind: rules instead of a language model.

Selected with ``LLM_BASE_URL=mock`` in ``.env``. It reads the very briefing a
real model would get and answers with a policy in the same JSON shape, so the
whole mind pipeline -- scheduling, parsing, validation, the HUD -- runs for
free, offline and instantly. Useful for building on top of the minds without
spending anything; it is not meant to play well.

Same briefing in, same policy out: no randomness here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_HUNGER = re.compile(r"Hunger: (\d+)/100")
_PEOPLE = re.compile(r"People: (\d+)\.")
_GRANARY = re.compile(r"\((\d+)% full\)")
_NEIGHBOUR = re.compile(r"tribe_id (\d+) .*?: (\d+) people, .*?relation (-?\d+), (\d+) tiles( AT WAR)?")
_MATERIALS = re.compile(r"Materials at hand: (.*)\.")
_YOU = re.compile(r"\(tribe_id (\d+)\)")

# Crafts the mock may propose, first match wins. Materials must be at hand;
# the engine validates them again either way.
_RECIPES: Tuple[Tuple[Tuple[str, ...], str, str, str], ...] = (
    (("water", "forest"), "Fish Traps", "food", "The river can feed us if we weave traps."),
    (("mountain", "war"), "Stone Walls", "combat", "Stone will hold where wood burns."),
    (("forest", "food"), "Herb Garden", "medicine", "Herbs near the camp keep the sick alive."),
    (("grass", "settlement"), "Sown Fields", "food", "Seed the open ground and stay put."),
)

_INTENTS = {
    "survive": "Keep everyone alive until the land recovers.",
    "farm": "Fill the granary before the lean months.",
    "expand": "Spread out and claim the good ground.",
    "defend": "Guard the camp and hold our ground.",
    "research": "Our food is safe; let the wise ones study.",
    "raid": "Take what our enemies cannot hold.",
    "trade": "Friends at the border are worth more than land.",
}


def _neighbours(text: str) -> List[Dict[str, Any]]:
    rows = []
    for match in _NEIGHBOUR.finditer(text):
        rows.append({
            "id": int(match.group(1)),
            "size": int(match.group(2)),
            "relation": int(match.group(3)),
            "distance": int(match.group(4)),
            "war": bool(match.group(5)),
        })
    return rows


def _invention(materials: List[str]) -> Optional[Dict[str, Any]]:
    have = set(materials)
    for needs, name, effect, why in _RECIPES:
        if set(needs) <= have:
            return {"name": name, "from": list(needs), "effect": effect, "why": why}
    return None


def decide(briefing: str) -> Dict[str, Any]:
    """A policy for the tribe described by ``briefing``."""
    hunger = int(_HUNGER.search(briefing).group(1)) if _HUNGER.search(briefing) else 30
    size = int(_PEOPLE.search(briefing).group(1)) if _PEOPLE.search(briefing) else 10
    granary_match = _GRANARY.search(briefing)
    granary = int(granary_match.group(1)) if granary_match else None
    neighbours = _neighbours(briefing)
    materials_match = _MATERIALS.search(briefing)
    materials = [m.strip() for m in materials_match.group(1).split(",")] if materials_match else []

    enemies = [n for n in neighbours if n["war"]]
    stronger_enemy = any(n["size"] > size for n in enemies)

    if enemies:
        stance = "defend" if stronger_enemy else "raid"
    elif hunger >= 55 or (granary is not None and granary < 25):
        stance = "farm"
    elif hunger >= 40:
        stance = "survive"
    elif size >= 30 and granary is not None and granary >= 50:
        stance = "research"
    elif any(n["relation"] >= 20 for n in neighbours):
        stance = "trade"
    else:
        stance = "expand"

    shares = {
        "farm": (0.08, 0.50, 0.08, 0.06),
        "survive": (0.08, 0.40, 0.06, 0.10),
        "defend": (0.30, 0.30, 0.06, 0.08),
        "raid": (0.35, 0.25, 0.05, 0.06),
        "research": (0.08, 0.32, 0.24, 0.08),
        "trade": (0.08, 0.36, 0.14, 0.08),
        "expand": (0.10, 0.34, 0.12, 0.08),
    }[stance]

    toward = []
    for n in neighbours:
        if n["war"] and stance == "raid":
            toward.append({"tribe_id": n["id"], "action": "war"})
        elif n["war"]:
            toward.append({"tribe_id": n["id"], "action": "peace"})
        elif n["relation"] >= 20:
            toward.append({"tribe_id": n["id"], "action": "trade"})
    toward = toward[:3]

    return {
        "stance": stance,
        "warrior_share": shares[0],
        "farmer_share": shares[1],
        "scholar_share": shares[2],
        "healer_share": shares[3],
        "cohesion": 0.75 if stance in ("defend", "farm", "survive") else 0.45,
        "fight": 1.4 if stance == "raid" else (1.1 if stance == "defend" else 0.9),
        "flee": 0.8 if stance in ("raid", "defend") else 1.1,
        "seek_food": 1.3 if stance in ("farm", "survive") else 1.0,
        "seek_mate": 0.8 if hunger >= 50 else 1.1,
        "toward": toward,
        "invent": _invention(materials),
        "intent": _INTENTS[stance],
    }


def reply(messages: List[Dict[str, str]]) -> str:
    """The JSON text a chat model would have returned for these messages."""
    briefing = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    if "tribe_id" not in briefing:
        # Not a tribe briefing (e.g. the narrator): answer plainly.
        return "The mock mind keeps no history. Configure a real model to narrate."
    return json.dumps(decide(briefing))
