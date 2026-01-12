from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ScoreItem:
    name: str
    score: float
    note: str

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def stars_from_score(score_0_100: float) -> str:
    s = int(round(clamp(score_0_100,0,100) / 20))
    return "⭐️"*s if s>0 else "—"

def summarize(items: List[ScoreItem]) -> tuple[float, List[Dict[str, Any]]]:
    total = sum(i.score for i in items)
    total = clamp(total, 0, 100)
    details = [{"name": i.name, "score": round(i.score,1), "note": i.note} for i in items]
    return total, details
