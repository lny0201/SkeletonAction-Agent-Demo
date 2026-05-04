from typing import Dict


DEFAULT_KNOWLEDGE = {
    "definition": "This action is inferred from skeleton-based motion patterns rather than raw RGB video semantics.",
    "risk_hint": "If the confidence is low or the top-k classes are close, the result should be treated cautiously.",
}


ACTION_KNOWLEDGE: Dict[str, Dict[str, str]] = {
    "throw": {
        "definition": "Usually involves a clear forward or outward arm swing, often with coordinated trunk rotation and a quick release-like motion.",
        "risk_hint": "In public-space or safety scenarios, this action may deserve extra attention because it can involve object projection.",
    },
    "cheer up": {
        "definition": "Typically involves uplifting arm and upper-body gestures with positive, expressive movement.",
        "risk_hint": "Usually a low-risk daily action.",
    },
    "drink water": {
        "definition": "Usually shows one hand moving toward the mouth with a short, controlled upper-body motion.",
        "risk_hint": "Usually a low-risk daily action.",
    },
    "sitting down": {
        "definition": "Usually shows the body center lowering in a controlled way as the subject transitions from standing to sitting.",
        "risk_hint": "Generally low risk, but can be confused with falling when the transition is abrupt.",
    },
    "standing up": {
        "definition": "Usually shows the body center rising from a lower posture with coordinated trunk and leg extension.",
        "risk_hint": "Generally low risk, though unstable motions can reduce confidence.",
    },
    "hand waving": {
        "definition": "Usually shows repeated arm movement with relatively stable lower-body posture.",
        "risk_hint": "Usually a low-risk expressive gesture.",
    },
    "salute": {
        "definition": "Usually shows a raised hand posture held near the head or forehead with more static upper-body structure than a throwing motion.",
        "risk_hint": "Usually a low-risk gesture, though it may look similar to other arm-raised actions in isolated frames.",
    },
    "jump up": {
        "definition": "Usually shows a brief upward displacement of the body center with coordinated leg extension.",
        "risk_hint": "Usually low risk, but may be confused with abrupt vertical motion in some samples.",
    },
    "taking a selfie": {
        "definition": "Usually shows one arm extended outward with controlled upper-body orientation.",
        "risk_hint": "Usually a low-risk daily action.",
    },
    "falling": {
        "definition": "Usually shows a rapid drop in body center and abrupt posture change over a short time window.",
        "risk_hint": "In safety or care scenarios, this should be treated as a potentially risky action.",
    },
    "walking": {
        "definition": "Usually shows stable, periodic gait motion and continuous displacement.",
        "risk_hint": "Usually a low-risk daily action.",
    },
}


def get_action_knowledge(action_name: str) -> Dict[str, str]:
    key = (action_name or "").strip().lower()
    return ACTION_KNOWLEDGE.get(key, DEFAULT_KNOWLEDGE)
