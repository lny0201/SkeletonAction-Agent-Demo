from typing import Any, Callable, Dict, List

from deployment.fastapi.action_knowledge import get_action_knowledge

LOW_RISK_ACTIONS = {
    "drink water",
    "cheer up",
    "hand waving",
    "salute",
    "taking a selfie",
    "walking",
}

MEDIUM_RISK_ACTIONS = {
    "jump up",
    "running",
    "sitting down",
    "standing up",
}

HIGH_RISK_ACTIONS = {
    "throw",
    "falling",
    "kicking",
    "punching",
    "hitting",
}

SCENARIO_HIGH_KEYWORDS = {
    "public safety",
    "security",
    "danger",
    "dangerous",
    "violence",
    "weapon",
    "attack",
    "fight",
    "knife",
    "gun",
    "hazard",
    "crowd panic",
    "risk control",
    "高风险",
    "危险",
    "安防",
    "公共安全",
    "打斗",
    "袭击",
    "刀",
    "枪",
}

SCENARIO_MEDIUM_KEYWORDS = {
    "street",
    "road",
    "traffic",
    "factory",
    "construction",
    "night",
    "emergency",
    "care center",
    "hospital",
    "school",
    "city",
    "场景",
    "道路",
    "车流",
    "工地",
    "夜间",
    "医院",
    "校园",
    "城市",
}


def _level_to_score(level: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(level, 0)


def _score_to_level(score: int) -> str:
    score = max(0, min(2, int(score)))
    return {0: "low", 1: "medium", 2: "high"}[score]


def _prior_action_risk(action_name: str) -> str:
    key = (action_name or "").strip().lower()
    if key in HIGH_RISK_ACTIONS:
        return "high"
    if key in MEDIUM_RISK_ACTIONS:
        return "medium"
    if key in LOW_RISK_ACTIONS:
        return "low"
    return "low"


def _scenario_condition_risk(scenario: str) -> str:
    text = (scenario or "").strip().lower()
    if not text:
        return "low"
    if any(token in text for token in SCENARIO_HIGH_KEYWORDS):
        return "high"
    if any(token in text for token in SCENARIO_MEDIUM_KEYWORDS):
        return "medium"
    return "low"


def _topk(prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = prediction.get("topk", [])
    if not isinstance(data, list):
        return []
    return data


def check_reliability(prediction: Dict[str, Any]) -> Dict[str, Any]:
    topk = _topk(prediction)
    top1 = float(prediction.get("score", 0.0))
    top2 = float(topk[1].get("score", 0.0)) if len(topk) > 1 else 0.0
    gap = top1 - top2

    if top1 >= 0.8 and gap >= 0.15:
        level = "high"
        verdict = "stable"
    elif top1 >= 0.6 and gap >= 0.08:
        level = "medium"
        verdict = "mostly_stable"
    else:
        level = "low"
        verdict = "uncertain"

    summary = f"confidence={top1:.4f}, gap={gap:.4f}, reliability={level}"
    return {
        "verdict": verdict,
        "reliability_level": level,
        "top1_score": top1,
        "top2_score": top2,
        "score_gap": gap,
        "summary": summary,
    }


def compare_topk_actions(prediction: Dict[str, Any], candidate_indices: List[int]) -> Dict[str, Any]:
    topk = _topk(prediction)
    if not topk:
        return {
            "compared": [],
            "summary": "No top-k candidates available.",
        }

    if not candidate_indices:
        candidate_indices = [0, 1]
    selected = []
    for idx in candidate_indices:
        if 0 <= idx < len(topk):
            selected.append(topk[idx])

    if len(selected) < 2 and len(topk) > 1:
        selected = [topk[0], topk[1]]

    if len(selected) < 2:
        return {
            "compared": selected,
            "summary": "Not enough candidates to compare.",
        }

    a = selected[0]
    b = selected[1]
    gap = float(a.get("score", 0.0)) - float(b.get("score", 0.0))
    preferred = a.get("name", "") if gap >= 0 else b.get("name", "")
    summary = (
        f"Compared {a.get('name', '')} ({float(a.get('score', 0.0)):.4f}) vs "
        f"{b.get('name', '')} ({float(b.get('score', 0.0)):.4f}); "
        f"current preference: {preferred}."
    )
    return {
        "compared": selected[:2],
        "score_gap": gap,
        "preferred": preferred,
        "summary": summary,
    }


def assess_action_risk(prediction: Dict[str, Any], scenario: str = "") -> Dict[str, Any]:
    pred_name = str(prediction.get("pred_name", ""))
    knowledge = get_action_knowledge(pred_name)
    prior_level = _prior_action_risk(pred_name)
    scenario_level = _scenario_condition_risk(scenario)

    prior_score = _level_to_score(prior_level)
    scenario_score = _level_to_score(scenario_level)

    # Two-layer fusion:
    # 1) action prior is the base
    # 2) scenario can raise by at most one level for low-risk actions
    #    and at most keep/elevate for already higher-risk actions.
    if prior_score == 0:
        combined_score = min(1, max(prior_score, scenario_score))
    else:
        combined_score = min(2, max(prior_score, scenario_score))
    level = _score_to_level(combined_score)

    summary = (
        f"risk_level={level} (prior={prior_level}, scenario={scenario_level}) "
        f"for action={pred_name}"
    )
    return {
        "risk_level": level,
        "action": pred_name,
        "scenario": scenario,
        "prior_risk_level": prior_level,
        "scenario_risk_level": scenario_level,
        "knowledge_risk_hint": knowledge.get("risk_hint", ""),
        "summary": summary,
    }


def explain_action(
    prediction: Dict[str, Any],
    llm_explain_fn: Callable[[Dict[str, Any], str], Dict[str, Any]],
    language: str = "zh",
) -> Dict[str, Any]:
    result = llm_explain_fn(prediction, language)
    mode = str(result.get("mode", "template"))
    text = str(result.get("text", ""))
    return {
        "mode": mode,
        "text": text,
        "error": str(result.get("error", "")),
        "summary": f"explanation_mode={mode}",
    }


def recognize_action(
    video_filename: str,
    topk: int,
    resolver_fn: Callable[[str, int], Dict[str, Any]],
) -> Dict[str, Any]:
    return resolver_fn(video_filename, topk)
