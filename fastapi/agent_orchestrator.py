import time
from typing import Any, Callable, Dict, List, Optional

from deployment.fastapi.agent_schemas import AgentPlanStep, AgentToolResult, ToolName
from deployment.fastapi.agent_tools import (
    assess_action_risk,
    check_reliability,
    compare_topk_actions,
    explain_action,
    recognize_action,
)


LOW_CONFIDENCE_THRESHOLD = 0.65
LOW_GAP_THRESHOLD = 0.08


def _contains_any(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _topk(prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    topk = prediction.get("topk", [])
    return topk if isinstance(topk, list) else []


class ActionAgentOrchestrator:
    def __init__(
        self,
        resolver_fn: Callable[[str, int], Dict[str, Any]],
        llm_explain_fn: Callable[[Dict[str, Any], str], Dict[str, Any]],
        llm_answer_fn: Callable[[Dict[str, Any], str, str], Dict[str, Any]],
    ):
        self.resolver_fn = resolver_fn
        self.llm_explain_fn = llm_explain_fn
        self.llm_answer_fn = llm_answer_fn

    def plan(self, instruction: str) -> List[AgentPlanStep]:
        steps: List[AgentPlanStep] = [
            AgentPlanStep(tool="recognize_action", reason="Recognition is the required first step for all requests."),
            AgentPlanStep(tool="check_reliability", reason="Reliability is always checked to support downstream tool decisions."),
        ]

        text = instruction.strip().lower()

        if _contains_any(
            text,
            ["compare", "why not", "top-2", "top2", "second", "compare candidates", "比较", "为什么不是", "第二"],
        ):
            steps.append(AgentPlanStep(tool="compare_topk_actions", reason="The instruction explicitly asks for candidate comparison."))

        if _contains_any(text, ["risk", "danger", "safety", "hazard", "危险", "风险", "安全", "场景"]):
            steps.append(AgentPlanStep(tool="assess_action_risk", reason="The instruction asks for a risk or safety judgment."))

        steps.append(AgentPlanStep(tool="explain_action", reason="A concise explanation supports the final analysis report."))
        return steps

    def run(self, video_filename: str, instruction: str, topk: int = 5, language: str = "zh") -> Dict[str, Any]:
        t_total = time.perf_counter()
        plan = self.plan(instruction)
        used_tools: List[ToolName] = []
        tool_results: List[AgentToolResult] = []
        outputs_by_tool: Dict[str, Dict[str, Any]] = {}

        recognition = recognize_action(video_filename=video_filename, topk=topk, resolver_fn=self.resolver_fn)
        predictions = recognition.get("predictions", [])
        if not predictions:
            raise RuntimeError("Prediction result is empty.")
        prediction = predictions[0]

        used_tools.append("recognize_action")
        tool_results.append(
            AgentToolResult(
                tool="recognize_action",
                summary=f"Predicted {prediction.get('pred_name', '')} with confidence {float(prediction.get('score', 0.0)):.4f}.",
                output=recognition,
            )
        )
        outputs_by_tool["recognize_action"] = recognition

        planned_tools = [step.tool for step in plan]

        reliability_output = check_reliability(prediction)
        used_tools.append("check_reliability")
        tool_results.append(
            AgentToolResult(
                tool="check_reliability",
                summary=str(reliability_output.get("summary", "")),
                output=reliability_output,
            )
        )
        outputs_by_tool["check_reliability"] = reliability_output

        auto_compare_reason = self._maybe_add_auto_compare(plan, planned_tools, reliability_output)
        if auto_compare_reason:
            plan.append(AgentPlanStep(tool="compare_topk_actions", reason=auto_compare_reason))
            planned_tools.append("compare_topk_actions")

        for step in plan[2:]:
            if step.tool in used_tools:
                continue

            if step.tool == "compare_topk_actions":
                output = compare_topk_actions(prediction, candidate_indices=[])
            elif step.tool == "assess_action_risk":
                output = assess_action_risk(prediction, scenario=instruction)
            elif step.tool == "explain_action":
                output = explain_action(prediction, llm_explain_fn=self.llm_explain_fn, language=language)
            else:
                continue

            used_tools.append(step.tool)
            tool_results.append(
                AgentToolResult(
                    tool=step.tool,
                    summary=str(output.get("summary", "")),
                    output=output,
                )
            )
            outputs_by_tool[step.tool] = output

        analysis_report = self._build_analysis_report(
            prediction=prediction,
            instruction=instruction,
            tool_outputs=outputs_by_tool,
            language=language,
        )

        narrative = self.llm_answer_fn(prediction, instruction, language)
        narrative_text = str(narrative.get("text", "")).strip()
        final_answer = self._format_report_text(analysis_report, narrative_text=narrative_text, language=language)

        metrics = dict(recognition.get("metrics", {}))
        metrics["agent_total_ms"] = round((time.perf_counter() - t_total) * 1000, 3)

        return {
            "user_goal": instruction,
            "instruction": instruction,
            "used_tools": used_tools,
            "plan": [step.dict() for step in plan],
            "tool_results": [item.dict() for item in tool_results],
            "analysis_report": analysis_report,
            "final_answer": final_answer,
            "prediction": prediction,
            "mapping": recognition.get("mapping", {}),
            "metrics": metrics,
        }

    def _maybe_add_auto_compare(
        self,
        plan: List[AgentPlanStep],
        planned_tools: List[ToolName],
        reliability_output: Dict[str, Any],
    ) -> Optional[str]:
        if "compare_topk_actions" in planned_tools:
            return None

        top1_score = float(reliability_output.get("top1_score", 0.0))
        score_gap = float(reliability_output.get("score_gap", 0.0))
        reliability_level = str(reliability_output.get("reliability_level", ""))
        if top1_score < LOW_CONFIDENCE_THRESHOLD or score_gap < LOW_GAP_THRESHOLD or reliability_level == "low":
            return (
                "The agent auto-added top-2 comparison because confidence is low or the "
                "Top-1 / Top-2 score gap is narrow."
            )
        return None

    def _build_analysis_report(
        self,
        prediction: Dict[str, Any],
        instruction: str,
        tool_outputs: Dict[str, Dict[str, Any]],
        language: str,
    ) -> Dict[str, Any]:
        reliability = tool_outputs.get("check_reliability", {})
        comparison = tool_outputs.get("compare_topk_actions", {})
        risk = tool_outputs.get("assess_action_risk", {})
        explanation = tool_outputs.get("explain_action", {})

        topk = _topk(prediction)
        top2_name = topk[1].get("name", "") if len(topk) > 1 else ""
        top2_score = float(topk[1].get("score", 0.0)) if len(topk) > 1 else 0.0

        report = {
            "action": {
                "predicted_action": prediction.get("pred_name", ""),
                "label_id": prediction.get("pred_label", ""),
                "confidence": round(float(prediction.get("score", 0.0)), 6),
            },
            "reliability": {
                "level": reliability.get("reliability_level", "unknown"),
                "verdict": reliability.get("verdict", "unknown"),
                "score_gap": round(float(reliability.get("score_gap", 0.0)), 6),
                "auto_compared_top2": "compare_topk_actions" in tool_outputs,
            },
            "top2_comparison": {
                "available": bool(comparison),
                "second_candidate": top2_name,
                "second_score": round(top2_score, 6),
                "preferred": comparison.get("preferred", prediction.get("pred_name", "")),
                "comparison_summary": comparison.get("summary", ""),
            },
            "risk_assessment": {
                "level": risk.get("risk_level", "not_requested"),
                "prior_level": risk.get("prior_risk_level", ""),
                "scenario_level": risk.get("scenario_risk_level", ""),
                "summary": risk.get("summary", ""),
            },
            "explanation": {
                "mode": explanation.get("mode", ""),
                "text": explanation.get("text", ""),
            },
            "recommendation": {
                "decision": self._recommendation_decision(reliability, risk),
                "next_step": self._recommendation_next_step(reliability, risk, comparison, language),
            },
            "instruction_focus": instruction,
        }
        return report

    def _recommendation_decision(self, reliability: Dict[str, Any], risk: Dict[str, Any]) -> str:
        reliability_level = str(reliability.get("reliability_level", "unknown"))
        risk_level = str(risk.get("risk_level", "not_requested"))
        if risk_level == "high" and reliability_level == "low":
            return "high_risk_low_confidence_review"
        if risk_level == "high":
            return "high_risk_attention"
        if reliability_level == "low":
            return "needs_candidate_review"
        return "usable_result"

    def _recommendation_next_step(
        self,
        reliability: Dict[str, Any],
        risk: Dict[str, Any],
        comparison: Dict[str, Any],
        language: str,
    ) -> str:
        reliability_level = str(reliability.get("reliability_level", "unknown"))
        risk_level = str(risk.get("risk_level", "not_requested"))
        has_comparison = bool(comparison)
        zh = str(language).lower().startswith("zh")

        if risk_level == "high" and reliability_level == "low":
            return "建议人工复核，并结合更多上下文确认是否存在真实安全风险。" if zh else "Recommend manual review with more context before making a safety judgment."
        if reliability_level == "low" and not has_comparison:
            return "建议补充 top-2 候选对比，避免直接采用单一结论。" if zh else "Recommend adding a top-2 comparison before relying on a single conclusion."
        if risk_level == "high":
            return "可将该动作标记为需要关注的行为，再结合场景做进一步处置。" if zh else "Treat this action as attention-worthy and confirm with scene context."
        return "当前结果可作为演示级分析结论使用。" if zh else "The current result is usable as a demo-level analysis output."

    def _format_report_text(self, report: Dict[str, Any], narrative_text: str, language: str) -> str:
        action = report.get("action", {})
        reliability = report.get("reliability", {})
        comparison = report.get("top2_comparison", {})
        risk = report.get("risk_assessment", {})
        explanation = report.get("explanation", {})
        recommendation = report.get("recommendation", {})

        if str(language).lower().startswith("zh"):
            lines = [
                "【Action】",
                f"- 预测动作: {action.get('predicted_action', '')}",
                f"- 置信度: {action.get('confidence', '')}",
                "",
                "【Reliability】",
                f"- 可靠性等级: {reliability.get('level', 'unknown')}",
                f"- Top1/Top2 分数差: {reliability.get('score_gap', '')}",
                f"- 是否自动触发 top2 对比: {reliability.get('auto_compared_top2', False)}",
                "",
                "【Top-2 Comparison】",
                f"- 第二候选: {comparison.get('second_candidate', '')}",
                f"- 当前更偏向: {comparison.get('preferred', '')}",
                f"- 对比结论: {comparison.get('comparison_summary', 'not triggered')}",
                "",
                "【Risk Assessment】",
                f"- 风险等级: {risk.get('level', 'not_requested')}",
                f"- 动作先验风险: {risk.get('prior_level', '') or 'not requested'}",
                f"- 场景条件风险: {risk.get('scenario_level', '') or 'not requested'}",
                f"- 风险说明: {risk.get('summary', '') or 'not requested'}",
                "",
                "【Why This Label】",
                explanation.get("text", "") or "No explanation available.",
                "",
                "【Recommendation】",
                f"- 决策标签: {recommendation.get('decision', '')}",
                f"- 建议动作: {recommendation.get('next_step', '')}",
            ]
            if narrative_text:
                lines.extend(["", "【Narrative Summary】", narrative_text])
            return "\n".join(lines)

        lines = [
            "[Action]",
            f"- Predicted action: {action.get('predicted_action', '')}",
            f"- Confidence: {action.get('confidence', '')}",
            "",
            "[Reliability]",
            f"- Reliability level: {reliability.get('level', 'unknown')}",
            f"- Top1/Top2 gap: {reliability.get('score_gap', '')}",
            f"- Auto top-2 comparison: {reliability.get('auto_compared_top2', False)}",
            "",
            "[Top-2 Comparison]",
            f"- Second candidate: {comparison.get('second_candidate', '')}",
            f"- Current preference: {comparison.get('preferred', '')}",
            f"- Comparison note: {comparison.get('comparison_summary', 'not triggered')}",
            "",
            "[Risk Assessment]",
            f"- Risk level: {risk.get('level', 'not_requested')}",
            f"- Action prior risk: {risk.get('prior_level', '') or 'not requested'}",
            f"- Scenario condition risk: {risk.get('scenario_level', '') or 'not requested'}",
            f"- Risk note: {risk.get('summary', '') or 'not requested'}",
            "",
            "[Why This Label]",
            explanation.get("text", "") or "No explanation available.",
            "",
            "[Recommendation]",
            f"- Decision tag: {recommendation.get('decision', '')}",
            f"- Next step: {recommendation.get('next_step', '')}",
        ]
        if narrative_text:
            lines.extend(["", "[Narrative Summary]", narrative_text])
        return "\n".join(lines)
