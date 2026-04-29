from typing import Dict, List

from deployment.fastapi.action_knowledge import get_action_knowledge


def _topk_lines(prediction: Dict) -> List[str]:
    lines: List[str] = []
    for idx, item in enumerate(prediction.get("topk", []), start=1):
        lines.append(f"{idx}. {item.get('name', '')}: {float(item.get('score', 0.0)):.6f}")
    return lines


def _score_gap(prediction: Dict) -> float:
    topk = prediction.get("topk", [])
    if len(topk) < 2:
        return float(prediction.get("score", 0.0))
    return float(topk[0].get("score", 0.0)) - float(topk[1].get("score", 0.0))


def _language_pack(language: str) -> Dict[str, str]:
    if str(language).lower().startswith("zh"):
        return {
            "preferred_language": "中文",
            "no_topk": "无候选类别信息",
            "system_explain": "你是一个人体动作识别结果解释助手。",
            "system_qa": "你是一个人体动作识别结果问答助手。",
        }
    return {
        "preferred_language": "English",
        "no_topk": "No top-k candidates.",
        "system_explain": "You explain skeleton-based action recognition results.",
        "system_qa": "You answer questions about skeleton-based action recognition results.",
    }


def build_explanation_messages(prediction: Dict, language: str = "zh") -> List[Dict[str, str]]:
    text = _language_pack(language)
    knowledge = get_action_knowledge(prediction.get("pred_name", ""))
    topk_text = "\n".join(_topk_lines(prediction)) or text["no_topk"]
    score = float(prediction.get("score", 0.0))
    gap = _score_gap(prediction)

    content = (
        f"请使用{text['preferred_language']}回答。\n\n"
        "任务：根据骨架动作识别结果，生成一段自然、专业、像人写的解释。\n\n"
        "你必须遵守这些约束：\n"
        "1. 只能依据提供的识别结果、候选类别分数和动作知识回答。\n"
        "2. 不要声称你直接看到了原始视频，也不要编造视频细节。\n"
        "3. 语气自然，不要像接口文档、实验报告或项目汇报。\n"
        "4. 先直接给结论，再补一两句依据；只有在确实需要时才提风险提示。\n"
        "5. 不要机械复述所有字段，不要把 top-k 原样抄成一段话。\n"
        "6. 如果置信度高且与第二候选差距明显，就语气更肯定；如果差距小，就明确指出存在相似类别。\n\n"
        "识别结果：\n"
        f"- 预测类别：{prediction.get('pred_name', '')}\n"
        f"- 置信度：{score:.6f}\n"
        f"- 与第二候选的分差：{gap:.6f}\n"
        f"- 样本名：{prediction.get('sample_name', '')}\n"
        f"- Top-k 候选：\n{topk_text}\n\n"
        "动作知识：\n"
        f"- 动作定义：{knowledge.get('definition', '')}\n"
        f"- 风险提示：{knowledge.get('risk_hint', '')}\n\n"
        "输出风格要求：\n"
        "- 中文场景下优先使用自然书面中文，必要时可保留动作英文标签。\n"
        "- 长度控制在 3 到 5 句。\n"
        "- 第一两句回答“为什么会判成这个动作”。\n"
        "- 如果需要风险提示，单独一句点到为止。\n"
    )
    return [
        {"role": "system", "content": text["system_explain"]},
        {"role": "user", "content": content},
    ]


def build_qa_messages(prediction: Dict, question: str, language: str = "zh") -> List[Dict[str, str]]:
    text = _language_pack(language)
    knowledge = get_action_knowledge(prediction.get("pred_name", ""))
    topk_text = "\n".join(_topk_lines(prediction)) or text["no_topk"]
    score = float(prediction.get("score", 0.0))
    gap = _score_gap(prediction)

    content = (
        f"请使用{text['preferred_language']}回答。\n\n"
        "任务：回答用户关于动作识别结果的追问。\n\n"
        "你必须遵守这些约束：\n"
        "1. 只能依据提供的识别结果、候选类别分数和动作知识回答。\n"
        "2. 不要声称你看到了原始视频，不要虚构场景细节。\n"
        "3. 回答要自然、直接，像在认真解释，而不是在背模板。\n"
        "4. 先正面回答用户问题，再给支撑理由。\n"
        "5. 如果用户问“为什么不是另一个动作”，重点比较两类动作的运动模式差异和候选分数差异。\n"
        "6. 如果用户问风险、可靠性或应用场景，要明确区分“模型判断”与“真实场景结论”。\n"
        "7. 除非用户主动要求，不要写成长篇列表。\n\n"
        "识别结果：\n"
        f"- 预测类别：{prediction.get('pred_name', '')}\n"
        f"- 置信度：{score:.6f}\n"
        f"- 与第二候选的分差：{gap:.6f}\n"
        f"- 样本名：{prediction.get('sample_name', '')}\n"
        f"- Top-k 候选：\n{topk_text}\n\n"
        "动作知识：\n"
        f"- 动作定义：{knowledge.get('definition', '')}\n"
        f"- 风险提示：{knowledge.get('risk_hint', '')}\n\n"
        f"用户问题：\n{question}\n\n"
        "输出风格要求：\n"
        "- 回答长度通常控制在 3 到 6 句。\n"
        "- 如果问题很明确，第一句就直接回答“是/不是/更可能/不太像”。\n"
        "- 如果需要比较另一个类别，请把比较写得简洁清楚。\n"
        "- 如果存在不确定性，请自然地说出来，不要生硬加免责声明。\n"
    )
    return [
        {"role": "system", "content": text["system_qa"]},
        {"role": "user", "content": content},
    ]


def template_explain(prediction: Dict, language: str = "zh") -> str:
    knowledge = get_action_knowledge(prediction.get("pred_name", ""))
    pred_name = prediction.get("pred_name", "")
    score = float(prediction.get("score", 0.0))
    second = prediction.get("topk", [None, None])[1] if len(prediction.get("topk", [])) > 1 else None
    gap = _score_gap(prediction)

    if str(language).lower().startswith("zh"):
        text = (
            f"系统当前更倾向于把这个动作识别为 {pred_name}，置信度约为 {score:.4f}。"
            f"从骨架时序特征来看，它和这个类别的运动模式比较接近。"
            f"{knowledge.get('definition', '')}"
        )
        if second is not None and gap < 0.1:
            text += (
                f" 不过它和第二候选 {second.get('name', '')}"
                f"（{float(second.get('score', 0.0)):.4f}）之间仍有一定相似性。"
            )
        text += f" {knowledge.get('risk_hint', '')}"
        return text

    text = (
        f"The system currently leans toward {pred_name} with confidence {score:.4f}. "
        f"Based on the skeleton motion pattern, it is fairly consistent with this class. "
        f"{knowledge.get('definition', '')}"
    )
    if second is not None and gap < 0.1:
        text += (
            f" It still has some similarity to the second candidate "
            f"{second.get('name', '')} ({float(second.get('score', 0.0)):.4f})."
        )
    text += f" {knowledge.get('risk_hint', '')}"
    return text


def template_answer(prediction: Dict, question: str, language: str = "zh") -> str:
    pred_name = prediction.get("pred_name", "")
    score = float(prediction.get("score", 0.0))
    knowledge = get_action_knowledge(pred_name)
    second = prediction.get("topk", [None, None])[1] if len(prediction.get("topk", [])) > 1 else None
    gap = _score_gap(prediction)

    if str(language).lower().startswith("zh"):
        answer = (
            f"基于当前识别结果，系统更倾向于把它判断为 {pred_name}，置信度是 {score:.4f}。"
        )
        if second is not None:
            answer += (
                f" 第二候选是 {second.get('name', '')}"
                f"（{float(second.get('score', 0.0)):.4f}），"
            )
            if gap >= 0.1:
                answer += "两者分数差距比较明显，所以当前判断相对更稳一些。"
            else:
                answer += "说明它和其他相似动作之间还存在一定混淆空间。"
        answer += f" 从动作定义来看，{knowledge.get('definition', '')} {knowledge.get('risk_hint', '')}"
        return answer

    answer = f"The system currently leans toward {pred_name} with confidence {score:.4f}. "
    if second is not None:
        answer += (
            f"The second candidate is {second.get('name', '')} "
            f"({float(second.get('score', 0.0)):.4f}). "
        )
        if gap >= 0.1:
            answer += "That gap makes the current prediction relatively stable. "
        else:
            answer += "That small gap suggests some ambiguity with similar actions. "
    answer += f"{knowledge.get('definition', '')} {knowledge.get('risk_hint', '')}"
    return answer
