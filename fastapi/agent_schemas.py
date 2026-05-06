from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ToolName = Literal[
    "recognize_action",
    "check_reliability",
    "compare_topk_actions",
    "assess_action_risk",
    "explain_action",
]


class AgentRunToolRequest(BaseModel):
    tool: ToolName
    video_filename: str = Field(default="", description="Used by recognize_action.")
    topk: int = Field(default=5, ge=1, le=120, description="Used by recognize_action.")
    prediction: Optional[Dict[str, Any]] = Field(default=None, description="Used by non-recognition tools.")
    language: str = Field(default="zh", description="Used by explain_action.")
    scenario: str = Field(default="", description="Used by assess_action_risk.")
    candidate_indices: List[int] = Field(default_factory=list, description="Used by compare_topk_actions.")


class AgentToolResult(BaseModel):
    tool: ToolName
    summary: str
    output: Dict[str, Any]


class AgentAnalyzeRequest(BaseModel):
    video_filename: str = Field(..., min_length=1)
    instruction: str = Field(..., min_length=1)
    topk: int = Field(default=5, ge=1, le=120)
    language: str = Field(default="zh")


class AgentPlanStep(BaseModel):
    tool: ToolName
    reason: str


class AgentAnalyzeResponse(BaseModel):
    user_goal: str
    instruction: str
    used_tools: List[ToolName]
    plan: List[AgentPlanStep]
    tool_results: List[AgentToolResult]
    analysis_report: Dict[str, Any]
    final_answer: str
    prediction: Dict[str, Any]
    mapping: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
