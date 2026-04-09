"""旅行规划 LangGraph 工作流。

新的实现以 supervisor + 多独立 agent 为核心：
1. supervisor 先生成检索计划。
2. 景点 / 天气 / 酒店 agent 并行执行。
3. planner 只基于真实候选数据编排路线。
4. validator / meal agent 在后处理中补齐稳定性与真实餐饮。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from ..agents.langgraph_agents import (
    AttractionAgent,
    HotelAgent,
    MealService,
    PlanValidator,
    PlannerAgent,
    SupervisorPlan,
    SupervisorAgent,
    WeatherAgent,
)
from ..models.schemas import TripPlan, TripRequest
from ..tools.amap_mcp_tools import get_amap_mcp_client
from .trip_planner_state import TripPlannerState, create_initial_state

logger = logging.getLogger(__name__)


class TripPlannerWorkflow:
    """旅行规划工作流。"""

    def __init__(self) -> None:
        self.amap_client = get_amap_mcp_client()
        self.supervisor = SupervisorAgent()
        self.attraction_agent = AttractionAgent()
        self.weather_agent = WeatherAgent()
        self.hotel_agent = HotelAgent()
        self.planner_agent = PlannerAgent()
        self.plan_validator = PlanValidator()
        self.meal_service = MealService()
        self.graph = self._build_graph()

    @property
    def tool_names(self) -> list[str]:
        """暴露当前加载的 MCP 工具，便于健康检查。"""
        return self.amap_client.tool_names

    def _build_graph(self):
        """构建 supervisor + 多 agent 的状态图。"""
        workflow = StateGraph(TripPlannerState)

        workflow.add_node("supervisor", self._supervisor)
        workflow.add_node("search_attractions", self._search_attractions)
        workflow.add_node("check_weather", self._check_weather)
        workflow.add_node("find_hotels", self._find_hotels)
        workflow.add_node("quality_check", self._quality_check)
        workflow.add_node("plan_itinerary", self._plan_itinerary)
        workflow.add_node("abnormal_check", self._abnormal_check)
        workflow.add_node("find_meals", self._find_meals)
        workflow.add_node("handle_error", self._handle_error)

        workflow.set_entry_point("supervisor")
        workflow.add_edge("supervisor", "search_attractions")
        workflow.add_edge("supervisor", "check_weather")
        workflow.add_edge("supervisor", "find_hotels")
        workflow.add_edge(["search_attractions", "check_weather", "find_hotels"], "quality_check")
        workflow.add_conditional_edges(
            "quality_check",
            self._route_after_quality_check,
            {"continue": "plan_itinerary", "error": "handle_error"},
        )
        workflow.add_edge("plan_itinerary", "abnormal_check")
        workflow.add_edge("abnormal_check", "find_meals")
        workflow.add_edge("find_meals", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    async def _supervisor(self, state: TripPlannerState) -> Dict[str, Any]:
        """Supervisor 生成检索计划。"""
        started = time.perf_counter()
        try:
            plan = await self.supervisor.run(state["request"])
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info("node_done node=supervisor elapsed_ms=%d", elapsed_ms)
            return {
                "supervisor_plan": plan.model_dump(),
                "current_step": "supervisor_done",
                "messages": [{"role": "assistant", "content": "Supervisor 已完成任务拆解"}],
            }
        except Exception as exc:
            logger.error("Supervisor 执行失败: %s", exc, exc_info=True)
            return {"error": f"supervisor_failed: {exc}", "current_step": "supervisor_failed"}

    async def _search_attractions(self, state: TripPlannerState) -> Dict[str, Any]:
        """景点 agent。"""
        if state.get("error"):
            return {}
        try:
            attractions, summary = await self.attraction_agent.run(
                state["request"],
                self._ensure_supervisor_plan(state),
            )
            return {
                "attractions": attractions,
                "current_step": "attractions_done",
                "messages": [{"role": "assistant", "content": summary}],
            }
        except Exception as exc:
            logger.error("景点 agent 失败: %s", exc, exc_info=True)
            return {"error": f"attraction_failed: {exc}", "current_step": "attractions_failed"}

    async def _check_weather(self, state: TripPlannerState) -> Dict[str, Any]:
        """天气 agent。"""
        if state.get("error"):
            return {}
        try:
            weather_info, weather_notes = await self.weather_agent.run(state["request"])
            return {
                "weather_info": weather_info,
                "weather_notes": weather_notes,
                "current_step": "weather_done",
                "messages": [{"role": "assistant", "content": "天气智能体已完成真实天气检索"}],
            }
        except Exception as exc:
            logger.error("天气 agent 失败: %s", exc, exc_info=True)
            return {"error": f"weather_failed: {exc}", "current_step": "weather_failed"}

    async def _find_hotels(self, state: TripPlannerState) -> Dict[str, Any]:
        """酒店 agent。"""
        if state.get("error"):
            return {}
        try:
            hotels, summary = await self.hotel_agent.run(state["request"], self._ensure_supervisor_plan(state))
            return {
                "hotels": hotels,
                "current_step": "hotels_done",
                "messages": [{"role": "assistant", "content": summary}],
            }
        except Exception as exc:
            logger.error("酒店 agent 失败: %s", exc, exc_info=True)
            return {"error": f"hotel_failed: {exc}", "current_step": "hotels_failed"}

    async def _quality_check(self, state: TripPlannerState) -> Dict[str, Any]:
        """聚合后的质量检查。"""
        if state.get("error"):
            return {}

        problems = []
        if not state.get("attractions"):
            problems.append("景点结果为空")
        if not state.get("weather_info"):
            problems.append("天气结果为空")
        if not state.get("hotels"):
            problems.append("酒店结果为空")

        if problems:
            logger.error("质量检查失败: %s", "; ".join(problems))
            return {
                "error": "quality_failed: " + "; ".join(problems),
                "current_step": "quality_failed",
                "messages": [{"role": "assistant", "content": "真实检索结果不足，无法继续规划"}],
            }

        logger.info(
            "node_done node=quality_check elapsed_ms=%d warns=%d",
            0,
            0,
        )
        return {
            "current_step": "quality_passed",
            "messages": [{"role": "assistant", "content": "景点 / 天气 / 酒店真实数据已齐备"}],
        }

    def _route_after_quality_check(self, state: TripPlannerState) -> str:
        """质量检查后的路由。"""
        return "error" if state.get("error") else "continue"

    async def _plan_itinerary(self, state: TripPlannerState) -> Dict[str, Any]:
        """规划 agent。"""
        if state.get("error"):
            return {}
        started = time.perf_counter()
        try:
            trip_plan = await self.planner_agent.run(
                state["request"],
                self._ensure_supervisor_plan(state),
                state.get("attractions", []),
                state.get("hotels", []),
                state.get("weather_info", []),
                state.get("weather_notes", []),
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info("node_done node=plan_itinerary status=success elapsed_ms=%d", elapsed_ms)
            return {
                "trip_plan": trip_plan,
                "current_step": "plan_done",
                "messages": [{"role": "assistant", "content": "规划智能体已生成结构化行程"}],
            }
        except Exception as exc:
            logger.error("规划 agent 失败: %s", exc, exc_info=True)
            return {"error": f"planner_failed: {exc}", "current_step": "planner_failed"}

    async def _abnormal_check(self, state: TripPlannerState) -> Dict[str, Any]:
        """规则校验 / 修复。"""
        if state.get("error") or not state.get("trip_plan"):
            return {}
        started = time.perf_counter()

        trip_plan, alerts = self.plan_validator.run(
            state["request"],
            state["trip_plan"],
            state.get("attractions", []),
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "node_done node=abnormal_check status=checked initial_alerts=%d repaired=%d remaining=%d elapsed_ms=%d",
            len(alerts),
            len(alerts),
            0,
            elapsed_ms,
        )
        return {
            "trip_plan": trip_plan,
            "abnormal_alerts": alerts,
            "current_step": "abnormal_checked",
            "messages": [{"role": "assistant", "content": "规则校验与天气修复已完成"}],
        }

    async def _find_meals(self, state: TripPlannerState) -> Dict[str, Any]:
        """餐饮 agent。"""
        if state.get("error") or not state.get("trip_plan"):
            return {}
        try:
            trip_plan = await self.meal_service.run(state["request"], state["trip_plan"])
            return {
                "trip_plan": trip_plan,
                "current_step": "meal_done",
                "messages": [{"role": "assistant", "content": "餐饮智能体已补齐真实餐饮信息"}],
            }
        except Exception as exc:
            logger.error("餐饮 agent 失败: %s", exc, exc_info=True)
            # 餐饮补齐失败不直接打断主流程，保留 planner 给出的占位餐饮。
            return {
                "current_step": "meal_failed",
                "messages": [{"role": "assistant", "content": f"餐饮补齐失败，保留规划餐饮占位: {exc}"}],
            }

    async def _handle_error(self, state: TripPlannerState) -> Dict[str, Any]:
        """错误终止节点。"""
        logger.error("工作流中止: %s", state.get("error"))
        return {"current_step": "error"}

    def _ensure_supervisor_plan(self, state: TripPlannerState):
        """把 state 中的 supervisor_plan 还原成模型。"""
        return SupervisorPlan.model_validate(state.get("supervisor_plan") or {})

    async def plan_trip(self, request: TripRequest) -> TripPlan:
        """执行完整工作流。"""
        started = time.perf_counter()
        logger.info(
            "workflow_start city=%s start=%s end=%s days=%d",
            request.city,
            request.start_date,
            request.end_date,
            request.travel_days,
        )

        final_state = await self.graph.ainvoke(create_initial_state(request))

        if final_state.get("error"):
            raise RuntimeError(str(final_state.get("error")))

        trip_plan = final_state.get("trip_plan")
        if not isinstance(trip_plan, TripPlan):
            raise RuntimeError("workflow finished without trip plan")

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "workflow_done city=%s days=%d alerts=%d elapsed_ms=%d",
            request.city,
            len(trip_plan.days),
            len(final_state.get("abnormal_alerts", [])),
            elapsed_ms,
        )
        return trip_plan


_trip_planner_workflow: Optional[TripPlannerWorkflow] = None


def get_trip_planner_workflow() -> TripPlannerWorkflow:
    """获取单例工作流。"""
    global _trip_planner_workflow
    if _trip_planner_workflow is None:
        _trip_planner_workflow = TripPlannerWorkflow()
    return _trip_planner_workflow


def reset_workflow() -> None:
    """重置单例工作流。"""
    global _trip_planner_workflow
    _trip_planner_workflow = None
