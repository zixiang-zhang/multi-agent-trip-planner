"""旅行规划工作流状态。"""

from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional

from typing_extensions import Annotated, TypedDict

from ..models.schemas import Attraction, Hotel, TripPlan, TripRequest, WeatherInfo


def update_step(prev: str, new: str) -> str:
    """current_step 的 reducer。"""
    return new or prev


def merge_error(prev: Optional[str], new: Optional[str]) -> Optional[str]:
    """并行分支合并错误，避免 LangGraph 的并发更新冲突。"""
    if not prev:
        return new
    if not new or new == prev:
        return prev
    return f"{prev}; {new}"


class TripPlannerState(TypedDict):
    """LangGraph 中共享的状态。"""

    request: TripRequest
    supervisor_plan: Dict[str, Any]

    attractions: List[Attraction]
    hotels: List[Hotel]
    weather_info: List[WeatherInfo]
    weather_notes: List[str]
    abnormal_alerts: List[str]

    trip_plan: Optional[TripPlan]
    error: Annotated[Optional[str], merge_error]
    current_step: Annotated[str, update_step]
    messages: Annotated[List[Dict[str, Any]], operator.add]


def create_initial_state(request: TripRequest) -> TripPlannerState:
    """构造初始状态。"""
    return {
        "request": request,
        "supervisor_plan": {},
        "attractions": [],
        "hotels": [],
        "weather_info": [],
        "weather_notes": [],
        "abnormal_alerts": [],
        "trip_plan": None,
        "error": None,
        "current_step": "started",
        "messages": [],
    }


def has_error(state: TripPlannerState) -> bool:
    """是否已经有错误。"""
    return state.get("error") is not None
