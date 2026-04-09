from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field

from ..models.schemas import Attraction, Budget, DayPlan, Hotel, Location, Meal, TripPlan, TripRequest, WeatherInfo
from ..services.llm_service import ainvoke_json
from ..tools.amap_mcp_tools import AmapMcpError, get_amap_mcp_client

logger = logging.getLogger(__name__)

COMMON_CITY_NAMES = {
    "北京", "上海", "广州", "深圳", "杭州", "苏州", "南京", "济南", "青岛", "成都", "重庆", "西安", "武汉", "长沙"
}
MEAL_SPECS: tuple[tuple[str, str, int], ...] = (
    ("breakfast", "早餐", 30),
    ("lunch", "午餐", 60),
    ("dinner", "晚餐", 80),
)


class SupervisorPlan(BaseModel):
    """Supervisor 产出的检索计划。"""

    planning_style: str = Field(default="节奏均衡，优先真实可落地安排")
    attraction_keywords: list[str] = Field(default_factory=list)
    hotel_keywords: list[str] = Field(default_factory=list)


class IdSelection(BaseModel):
    """候选筛选结果。"""

    selected_ids: list[str] = Field(default_factory=list)


class WeatherSummary(BaseModel):
    """天气总结结果。"""

    notes: list[str] = Field(default_factory=list)


class PlannerDayDraft(BaseModel):
    """轻量骨架规划结果。"""

    date: str
    description: str = ""
    attraction_ids: list[str] = Field(default_factory=list)
    hotel_id: str = ""


class PlannerDraft(BaseModel):
    """Planner 只输出骨架，不直接生成完整 TripPlan。"""

    days: list[PlannerDayDraft] = Field(default_factory=list)
    overall_suggestions: str = ""


@dataclass(slots=True)
class AttractionCandidate:
    """景点候选。"""

    id: str
    name: str
    address: str
    location: Location
    category: str
    rating: float | None
    photos: list[str]
    ticket_price: int
    visit_duration: int
    is_outdoor: bool
    description: str


@dataclass(slots=True)
class HotelCandidate:
    """酒店候选。"""

    id: str
    name: str
    address: str
    location: Location | None
    hotel_type: str
    rating: float | None
    estimated_cost: int
    price_range: str


class TripAgentBase:
    """所有 agent / 模块共享的基础能力。"""

    agent_name = "agent"

    @property
    def amap(self):
        return get_amap_mcp_client()

    async def ask_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> tuple[Any, str]:
        return await ainvoke_json(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            run_name=self.agent_name,
            tags=[self.agent_name],
        )


class SupervisorAgent(TripAgentBase):
    """负责把用户请求压缩为检索计划。"""

    agent_name = "supervisor"

    async def run(self, request: TripRequest) -> SupervisorPlan:
        system_prompt = (
            "你是旅行规划总控智能体。"
            "请只输出 JSON，结构为"
            " {\"planning_style\":\"\",\"attraction_keywords\":[\"\"],\"hotel_keywords\":[\"\"]}。"
            " attraction_keywords 只保留景点检索词，hotel_keywords 只保留酒店检索词，不要输出跨城市内容。"
        )
        user_prompt = json.dumps(
            {
                "city": request.city,
                "dates": build_trip_dates(request),
                "transportation": request.transportation,
                "accommodation": request.accommodation,
                "preferences": request.preferences,
                "free_text_input": request.free_text_input,
            },
            ensure_ascii=False,
        )
        try:
            parsed, _ = await self.ask_json(
                system_prompt,
                user_prompt,
                temperature=0.1,
                max_tokens=260,
                timeout=25.0,
            )
            plan = SupervisorPlan.model_validate(parsed)
            return self._sanitize_plan(request, plan)
        except Exception:
            logger.warning("SupervisorAgent 规划失败，回退到轻量规则计划", exc_info=True)
            return self._fallback_plan(request)

    def _sanitize_plan(self, request: TripRequest, plan: SupervisorPlan) -> SupervisorPlan:
        return SupervisorPlan(
            planning_style=(plan.planning_style or "节奏均衡，优先真实可落地安排")[:28],
            attraction_keywords=sanitize_supervisor_keywords(plan.attraction_keywords, request.city, request.preferences),
            hotel_keywords=dedup_list([*(plan.hotel_keywords or []), request.accommodation, "酒店"], limit=3),
        )

    def _fallback_plan(self, request: TripRequest) -> SupervisorPlan:
        return SupervisorPlan(
            planning_style="节奏均衡，优先真实可落地安排",
            attraction_keywords=sanitize_supervisor_keywords([], request.city, request.preferences),
            hotel_keywords=dedup_list([request.accommodation, "酒店", "宾馆"], limit=3),
        )


class AttractionAgent(TripAgentBase):
    """负责真实景点检索与候选筛选。"""

    agent_name = "attraction_agent"

    async def run(self, request: TripRequest, plan: SupervisorPlan) -> tuple[list[Attraction], str]:
        started = time.perf_counter()
        specs = build_attraction_search_specs(request, plan)
        results = await asyncio.gather(
            *(self.amap.text_search(spec, city=request.city, citylimit=True) for spec in specs),
            return_exceptions=True,
        )
        raw_pois: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            raw_pois.extend(((result or {}).get("pois") or [])[:6])

        raw_pois = await enrich_pois_with_details(self.amap, raw_pois, limit=24)

        candidates = dedup_attraction_candidates(parse_attraction_candidate(poi) for poi in raw_pois)
        ranked = rank_attraction_candidates(candidates, request)
        target = min(max(request.travel_days * 2, 4), 8)
        selected = await self._select_candidates(request, ranked[:10], target)
        attractions = [candidate_to_attraction(item) for item in selected]
        logger.info(
            "parse_done type=attractions records=%d output=%d elapsed_ms=%d",
            len(raw_pois),
            len(attractions),
            int((time.perf_counter() - started) * 1000),
        )
        return attractions, f"景点智能体已基于高德真实数据筛出 {len(attractions)} 个候选景点"

    async def _select_candidates(
        self,
        request: TripRequest,
        candidates: list[AttractionCandidate],
        target: int,
    ) -> list[AttractionCandidate]:
        if len(candidates) <= target:
            return candidates

        system_prompt = (
            "你是景点筛选智能体。"
            "请结合用户偏好，从候选中选择最适合本次行程的景点。"
            "只输出 JSON：{\"selected_ids\":[\"\"]}。"
            f" 最多选择 {target} 个，优先保留不同类型、不同区域的主景点。"
        )
        user_prompt = json.dumps(
            {
                "city": request.city,
                "preferences": request.preferences,
                "free_text_input": request.free_text_input,
                "travel_days": request.travel_days,
                "candidates": [
                    {
                        "id": item.id,
                        "name": item.name,
                        "category": item.category,
                        "rating": item.rating,
                        "is_outdoor": item.is_outdoor,
                        "visit_duration": item.visit_duration,
                    }
                    for item in candidates
                ],
            },
            ensure_ascii=False,
        )
        try:
            parsed, _ = await self.ask_json(
                system_prompt,
                user_prompt,
                temperature=0.15,
                max_tokens=260,
                timeout=25.0,
            )
            selection = IdSelection.model_validate(parsed)
            selected_ids = set(selection.selected_ids)
            picked = [item for item in candidates if item.id in selected_ids]
            if picked:
                return picked[:target]
        except Exception:
            logger.warning("AttractionAgent 候选筛选失败，回退到排序结果", exc_info=True)
        return candidates[:target]


class WeatherAgent(TripAgentBase):
    """负责天气工具调用和天气总结。"""

    agent_name = "weather_agent"

    async def run(self, request: TripRequest) -> tuple[list[WeatherInfo], list[str]]:
        started = time.perf_counter()
        payload = await self.amap.weather(request.city)
        weather_info = align_weather_to_trip_dates(parse_weather_payload(payload), request)
        notes = await self._build_notes(request, weather_info)
        logger.info(
            "node_done node=check_weather source=mcp count=%d elapsed_ms=%d city=%s",
            len(weather_info),
            int((time.perf_counter() - started) * 1000),
            request.city,
        )
        return weather_info, notes

    async def _build_notes(self, request: TripRequest, weather_info: list[WeatherInfo]) -> list[str]:
        system_prompt = (
            "你是天气分析智能体。"
            "请根据天气数据生成 1 到 2 条简短出行提醒。"
            "只输出 JSON：{\"notes\":[\"\"]}。"
        )
        user_prompt = json.dumps(
            {
                "city": request.city,
                "dates": build_trip_dates(request),
                "weather": [
                    {
                        "date": item.date,
                        "day_weather": item.day_weather,
                        "night_weather": item.night_weather,
                        "day_temp": item.day_temp,
                        "night_temp": item.night_temp,
                    }
                    for item in weather_info
                ],
            },
            ensure_ascii=False,
        )
        try:
            parsed, _ = await self.ask_json(
                system_prompt,
                user_prompt,
                temperature=0.1,
                max_tokens=180,
                timeout=20.0,
            )
            summary = WeatherSummary.model_validate(parsed)
            notes = dedup_list(summary.notes, limit=2)
            if notes:
                return notes
        except Exception:
            logger.warning("WeatherAgent 提醒生成失败，回退到规则提醒", exc_info=True)
        return build_weather_notes(weather_info)

class HotelAgent(TripAgentBase):
    """负责真实酒店检索与候选筛选。"""

    agent_name = "hotel_agent"

    async def run(self, request: TripRequest, plan: SupervisorPlan) -> tuple[list[Hotel], str]:
        started = time.perf_counter()
        keywords = dedup_list([*(plan.hotel_keywords or []), request.accommodation, "酒店", "宾馆"], limit=4)
        results = await asyncio.gather(
            *(self.amap.text_search(keyword, city=request.city, citylimit=True) for keyword in keywords),
            return_exceptions=True,
        )
        raw_pois: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            raw_pois.extend(((result or {}).get("pois") or [])[:6])

        raw_pois = await enrich_pois_with_details(self.amap, raw_pois, limit=18)

        candidates = dedup_hotel_candidates(parse_hotel_candidate(poi, request.accommodation) for poi in raw_pois)
        ranked = rank_hotel_candidates(candidates, request)
        target = min(max(request.travel_days, 2), 3)
        selected = await self._select_candidates(request, ranked[:6], target)
        hotels = [candidate_to_hotel(item) for item in selected]
        logger.info(
            "parse_done type=hotels records=%d output=%d elapsed_ms=%d",
            len(raw_pois),
            len(hotels),
            int((time.perf_counter() - started) * 1000),
        )
        return hotels, f"酒店智能体已基于高德真实数据筛出 {len(hotels)} 个酒店候选"

    async def _select_candidates(
        self,
        request: TripRequest,
        candidates: list[HotelCandidate],
        target: int,
    ) -> list[HotelCandidate]:
        if len(candidates) <= target:
            return candidates

        system_prompt = (
            "你是酒店筛选智能体。"
            "请根据住宿偏好、交通方式和整体行程舒适度选择酒店。"
            "只输出 JSON：{\"selected_ids\":[\"\"]}。"
            f" 最多选择 {target} 个。"
        )
        user_prompt = json.dumps(
            {
                "city": request.city,
                "transportation": request.transportation,
                "accommodation": request.accommodation,
                "preferences": request.preferences,
                "candidates": [
                    {
                        "id": item.id,
                        "name": item.name,
                        "type": item.hotel_type,
                        "rating": item.rating,
                        "estimated_cost": item.estimated_cost,
                    }
                    for item in candidates
                ],
            },
            ensure_ascii=False,
        )
        try:
            parsed, _ = await self.ask_json(
                system_prompt,
                user_prompt,
                temperature=0.15,
                max_tokens=220,
                timeout=25.0,
            )
            selection = IdSelection.model_validate(parsed)
            selected_ids = set(selection.selected_ids)
            picked = [item for item in candidates if item.id in selected_ids]
            if picked:
                return picked[:target]
        except Exception:
            logger.warning("HotelAgent 候选筛选失败，回退到排序结果", exc_info=True)
        return candidates[:target]


class PlannerAgent(TripAgentBase):
    """只负责编排每日骨架，不直接生成全量最终结果。"""

    agent_name = "planner_agent"

    async def run(
        self,
        request: TripRequest,
        plan: SupervisorPlan,
        attractions: list[Attraction],
        hotels: list[Hotel],
        weather_info: list[WeatherInfo],
        weather_notes: list[str],
    ) -> TripPlan:
        try:
            parsed, _ = await self.ask_json(
                self._system_prompt(request.travel_days),
                json.dumps(
                    {
                        "request": {
                            "city": request.city,
                            "dates": build_trip_dates(request),
                            "transportation": request.transportation,
                            "accommodation": request.accommodation,
                            "preferences": request.preferences,
                        },
                        "planning_style": plan.planning_style,
                        "weather_notes": weather_notes[:2],
                        "attractions": [
                            {
                                "id": attraction.poi_id or attraction.name,
                                "name": attraction.name,
                                "category": attraction.category,
                                "visit_duration": attraction.visit_duration,
                                "is_outdoor": is_outdoor_attraction(attraction),
                            }
                            for attraction in attractions[:6]
                        ],
                        "hotels": [
                            {
                                "id": hotel.name,
                                "name": hotel.name,
                                "type": hotel.type,
                            }
                            for hotel in hotels[:2]
                        ],
                        "weather": [
                            {
                                "date": item.date,
                                "day_weather": item.day_weather,
                                "night_weather": item.night_weather,
                                "rainy": is_rainy_weather(item),
                            }
                            for item in weather_info
                        ],
                    },
                    ensure_ascii=False,
                ),
                temperature=0.15,
                max_tokens=700,
                timeout=60.0,
            )
            draft = PlannerDraft.model_validate(parsed)
        except Exception:
            logger.warning("PlannerAgent 生成草案失败，回退到规则骨架规划", exc_info=True)
            draft = self._fallback_draft(request, attractions, hotels, weather_info)
        return self._build_trip_plan(request, draft, attractions, hotels, weather_info, weather_notes)

    def _system_prompt(self, travel_days: int) -> str:
        return (
            "你是行程编排智能体。"
            "请基于已有真实景点、酒店、天气数据，只输出轻量骨架 JSON。"
            "不要生成 meals、budget、weather_info 等最终结构。"
            "输出格式为"
            ' {"days":[{"date":"YYYY-MM-DD","description":"","attraction_ids":[""],"hotel_id":""}],"overall_suggestions":""}。'
            f" 必须输出 {travel_days} 天，每天 1 到 2 个景点，优先避免重复并考虑天气。"
        )

    def _fallback_draft(
        self,
        request: TripRequest,
        attractions: list[Attraction],
        hotels: list[Hotel],
        weather_info: list[WeatherInfo],
    ) -> PlannerDraft:
        dates = build_trip_dates(request)
        allocated = allocate_unique_attractions(attractions, len(dates), per_day_max=2)
        weather_by_date = {item.date: item for item in weather_info}
        default_hotel_id = hotels[0].name if hotels else ""
        days: list[PlannerDayDraft] = []
        for index, date in enumerate(dates):
            day_attractions = allocated[index] if index < len(allocated) else []
            rainy = is_rainy_weather(weather_by_date.get(date))
            days.append(
                PlannerDayDraft(
                    date=date,
                    description=build_day_description(index + 1, day_attractions, rainy),
                    attraction_ids=[item.poi_id or item.name for item in day_attractions],
                    hotel_id=default_hotel_id,
                )
            )
        return PlannerDraft(days=days, overall_suggestions="请提前确认开放时间并预留机动时间。")

    def _build_trip_plan(
        self,
        request: TripRequest,
        draft: PlannerDraft,
        attractions: list[Attraction],
        hotels: list[Hotel],
        weather_info: list[WeatherInfo],
        weather_notes: list[str],
    ) -> TripPlan:
        dates = build_trip_dates(request)
        attraction_map = {attraction.poi_id or attraction.name: attraction for attraction in attractions}
        hotel_map = {hotel.name: hotel for hotel in hotels}
        allocated = allocate_unique_attractions(attractions, len(dates), per_day_max=2)
        draft_by_date = {item.date: item for item in draft.days}

        days: list[DayPlan] = []
        for index, date in enumerate(dates):
            day_draft = draft_by_date.get(date)
            planned_attractions = [
                attraction_map[item_id]
                for item_id in (day_draft.attraction_ids if day_draft else [])
                if item_id in attraction_map
            ]
            if not planned_attractions:
                planned_attractions = allocated[index] if index < len(allocated) else []
            if not planned_attractions and attractions:
                planned_attractions = attractions[:1]

            hotel = hotel_map.get(day_draft.hotel_id if day_draft else "") or (hotels[0] if hotels else None)
            description = (day_draft.description if day_draft else "") or build_day_description(index + 1, planned_attractions, False)
            meals = build_placeholder_meals(request.city)
            days.append(
                DayPlan(
                    date=date,
                    day_index=index + 1,
                    description=description,
                    transportation=request.transportation,
                    accommodation=hotel.name if hotel else request.accommodation,
                    hotel=hotel,
                    attractions=planned_attractions,
                    meals=meals,
                )
            )

        overall = draft.overall_suggestions.strip() if draft.overall_suggestions else ""
        if not overall:
            overall = "；".join(weather_notes[:2]) if weather_notes else "建议合理控制节奏，优先选择已确认开放的景点。"
        budget = estimate_budget_from_plan(days)
        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=weather_info,
            overall_suggestions=overall,
            budget=budget,
        )


class PlanValidator:
    """规则校验与修复模块。"""

    def run(
        self,
        request: TripRequest,
        trip_plan: TripPlan,
        attraction_pool: list[Attraction],
    ) -> tuple[TripPlan, list[str]]:
        alerts: list[str] = []
        weather_by_date = {item.date: item for item in trip_plan.weather_info}
        used_ids: set[str] = set()
        indoor_pool = [item for item in attraction_pool if not is_outdoor_attraction(item)]

        repaired_days: list[DayPlan] = []
        for day in trip_plan.days:
            rainy = is_rainy_weather(weather_by_date.get(day.date))
            day_attractions: list[Attraction] = []
            for attraction in dedup_attractions(day.attractions):
                key = attraction.poi_id or attraction.name
                current = attraction
                if key in used_ids:
                    replacement = pick_replacement(indoor_pool if rainy else attraction_pool, used_ids)
                    if replacement is not None:
                        alerts.append(f"{day.date} 的重复景点 {attraction.name} 已替换为 {replacement.name}")
                        current = replacement
                    else:
                        continue
                if rainy and is_outdoor_attraction(current):
                    replacement = pick_replacement(indoor_pool, used_ids)
                    if replacement is not None:
                        alerts.append(f"{day.date} 雨天已将 {current.name} 替换为 {replacement.name}")
                        current = replacement
                used_ids.add(current.poi_id or current.name)
                day_attractions.append(current)

            if not day_attractions:
                replacement = pick_replacement(indoor_pool if rainy else attraction_pool, used_ids)
                if replacement is not None:
                    alerts.append(f"{day.date} 自动补入 {replacement.name}")
                    used_ids.add(replacement.poi_id or replacement.name)
                    day_attractions.append(replacement)

            hotel = day.hotel
            if hotel and hotel.location and day_attractions and day_attractions[0].location:
                hotel = hotel.model_copy(update={"distance": build_hotel_distance(hotel.location, day_attractions[0].location)})

            repaired_days.append(
                day.model_copy(
                    update={
                        "attractions": day_attractions,
                        "description": build_final_day_description(day.day_index, day_attractions, rainy),
                        "hotel": hotel,
                    }
                )
            )

        repaired_plan = trip_plan.model_copy(update={"days": repaired_days, "budget": estimate_budget_from_plan(repaired_days)})
        return repaired_plan, alerts


class MealService(TripAgentBase):
    """真实餐饮补全模块。"""

    agent_name = "meal_service"

    async def run(self, request: TripRequest, trip_plan: TripPlan) -> TripPlan:
        started = time.perf_counter()
        used_names: set[str] = set()
        updated_days: list[DayPlan] = []
        for day in trip_plan.days:
            meals: list[Meal] = []
            for meal_type, keyword, estimated_cost in MEAL_SPECS:
                meal = await self._find_meal(request, day, used_names, meal_type, keyword, estimated_cost)
                used_names.add(meal.name)
                meals.append(meal)
            updated_days.append(day.model_copy(update={"meals": meals}))

        updated_plan = trip_plan.model_copy(update={"days": updated_days, "budget": estimate_budget_from_plan(updated_days)})
        logger.info(
            "node_done node=find_meals source=mcp count=%d elapsed_ms=%d",
            sum(len(day.meals) for day in updated_days),
            int((time.perf_counter() - started) * 1000),
        )
        return updated_plan

    async def _find_meal(
        self,
        request: TripRequest,
        day: DayPlan,
        used_names: set[str],
        meal_type: str,
        keyword: str,
        estimated_cost: int,
    ) -> Meal:
        location = first_location(day)
        try:
            if location is not None:
                payload = await self.amap.around_search(
                    f"{location.longitude},{location.latitude}",
                    radius=1800,
                    keywords=keyword,
                )
            else:
                payload = await self.amap.text_search(keyword, city=request.city, citylimit=True)
        except AmapMcpError:
            logger.warning("MealService 检索餐饮失败: type=%s city=%s", meal_type, request.city, exc_info=True)
            return default_meal(request.city, meal_type, estimated_cost)

        pois = (payload or {}).get("pois") or []
        meal = pick_meal_from_pois(pois, meal_type, estimated_cost, used_names)
        return meal or default_meal(request.city, meal_type, estimated_cost)


def dedup_list(items: Iterable[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = (item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
        if limit is not None and len(result) >= limit:
            break
    return result


def sanitize_supervisor_keywords(keywords: Sequence[str], city: str, preferences: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    for item in keywords:
        value = (item or "").strip()
        if not value:
            continue
        value = value.replace(city, "").replace(f"{city}市", "").strip(" -·")
        if any(other in value for other in COMMON_CITY_NAMES if other not in {city, f"{city}市"}):
            continue
        cleaned.append(value)

    preference_map = {
        "历史文化": ["博物馆", "古迹", "老街"],
        "自然风光": ["公园", "湖", "风景区"],
        "亲子": ["动物园", "公园", "乐园"],
        "美食": ["老街", "步行街"],
        "拍照": ["地标", "湖", "公园"],
    }
    for pref in preferences:
        cleaned.extend(preference_map.get(pref, []))
    cleaned.extend(["热门景点", "博物馆", "公园"])
    return dedup_list(cleaned, limit=5)


def build_trip_dates(request: TripRequest) -> list[str]:
    start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
    return [(start + timedelta(days=index)).isoformat() for index in range(request.travel_days)]


def parse_weather_payload(payload: dict[str, Any]) -> list[WeatherInfo]:
    casts = extract_weather_entries(payload)

    result: list[WeatherInfo] = []
    for item in casts:
        result.append(
            WeatherInfo(
                date=str(item.get("date") or ""),
                day_weather=str(item.get("dayweather") or item.get("day_weather") or ""),
                night_weather=str(item.get("nightweather") or item.get("night_weather") or ""),
                day_temp=item.get("daytemp") or item.get("day_temp") or 0,
                night_temp=item.get("nighttemp") or item.get("night_temp") or 0,
                wind_direction=str(item.get("daywind") or item.get("winddirection") or ""),
                wind_power=str(item.get("daypower") or item.get("windpower") or ""),
            )
        )
    if not result and (payload.get("forecasts") or payload.get("casts")):
        logger.warning("天气解析结果为空，可能是上游返回结构发生变化: keys=%s", list(payload.keys()))
    return result


def extract_weather_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """兼容高德 MCP 不同版本的天气返回结构。"""
    forecasts = payload.get("forecasts") or []
    if isinstance(forecasts, list) and forecasts:
        first = forecasts[0]
        if isinstance(first, dict) and "date" in first:
            return [item for item in forecasts if isinstance(item, dict)]
        if isinstance(first, dict) and isinstance(first.get("casts"), list):
            return [item for item in first.get("casts") or [] if isinstance(item, dict)]

    casts = payload.get("casts") or []
    if isinstance(casts, list):
        return [item for item in casts if isinstance(item, dict)]
    return []


def align_weather_to_trip_dates(weather: list[WeatherInfo], request: TripRequest) -> list[WeatherInfo]:
    by_date = {item.date: item for item in weather}
    aligned: list[WeatherInfo] = []
    for date in build_trip_dates(request):
        aligned.append(by_date.get(date) or WeatherInfo(date=date))
    return aligned


def build_weather_notes(weather_info: list[WeatherInfo]) -> list[str]:
    notes: list[str] = []
    for item in weather_info:
        if is_rainy_weather(item):
            notes.append(f"{item.date} 可能有降雨，建议优先安排室内景点。")
            break
    if weather_info:
        temps = [safe_int(item.day_temp) for item in weather_info]
        if temps and max(temps) - min(temps) >= 8:
            notes.append("昼夜温差较明显，建议携带薄外套。")
    if not notes:
        notes.append("天气整体平稳，建议按计划顺路游览。")
    return dedup_list(notes, limit=2)


async def enrich_pois_with_details(amap: Any, pois: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """为轻量检索结果补齐详情字段，避免 location/type 为空。"""
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    detail_ids: list[str] = []
    raw_by_id: dict[str, dict[str, Any]] = {}

    for poi in pois:
        poi_id = safe_str(poi.get("id"))
        if not poi_id or poi_id in seen_ids:
            continue
        seen_ids.add(poi_id)
        raw_by_id[poi_id] = dict(poi)
        detail_ids.append(poi_id)
        if len(detail_ids) >= limit:
            break

    detail_results = await asyncio.gather(
        *(amap.search_detail(poi_id) for poi_id in detail_ids),
        return_exceptions=True,
    )
    for poi_id, detail in zip(detail_ids, detail_results):
        base = raw_by_id[poi_id]
        if isinstance(detail, Exception):
            merged.append(base)
            continue
        merged.append({**base, **(detail or {})})
    return merged


def build_attraction_search_specs(request: TripRequest, plan: SupervisorPlan) -> list[str]:
    return dedup_list([*(plan.attraction_keywords or []), *request.preferences, "热门景点", "博物馆", "公园"], limit=5)


def parse_attraction_candidate(poi: dict[str, Any] | None) -> AttractionCandidate | None:
    if not poi:
        return None
    name = safe_str(poi.get("name"))
    if not name or is_sub_attraction_poi(poi) or not is_attraction_poi(poi):
        return None
    location = parse_location(poi.get("location"))
    if location is None:
        return None
    category = extract_category(poi)
    rating = safe_float(poi.get("biz_ext", {}).get("rating") if isinstance(poi.get("biz_ext"), dict) else poi.get("rating"))
    return AttractionCandidate(
        id=safe_str(poi.get("id")) or name,
        name=name,
        address=safe_str(poi.get("address")) or "地址待确认",
        location=location,
        category=category,
        rating=rating,
        photos=extract_photo_urls(poi),
        ticket_price=guess_ticket_price(category),
        visit_duration=guess_visit_duration(category),
        is_outdoor=is_outdoor_category(category),
        description=build_attraction_description(name, category, rating),
    )


def parse_hotel_candidate(poi: dict[str, Any] | None, accommodation: str) -> HotelCandidate | None:
    if not poi or not is_hotel_poi(poi):
        return None
    name = safe_str(poi.get("name"))
    if not name:
        return None
    return HotelCandidate(
        id=safe_str(poi.get("id")) or name,
        name=name,
        address=safe_str(poi.get("address")) or "地址待确认",
        location=parse_location(poi.get("location")),
        hotel_type=extract_category(poi) or accommodation,
        rating=safe_float(poi.get("biz_ext", {}).get("rating") if isinstance(poi.get("biz_ext"), dict) else poi.get("rating")),
        estimated_cost=guess_hotel_cost(accommodation, name),
        price_range=build_hotel_price_range(guess_hotel_cost(accommodation, name)),
    )


def rank_attraction_candidates(candidates: list[AttractionCandidate], request: TripRequest) -> list[AttractionCandidate]:
    pref_text = " ".join(request.preferences)
    return sorted(
        candidates,
        key=lambda item: (
            preference_bonus(item.category, pref_text),
            1 if not item.is_outdoor else 0,
            item.rating or 0,
            -item.ticket_price,
        ),
        reverse=True,
    )


def rank_hotel_candidates(candidates: list[HotelCandidate], request: TripRequest) -> list[HotelCandidate]:
    want_high_end = any(token in request.accommodation for token in ("豪华", "高端", "舒适", "星级"))
    return sorted(
        candidates,
        key=lambda item: (
            item.rating or 0,
            -abs(item.estimated_cost - (700 if want_high_end else 350)),
        ),
        reverse=True,
    )


def dedup_attraction_candidates(items: Iterable[AttractionCandidate | None]) -> list[AttractionCandidate]:
    by_name: dict[str, AttractionCandidate] = {}
    for item in items:
        if item is None:
            continue
        key = normalize_attraction_name(item.name)
        current = by_name.get(key)
        if current is None or (item.rating or 0) > (current.rating or 0):
            by_name[key] = item
    return list(by_name.values())


def dedup_hotel_candidates(items: Iterable[HotelCandidate | None]) -> list[HotelCandidate]:
    by_name: dict[str, HotelCandidate] = {}
    for item in items:
        if item is None:
            continue
        key = item.name.strip()
        current = by_name.get(key)
        if current is None or (item.rating or 0) > (current.rating or 0):
            by_name[key] = item
    return list(by_name.values())


def candidate_to_attraction(item: AttractionCandidate) -> Attraction:
    return Attraction(
        name=item.name,
        address=item.address,
        location=item.location,
        visit_duration=item.visit_duration,
        description=item.description,
        category=item.category,
        rating=item.rating,
        photos=item.photos,
        poi_id=item.id,
        image_url=item.photos[0] if item.photos else None,
        ticket_price=item.ticket_price,
    )


def candidate_to_hotel(item: HotelCandidate) -> Hotel:
    return Hotel(
        name=item.name,
        address=item.address,
        location=item.location,
        price_range=item.price_range,
        rating=(f"{item.rating:.1f}" if item.rating is not None else ""),
        distance="",
        type=item.hotel_type,
        estimated_cost=item.estimated_cost,
    )


def build_placeholder_meals(city: str) -> list[Meal]:
    return [default_meal(city, meal_type, cost) for meal_type, _, cost in MEAL_SPECS]


def allocate_unique_attractions(attractions: list[Attraction], day_count: int, per_day_max: int) -> list[list[Attraction]]:
    result: list[list[Attraction]] = [[] for _ in range(day_count)]
    for index, attraction in enumerate(attractions):
        result[index % max(day_count, 1)].append(attraction)
    return [day[:per_day_max] for day in result]


def dedup_attractions(attractions: Sequence[Attraction]) -> list[Attraction]:
    seen: set[str] = set()
    result: list[Attraction] = []
    for attraction in attractions:
        key = attraction.poi_id or attraction.name
        if key in seen:
            continue
        seen.add(key)
        result.append(attraction)
    return result


def pick_replacement(pool: Sequence[Attraction], used_ids: set[str]) -> Attraction | None:
    for item in pool:
        key = item.poi_id or item.name
        if key not in used_ids:
            return item
    return None


def build_day_description(day_index: int, attractions: Sequence[Attraction], rainy: bool) -> str:
    if attractions:
        names = "、".join(item.name for item in attractions[:2])
        prefix = "室内优先" if rainy else "顺路游览"
        return f"第{day_index}天以{prefix}为主，安排 {names}。"
    return f"第{day_index}天安排轻松游览，预留机动时间。"


def build_final_day_description(day_index: int, attractions: Sequence[Attraction], rainy: bool) -> str:
    if attractions:
        names = "、".join(item.name for item in attractions[:2])
        suffix = "，雨天已优先保留室内点位。" if rainy else "，整体节奏保持轻松。"
        return f"第{day_index}天游览 {names}{suffix}"
    return f"第{day_index}天建议以休整和自由活动为主。"


def estimate_budget_from_plan(days: Sequence[DayPlan]) -> Budget:
    attraction_total = sum(sum(max(0, attraction.ticket_price) for attraction in day.attractions) for day in days)
    hotel_total = sum((day.hotel.estimated_cost if day.hotel else 0) for day in days)
    meal_total = sum(sum(max(0, meal.estimated_cost) for meal in day.meals) for day in days)
    transportation_total = estimate_transport_cost(days)
    return Budget(
        total_attractions=attraction_total,
        total_hotels=hotel_total,
        total_meals=meal_total,
        total_transportation=transportation_total,
        total=attraction_total + hotel_total + meal_total + transportation_total,
    )


def estimate_transport_cost(days: Sequence[DayPlan]) -> int:
    total_points = sum(max(len(day.attractions), 1) for day in days)
    return total_points * 20


def default_meal(city: str, meal_type: str, estimated_cost: int) -> Meal:
    label = {"breakfast": "早餐", "lunch": "午餐", "dinner": "晚餐"}.get(meal_type, "餐饮")
    return Meal(
        type=meal_type,
        name=f"{city}{label}待补充",
        address=None,
        location=None,
        description=f"暂未检索到合适的{label}，建议在景点周边灵活安排。",
        estimated_cost=estimated_cost,
    )


def pick_meal_from_pois(
    pois: Sequence[dict[str, Any]],
    meal_type: str,
    estimated_cost: int,
    used_names: set[str],
) -> Meal | None:
    for poi in pois:
        name = safe_str(poi.get("name"))
        if not name or name in used_names:
            continue
        return Meal(
            type=meal_type,
            name=name,
            address=safe_str(poi.get("address")) or None,
            location=parse_location(poi.get("location")),
            description="高德周边真实检索结果，适合作为当餐选择。",
            estimated_cost=estimated_cost,
        )
    return None


def first_location(day: DayPlan) -> Location | None:
    if day.attractions and day.attractions[0].location:
        return day.attractions[0].location
    if day.hotel and day.hotel.location:
        return day.hotel.location
    return None


def build_hotel_distance(hotel_location: Location, attraction_location: Location) -> str:
    distance_km = haversine_km(hotel_location, attraction_location)
    return f"距当日景点约 {distance_km:.1f} km"


def haversine_km(a: Location, b: Location) -> float:
    lon1, lat1 = math.radians(a.longitude), math.radians(a.latitude)
    lon2, lat2 = math.radians(b.longitude), math.radians(b.latitude)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(hav))


def safe_str(value: Any) -> str:
    return str(value or "").strip()


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def parse_location(value: Any) -> Location | None:
    if not value:
        return None
    if isinstance(value, str) and "," in value:
        lng, lat = value.split(",", 1)
        try:
            return Location(longitude=float(lng), latitude=float(lat))
        except ValueError:
            return None
    if isinstance(value, dict):
        try:
            return Location(longitude=float(value.get("longitude")), latitude=float(value.get("latitude")))
        except (TypeError, ValueError):
            return None
    return None


def extract_category(poi: dict[str, Any]) -> str:
    raw = safe_str(poi.get("type") or poi.get("type_name") or poi.get("category"))
    if not raw:
        return "景点"
    return raw.split(";")[0]


def extract_photo_urls(poi: dict[str, Any]) -> list[str]:
    photos = poi.get("photos") or []
    result: list[str] = []
    if isinstance(photos, dict):
        url = safe_str(photos.get("url"))
        if url:
            result.append(url)
        return dedup_list(result, limit=3)
    if isinstance(photos, list):
        for item in photos:
            if isinstance(item, dict):
                url = safe_str(item.get("url"))
            else:
                url = safe_str(item)
            if url:
                result.append(url)
    return dedup_list(result, limit=3)


def is_attraction_poi(poi: dict[str, Any]) -> bool:
    category = extract_category(poi)
    keywords = (category + safe_str(poi.get("name"))).lower()
    return any(token in keywords for token in ("景", "博物馆", "公园", "古迹", "广场", "湖", "山"))


def is_sub_attraction_poi(poi: dict[str, Any]) -> bool:
    name = safe_str(poi.get("name"))
    address = safe_str(poi.get("address"))
    if any(token in address for token in ("景区内", "公园内", "园内", "博物院内", "馆内")):
        return True
    if re.search(r"[-·•]", name):
        return True
    if any(name.endswith(suffix) for suffix in ("东门", "西门", "南门", "北门", "游客中心", "停车场", "码头", "服务区")):
        return True
    return False


def is_hotel_poi(poi: dict[str, Any]) -> bool:
    category = extract_category(poi)
    name = safe_str(poi.get("name"))
    text = f"{category}{name}"
    return any(token in text for token in ("酒店", "宾馆", "旅馆", "民宿", "客栈"))


def normalize_attraction_name(name: str) -> str:
    value = safe_str(name)
    value = re.split(r"[-·•]", value)[0]
    return value.strip()


def preference_bonus(category: str, pref_text: str) -> int:
    score = 0
    if "博物馆" in category and "历史" in pref_text:
        score += 3
    if any(token in category for token in ("公园", "湖", "山", "风景")) and "自然" in pref_text:
        score += 3
    if any(token in category for token in ("公园", "乐园")) and "亲子" in pref_text:
        score += 2
    return score


def is_outdoor_category(category: str) -> bool:
    return any(token in category for token in ("公园", "湖", "山", "风景", "广场", "古镇", "景区"))


def is_outdoor_attraction(attraction: Attraction | None) -> bool:
    if attraction is None:
        return False
    return is_outdoor_category(attraction.category or "")


def is_rainy_weather(item: WeatherInfo | None) -> bool:
    if item is None:
        return False
    text = f"{item.day_weather}{item.night_weather}"
    return any(token in text for token in ("雨", "雪", "雷", "冰雹"))


def guess_ticket_price(category: str) -> int:
    if any(token in category for token in ("博物馆", "纪念馆", "公园")):
        return 0
    if any(token in category for token in ("山", "湖", "风景")):
        return 40
    return 20


def guess_visit_duration(category: str) -> int:
    if "博物馆" in category:
        return 180
    if any(token in category for token in ("山", "湖", "风景")):
        return 150
    return 120


def build_attraction_description(name: str, category: str, rating: float | None) -> str:
    parts = [name, f"类型为{category}"]
    if rating is not None:
        parts.append(f"评分约 {rating:.1f}")
    parts.append("来自高德真实检索结果")
    return "，".join(parts) + "。"


def guess_hotel_cost(accommodation: str, name: str) -> int:
    text = f"{accommodation}{name}"
    if any(token in text for token in ("豪华", "高端", "皇冠", "希尔顿", "万豪", "星级")):
        return 900
    if any(token in text for token in ("舒适", "商务", "假日", "全季", "亚朵")):
        return 500
    return 280


def build_hotel_price_range(cost: int) -> str:
    return f"¥{max(0, cost - 60)}-¥{cost + 80}/晚"


__all__ = [
    "SupervisorPlan",
    "SupervisorAgent",
    "AttractionAgent",
    "WeatherAgent",
    "HotelAgent",
    "PlannerAgent",
    "PlanValidator",
    "MealService",
]
