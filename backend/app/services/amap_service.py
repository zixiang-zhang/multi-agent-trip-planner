"""高德地图服务层。

该层负责把 MCP 原始返回结果转换成项目内部的 Pydantic 数据模型，
供 FastAPI 路由和多 agent 工作流共同使用。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ..agents.langgraph_agents import parse_weather_payload
from ..models.schemas import Location, POIInfo, RouteInfo, WeatherInfo
from ..tools.amap_mcp_tools import AmapMcpError, get_amap_mcp_client

logger = logging.getLogger(__name__)


class AmapService:
    """高德地图业务服务。"""

    def __init__(self) -> None:
        self.client = get_amap_mcp_client()

    async def search_poi(self, keywords: str, city: str, citylimit: bool = True) -> List[POIInfo]:
        """搜索 POI，并补齐坐标。"""
        search_data = await self.client.text_search(
            keywords=keywords,
            city=city if citylimit else None,
            citylimit=citylimit,
        )
        pois = search_data.get("pois") or []
        if not pois:
            return []

        results: List[POIInfo] = []
        for item in pois[:10]:
            poi_id = str(item.get("id") or "").strip()
            if not poi_id:
                continue
            detail = await self.client.search_detail(poi_id)
            location = self._parse_location(detail.get("location"))
            if location is None:
                continue
            results.append(
                POIInfo(
                    id=poi_id,
                    name=str(detail.get("name") or item.get("name") or "").strip(),
                    type=str(detail.get("type") or "").strip(),
                    address=str(detail.get("address") or item.get("address") or "").strip(),
                    location=location,
                    tel=self._extract_phone(detail),
                )
            )
        return results

    async def get_weather(self, city: str) -> List[WeatherInfo]:
        """查询天气并转换成前端可直接消费的结构。"""
        weather_data = await self.client.weather(city)
        return parse_weather_payload(weather_data)

    async def plan_route(
        self,
        origin_address: str,
        destination_address: str,
        origin_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        route_type: str = "walking",
    ) -> Optional[RouteInfo]:
        """根据地址做路径规划。"""
        origin = await self._geocode_location(origin_address, origin_city)
        destination = await self._geocode_location(destination_address, destination_city)
        if origin is None or destination is None:
            return None

        origin_text = f"{origin.longitude},{origin.latitude}"
        destination_text = f"{destination.longitude},{destination.latitude}"

        if route_type == "driving":
            route_data = await self.client.driving_route(origin_text, destination_text)
        elif route_type == "transit":
            route_data = await self.client.transit_route(
                origin_text,
                destination_text,
                origin_city or destination_city or "",
                destination_city or origin_city or "",
            )
        else:
            route_data = await self.client.walking_route(origin_text, destination_text)

        distance, duration = self._extract_route_metrics(route_data)
        return RouteInfo(
            distance=distance,
            duration=duration,
            route_type=route_type,
            description=self._build_route_description(route_type, distance, duration),
        )

    async def geocode(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        """公开 geocode 能力。"""
        return await self._geocode_location(address, city)

    async def get_poi_detail(self, poi_id: str) -> Dict[str, Any]:
        """获取 POI 详情。"""
        return await self.client.search_detail(poi_id)

    async def _geocode_location(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        """地址转坐标。"""
        try:
            data = await self.client.geocode(address, city)
        except AmapMcpError:
            logger.warning("地址转坐标失败: address=%s city=%s", address, city, exc_info=True)
            return None

        geocodes = data.get("geocodes") or []
        if not geocodes:
            return None
        return self._parse_location(geocodes[0].get("location"))

    def _parse_location(self, value: Any) -> Optional[Location]:
        """解析经纬度字符串。"""
        if value is None:
            return None
        if isinstance(value, Location):
            return value
        if isinstance(value, dict):
            lng = self._safe_float(value.get("longitude") or value.get("lng"))
            lat = self._safe_float(value.get("latitude") or value.get("lat"))
            if lng is None or lat is None:
                return None
            return Location(longitude=lng, latitude=lat)

        text = str(value).strip()
        if not text:
            return None
        match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", text)
        if not match:
            return None
        return Location(longitude=float(match.group(1)), latitude=float(match.group(2)))

    def _extract_phone(self, detail: Dict[str, Any]) -> Optional[str]:
        """提取联系电话。"""
        tel = detail.get("tel")
        if isinstance(tel, list):
            tel = " / ".join(str(item).strip() for item in tel if str(item).strip())
        value = str(tel or "").strip()
        return value or None

    def _join_wind(self, day_value: Any, night_value: Any) -> str:
        """拼接白天 / 夜间风向或风力。"""
        day = str(day_value or "").strip()
        night = str(night_value or "").strip()
        if day and night and day != night:
            return f"{day}/{night}"
        return day or night

    def _extract_route_metrics(self, route_data: Dict[str, Any]) -> tuple[float, int]:
        """尽量从高德路径规划结果中提取距离和时长。"""
        route = route_data.get("route") or {}
        if route.get("paths"):
            first = route["paths"][0]
            return (
                float(first.get("distance") or 0.0),
                int(float(first.get("duration") or 0.0)),
            )
        if route.get("transits"):
            first = route["transits"][0]
            return (
                float(first.get("distance") or 0.0),
                int(float(first.get("duration") or 0.0)),
            )
        return 0.0, 0

    def _build_route_description(self, route_type: str, distance: float, duration: int) -> str:
        """生成简洁路径描述。"""
        distance_km = distance / 1000 if distance > 0 else 0
        duration_min = max(1, int(duration / 60)) if duration > 0 else 0
        label = {
            "walking": "步行",
            "driving": "驾车",
            "transit": "公交",
        }.get(route_type, route_type)
        return f"{label}约 {distance_km:.1f} km，耗时约 {duration_min} 分钟"

    def _safe_float(self, value: Any) -> Optional[float]:
        """安全转换浮点数。"""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


_amap_service: Optional[AmapService] = None


def get_amap_service() -> AmapService:
    """获取单例高德服务。"""
    global _amap_service
    if _amap_service is None:
        _amap_service = AmapService()
    return _amap_service
