"""高德地图 MCP 客户端。

这里不再把异步 MCP 工具包装成同步工具，而是直接维护一个共享的异步 MCP 会话。
这样可以让 FastAPI / LangGraph 主链路保持异步，方便并发检索景点、天气和酒店。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ..config import get_settings

logger = logging.getLogger(__name__)

# 当前高德官方 MCP Server 的 npm 包名。
DEFAULT_AMAP_MCP_PACKAGE = os.getenv("AMAP_MCP_PACKAGE", "@amap/amap-maps-mcp-server")
AMAP_DISTRICT_API = "https://restapi.amap.com/v3/config/district"
CITY_ADCODE_HINTS: Dict[str, str] = {
    "北京": "110000",
    "北京市": "110000",
    "上海": "310000",
    "上海市": "310000",
    "广州": "440100",
    "广州市": "440100",
    "深圳": "440300",
    "深圳市": "440300",
    "杭州": "330100",
    "杭州市": "330100",
    "苏州": "320500",
    "苏州市": "320500",
    "济南": "370100",
    "济南市": "370100",
    "济宁": "370800",
    "济宁市": "370800",
    "南京": "320100",
    "南京市": "320100",
    "成都": "510100",
    "成都市": "510100",
    "重庆": "500000",
    "重庆市": "500000",
    "西安": "610100",
    "西安市": "610100",
    "武汉": "420100",
    "武汉市": "420100",
    "长沙": "430100",
    "长沙市": "430100",
    "洛阳": "410300",
    "洛阳市": "410300",
    "开封": "410200",
    "开封市": "410200",
    "厦门": "350200",
    "厦门市": "350200",
    "青岛": "370200",
    "青岛市": "370200",
    "昆明": "530100",
    "昆明市": "530100",
    "大理": "532900",
    "大理州": "532900",
    "丽江": "530700",
    "丽江市": "530700",
    "三亚": "460200",
    "三亚市": "460200",
    "哈尔滨": "230100",
    "哈尔滨市": "230100",
}


class AmapMcpError(RuntimeError):
    """高德 MCP 调用异常。"""


@dataclass(slots=True)
class AmapToolMetadata:
    """高德 MCP 工具的元数据。"""

    name: str
    description: str
    input_schema: Dict[str, Any]


class AmapMcpClient:
    """共享的高德 MCP 客户端。

    设计目标：
    1. 整个应用进程只维护一个 MCP 会话，避免重复拉起 server。
    2. 所有工具调用都走异步接口，便于在 LangGraph 中并发执行。
    3. 对外只暴露业务真正会用到的高德能力，调用失败时抛出清晰错误。
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._startup_lock = asyncio.Lock()
        self._rate_limit_lock = asyncio.Lock()
        self._call_semaphore = asyncio.Semaphore(
            max(1, int(os.getenv("AMAP_MCP_MAX_CONCURRENCY", "3")))
        )
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None
        self._tool_map: Dict[str, AmapToolMetadata] = {}
        self._city_adcode_cache: Dict[str, str] = {}
        self._result_cache: Dict[str, Any] = {}
        self._last_call_at: float = 0.0
        self._min_call_interval_seconds = float(os.getenv("AMAP_MCP_MIN_INTERVAL_SECONDS", "0.25"))
        self._max_retries = 3

    @property
    def tool_names(self) -> List[str]:
        """返回已加载的工具名称。"""
        return sorted(self._tool_map.keys())

    async def startup(self) -> None:
        """启动共享 MCP 会话。"""
        if self._session is not None:
            return

        async with self._startup_lock:
            if self._session is not None:
                return

            if not self._settings.amap_api_key:
                raise AmapMcpError("AMAP_API_KEY 未配置，无法启动高德 MCP 客户端")

            logger.info("启动高德 MCP 会话: package=%s", DEFAULT_AMAP_MCP_PACKAGE)

            stack = AsyncExitStack()
            try:
                params = StdioServerParameters(
                    command=self._pick_launcher_command(),
                    args=self._pick_launcher_args(),
                    env={"AMAP_MAPS_API_KEY": self._settings.amap_api_key},
                    encoding="utf-8",
                    encoding_error_handler="replace",
                )
                read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                await session.initialize()

                tool_result = await session.list_tools()
                self._tool_map = {
                    tool.name: AmapToolMetadata(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema or {},
                    )
                    for tool in tool_result.tools
                }

                self._exit_stack = stack
                self._session = session
                logger.info("高德 MCP 会话已就绪，工具数量=%d", len(self._tool_map))
            except Exception:
                await stack.aclose()
                logger.exception("高德 MCP 会话启动失败")
                raise

    async def shutdown(self) -> None:
        """关闭共享 MCP 会话。"""
        if self._exit_stack is None:
            return

        logger.info("关闭高德 MCP 会话")
        try:
            await self._exit_stack.aclose()
        except RuntimeError:
            # 某些测试场景下，stdio_client 会因为进入/退出任务不同而报错。
            # 这里做容错收尾，避免影响主流程结果与评测报告生成。
            logger.warning("高德 MCP 会话关闭时出现任务上下文异常，已强制重置客户端状态", exc_info=True)
        finally:
            self._exit_stack = None
            self._session = None
            self._tool_map = {}
            self._result_cache = {}

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """调用指定 MCP 工具并自动解析返回结果。"""
        await self.startup()

        if self._session is None:
            raise AmapMcpError("高德 MCP 会话尚未初始化")

        if name not in self._tool_map:
            raise AmapMcpError(f"高德 MCP 工具不存在: {name}")

        payload = arguments or {}
        cache_key = self._build_cache_key(name, payload)
        if cache_key in self._result_cache:
            return self._result_cache[cache_key]

        last_error = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                async with self._call_semaphore:
                    await self._respect_rate_limit()
                    result = await self._session.call_tool(name, payload)
                if result.isError:
                    error_text = self._extract_text_from_content(result.content) or f"{name} 调用失败"
                    if self._should_retry(error_text, attempt):
                        last_error = error_text
                        await asyncio.sleep(self._retry_delay_seconds(attempt))
                        continue
                    raise AmapMcpError(error_text)

                decoded = self._decode_content(result.content)
                self._result_cache[cache_key] = decoded
                return decoded
            except AmapMcpError:
                raise
            except Exception as exc:
                error_text = str(exc)
                if self._should_retry(error_text, attempt):
                    last_error = error_text
                    await asyncio.sleep(self._retry_delay_seconds(attempt))
                    continue
                raise AmapMcpError(error_text) from exc

        raise AmapMcpError(last_error or f"{name} 调用失败")

    async def text_search(
        self,
        keywords: str,
        city: str | None = None,
        types: str | None = None,
        citylimit: bool = True,
    ) -> Dict[str, Any]:
        """关键词搜索 POI。"""
        payload: Dict[str, Any] = {"keywords": keywords}
        if city:
            payload["city"] = city
        if types:
            payload["types"] = types
        payload["citylimit"] = citylimit
        result = await self.call_tool("maps_text_search", payload)
        return result if isinstance(result, dict) else {}

    async def around_search(
        self,
        location: str,
        radius: int = 2000,
        keywords: str | None = None,
    ) -> Dict[str, Any]:
        """按经纬度做周边搜索。"""
        payload: Dict[str, Any] = {"location": location, "radius": str(radius)}
        if keywords:
            payload["keywords"] = keywords
        result = await self.call_tool("maps_around_search", payload)
        return result if isinstance(result, dict) else {}

    async def search_detail(self, poi_id: str) -> Dict[str, Any]:
        """查询 POI 详情。"""
        result = await self.call_tool("maps_search_detail", {"id": poi_id})
        return result if isinstance(result, dict) else {}

    async def geocode(self, address: str, city: str | None = None) -> Dict[str, Any]:
        """地址转经纬度。"""
        payload: Dict[str, Any] = {"address": address}
        if city:
            payload["city"] = city
        result = await self.call_tool("maps_geo", payload)
        return result if isinstance(result, dict) else {}

    async def weather(self, city: str) -> Dict[str, Any]:
        """查询天气。

        高德 MCP 的天气工具对中文城市名支持不稳定，因此先将城市解析为 adcode，
        再使用 MCP 工具查询天气，保证天气数据依然来自真实 MCP 调用。
        """
        adcode = await self.resolve_city_adcode(city)
        query_city = adcode or city
        result = await self.call_tool("maps_weather", {"city": query_city})
        return result if isinstance(result, dict) else {}

    async def walking_route(self, origin: str, destination: str) -> Dict[str, Any]:
        """步行路径规划。"""
        result = await self.call_tool(
            "maps_direction_walking",
            {"origin": origin, "destination": destination},
        )
        return result if isinstance(result, dict) else {}

    async def driving_route(self, origin: str, destination: str) -> Dict[str, Any]:
        """驾车路径规划。"""
        result = await self.call_tool(
            "maps_direction_driving",
            {"origin": origin, "destination": destination},
        )
        return result if isinstance(result, dict) else {}

    async def transit_route(self, origin: str, destination: str, city: str, cityd: str) -> Dict[str, Any]:
        """公交综合路径规划。"""
        result = await self.call_tool(
            "maps_direction_transit_integrated",
            {"origin": origin, "destination": destination, "city": city, "cityd": cityd},
        )
        return result if isinstance(result, dict) else {}

    async def resolve_city_adcode(self, city: str) -> str | None:
        """解析城市 adcode。

        这里的目标是给 MCP 天气工具提供稳定输入，因此允许用高德 REST district 接口
        做 adcode 解析兜底，但真正的天气数据仍然通过 MCP 获取。
        """
        city = (city or "").strip()
        if not city:
            return None

        variants = self._city_variants(city)
        for variant in variants:
            hinted = CITY_ADCODE_HINTS.get(variant)
            if hinted:
                for variant_name in variants:
                    self._city_adcode_cache[variant_name] = hinted
                return hinted

            cached = self._city_adcode_cache.get(variant)
            if cached:
                return cached

        async with httpx.AsyncClient(timeout=10.0) as client:
            for keyword in variants:
                try:
                    resp = await client.get(
                        AMAP_DISTRICT_API,
                        params={
                            "keywords": keyword,
                            "subdistrict": 0,
                            "extensions": "base",
                            "key": self._settings.amap_api_key,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except Exception:
                    logger.warning("解析城市 adcode 失败: city=%s", keyword, exc_info=True)
                    continue

                districts = data.get("districts") or []
                for district in districts:
                    adcode = str(district.get("adcode") or "").strip()
                    if adcode:
                        for variant_name in variants:
                            self._city_adcode_cache[variant_name] = adcode
                        return adcode

        return None

    async def _respect_rate_limit(self) -> None:
        """在共享会话层做轻量节流，减少 QPS 超限。"""
        async with self._rate_limit_lock:
            now = time.monotonic()
            wait_seconds = self._min_call_interval_seconds - (now - self._last_call_at)
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            self._last_call_at = time.monotonic()

    def _should_retry(self, error_text: str, attempt: int) -> bool:
        """判断当前错误是否值得重试。"""
        if attempt >= self._max_retries:
            return False
        text = (error_text or "").upper()
        retry_terms = (
            "CUQPS_HAS_EXCEEDED_THE_LIMIT",
            "EXCEEDED_THE_LIMIT",
            "TIMEOUT",
            "TIMED OUT",
            "ETIMEDOUT",
            "ECONNABORTED",
            "ECONNRESET",
            "ENOTFOUND",
            "EAI_AGAIN",
            "429",
            "502",
            "503",
            "504",
        )
        return any(term in text for term in retry_terms)

    def _retry_delay_seconds(self, attempt: int) -> float:
        """简单指数退避。"""
        return 0.6 * attempt

    def _build_cache_key(self, name: str, payload: Dict[str, Any]) -> str:
        """构造工具调用缓存键。"""
        return f"{name}:{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"

    def _pick_launcher_command(self) -> str:
        """优先使用 npx 启动官方高德 MCP Server。"""
        return os.getenv("AMAP_MCP_LAUNCHER", "npx")

    def _pick_launcher_args(self) -> List[str]:
        """返回 MCP server 的启动参数。"""
        return ["-y", DEFAULT_AMAP_MCP_PACKAGE]

    def _decode_content(self, content: List[Any]) -> Any:
        """解析 MCP 工具返回的 text/json 内容。"""
        raw_text = self._extract_text_from_content(content)
        if not raw_text:
            return {}

        raw_text = raw_text.strip()
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return raw_text

    def _extract_text_from_content(self, content: List[Any]) -> str:
        """把 MCP Content 列表拼成字符串。"""
        parts: List[str] = []
        for item in content or []:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text)
        return "\n".join(parts).strip()

    def _city_variants(self, city: str) -> List[str]:
        """生成城市名称的常见变体，提升 adcode 解析命中率。"""
        raw = city.strip()
        normalized = raw
        for suffix in ("市", "地区", "自治州", "盟"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break

        candidates = [raw, normalized]
        if normalized:
            candidates.append(f"{normalized}市")

        dedup: List[str] = []
        seen: set[str] = set()
        for item in candidates:
            value = item.strip()
            if value and value not in seen:
                dedup.append(value)
                seen.add(value)
        return dedup


_amap_mcp_client: Optional[AmapMcpClient] = None


def get_amap_mcp_client() -> AmapMcpClient:
    """获取全局共享的高德 MCP 客户端。"""
    global _amap_mcp_client
    if _amap_mcp_client is None:
        _amap_mcp_client = AmapMcpClient()
    return _amap_mcp_client
