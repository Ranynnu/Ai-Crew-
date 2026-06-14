"""
============================================================
 weather_tool.py — 天气查询工具
 调用 Open-Meteo 免费天气 API（无需 API Key）
 支持同步（_run）和异步（_arun）两种模式
============================================================
"""
import httpx                          # 同时支持同步和异步 HTTP 请求
from crewai.tools import BaseTool     # CrewAI 工具基类


class WeatherTool(BaseTool):
    """查询指定城市当前天气的工具。
    Agent 调用时会传入 city 参数，返回中文天气描述。
    """

    name: str = "Weather Tool"
    description: str = "查询指定城市的当前天气（使用真实API）"

    # ==================== 同步入口（兜底路径） ====================
    def _run(self, city: str) -> str:
        """同步查询天气。CrewAI 在 kickoff() 时调用此方法。
        使用 httpx.Client（替代 requests 库，减少依赖）。
        """
        try:
            with httpx.Client(timeout=10) as client:
                # ---- 步骤1：城市名 → 经纬度（地理编码） ----
                geo_resp = client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={
                        "name": city, "count": 1,
                        "language": "zh", "format": "json",
                    },
                )
                geo_resp.raise_for_status()
                geo_data = geo_resp.json()

                if not geo_data.get("results"):
                    return f"未找到城市 '{city}'，请检查城市名称。"

                # 提取经纬度和官方城市名
                latitude = geo_data["results"][0]["latitude"]
                longitude = geo_data["results"][0]["longitude"]
                official_city_name = geo_data["results"][0]["name"]

                # ---- 步骤2：经纬度 → 实时天气 ----
                weather_resp = client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": latitude,
                        "longitude": longitude,
                        "current_weather": True,
                        "timezone": "auto",
                        "temperature_unit": "celsius",
                    },
                )
                weather_resp.raise_for_status()
                weather_data = weather_resp.json()

                if "current_weather" in weather_data:
                    return self._format_weather(official_city_name, weather_data)
                return "获取天气数据失败，未返回有效信息。"

        except httpx.HTTPError as e:
            return f"网络请求错误：{str(e)}"
        except Exception as e:
            return f"处理天气数据时发生未知错误：{str(e)}"

    # ==================== 异步入口（Flow kickoff_async 调用此方法） ====================
    async def _arun(self, city: str) -> str:
        """异步查询天气。Flow 的 kickoff_async() 会优先调用此方法，
        避免阻塞 asyncio 事件循环。
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # ---- 步骤1：城市名 → 经纬度（地理编码） ----
                geo_resp = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={
                        "name": city, "count": 1,
                        "language": "zh", "format": "json",
                    },
                )
                geo_resp.raise_for_status()
                geo_data = geo_resp.json()

                if not geo_data.get("results"):
                    return f"未找到城市 '{city}'，请检查城市名称。"

                latitude = geo_data["results"][0]["latitude"]
                longitude = geo_data["results"][0]["longitude"]
                official_city_name = geo_data["results"][0]["name"]

                # ---- 步骤2：经纬度 → 实时天气 ----
                weather_resp = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": latitude,
                        "longitude": longitude,
                        "current_weather": True,
                        "timezone": "auto",
                        "temperature_unit": "celsius",
                    },
                )
                weather_resp.raise_for_status()
                weather_data = weather_resp.json()

                if "current_weather" in weather_data:
                    return self._format_weather(official_city_name, weather_data)
                return "获取天气数据失败，未返回有效信息。"

        except httpx.HTTPError as e:
            return f"网络请求错误：{str(e)}"
        except Exception as e:
            return f"处理天气数据时发生未知错误：{str(e)}"

    # ==================== 格式化输出（同步/异步共用） ====================
    @staticmethod
    def _format_weather(city_name: str, weather_data: dict) -> str:
        """将 Open-Meteo 的天气代码转为中文描述文本。
        天气代码映射：
            0-1:  晴朗    2-3:   多云
            45-48: 有雾    51-55: 毛毛雨
            61-82: 有雨    71-86: 有雪
        """
        current = weather_data["current_weather"]
        temperature = current["temperature"]
        wind_speed = current["windspeed"]
        weathercode = current["weathercode"]

        # 天气代码 → 中文描述映射表
        weather_desc_map = {
            **dict.fromkeys([0, 1], "晴朗"),
            **dict.fromkeys([2, 3], "多云"),
            **dict.fromkeys([45, 48], "有雾"),
            **dict.fromkeys([51, 53, 55], "毛毛雨"),
            **dict.fromkeys([61, 63, 65, 80, 81, 82], "有雨"),
            **dict.fromkeys([71, 73, 75, 77, 85, 86], "有雪"),
        }
        weather_desc = weather_desc_map.get(weathercode, "未知")

        return (
            f"{city_name} 当前天气：{weather_desc}，温度 {temperature}°C，"
            f"风速 {wind_speed} km/h"
        )
