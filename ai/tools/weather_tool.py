import httpx
from crewai.tools import BaseTool


class WeatherTool(BaseTool):
    name: str = "Weather Tool"
    description: str = "查询指定城市的当前天气（使用真实API）"

    # ---------- 同步入口（兜底） ----------
    def _run(self, city: str) -> str:
        """同步查询天气（基于 httpx 同步客户端，避免 requests 依赖）"""
        try:
            with httpx.Client(timeout=10) as client:
                # 1. 地理编码
                geo_resp = client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": city, "count": 1, "language": "zh", "format": "json"},
                )
                geo_resp.raise_for_status()
                geo_data = geo_resp.json()

                if not geo_data.get("results"):
                    return f"未找到城市 '{city}'，请检查城市名称。"

                latitude = geo_data["results"][0]["latitude"]
                longitude = geo_data["results"][0]["longitude"]
                official_city_name = geo_data["results"][0]["name"]

                # 2. 实时天气
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

    # ---------- 异步入口（Flow kickoff_async 调用此方法） ----------
    async def _arun(self, city: str) -> str:
        """异步查询天气，避免阻塞事件循环"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # 1. 地理编码
                geo_resp = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": city, "count": 1, "language": "zh", "format": "json"},
                )
                geo_resp.raise_for_status()
                geo_data = geo_resp.json()

                if not geo_data.get("results"):
                    return f"未找到城市 '{city}'，请检查城市名称。"

                latitude = geo_data["results"][0]["latitude"]
                longitude = geo_data["results"][0]["longitude"]
                official_city_name = geo_data["results"][0]["name"]

                # 2. 实时天气
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

    # ---------- 格式化输出（同步/异步共用） ----------
    @staticmethod
    def _format_weather(city_name: str, weather_data: dict) -> str:
        current = weather_data["current_weather"]
        temperature = current["temperature"]
        wind_speed = current["windspeed"]
        weathercode = current["weathercode"]

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
