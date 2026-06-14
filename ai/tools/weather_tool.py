import requests
from crewai.tools import BaseTool


# 确保已安装 requests 库: pip install requests

class WeatherTool(BaseTool):
    name: str = "Weather Tool"
    description: str = "查询指定城市的当前天气（使用真实API）"

    def _run(self, city: str) -> str:
        # 1. 通过城市名获取地理坐标 (经纬度)
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": city, "count": 1, "language": "zh", "format": "json"}

        try:
            geo_resp = requests.get(geo_url, params=geo_params, timeout=10)
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()

            if not geo_data.get("results"):
                return f"未找到城市 '{city}'，请检查城市名称。"

            # 提取坐标和标准城市名
            latitude = geo_data["results"][0]["latitude"]
            longitude = geo_data["results"][0]["longitude"]
            official_city_name = geo_data["results"][0]["name"]

            # 2. 用坐标获取实时天气
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": True,
                "timezone": "auto",
                "temperature_unit": "celsius"  # 使用摄氏度
            }

            weather_resp = requests.get(weather_url, params=weather_params, timeout=10)
            weather_resp.raise_for_status()
            weather_data = weather_resp.json()

            if "current_weather" in weather_data:
                current = weather_data["current_weather"]
                temperature = current["temperature"]
                # Open-Meteo的风速单位是km/h, 天气代码需要转换, 这里简化处理
                wind_speed = current["windspeed"]
                weathercode = current["weathercode"]

                # 简单地将天气代码转换为文字描述
                weather_desc = "未知"
                if weathercode in [0, 1]:
                    weather_desc = "晴朗"
                elif weathercode in [2, 3]:
                    weather_desc = "多云"
                elif weathercode in [45, 48]:
                    weather_desc = "有雾"
                elif weathercode in [51, 53, 55]:
                    weather_desc = "毛毛雨"
                elif weathercode in [61, 63, 65, 80, 81, 82]:
                    weather_desc = "有雨"
                elif weathercode in [71, 73, 75, 77, 85, 86]:
                    weather_desc = "有雪"

                return (f"{official_city_name} 当前天气：{weather_desc}，温度 {temperature}°C，"
                        f"风速 {wind_speed} km/h")
            else:
                return "获取天气数据失败，未返回有效信息。"

        except requests.exceptions.RequestException as e:
            return f"网络请求错误：{str(e)}"
        except Exception as e:
            return f"处理天气数据时发生未知错误：{str(e)}"