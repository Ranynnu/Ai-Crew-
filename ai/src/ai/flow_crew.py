import asyncio
from pathlib import Path
from crewai.flow.flow import Flow, listen, start, router, or_, and_
from crewai import Crew, Agent, Task, LLM
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from tools.weather_tool import WeatherTool

# ==================== 路径配置 ====================
from pathlib import Path

# 获取当前文件所在目录的绝对路径
CURRENT_DIR = Path(__file__).parent.resolve()          # src/ai
PROJECT_ROOT = CURRENT_DIR.parent.parent               # D:\crewai\ai
KNOWLEDGE_FILE = PROJECT_ROOT / "knowledge" / "三亚旅游指南.docx"

print(f"DEBUG: 知识库路径 = {KNOWLEDGE_FILE}")
print(f"文件是否存在: {KNOWLEDGE_FILE.exists()}")

knowledge_source = None
if KNOWLEDGE_FILE.exists():
    try:
        from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
        knowledge_source = CrewDoclingSource(file_paths=[str(KNOWLEDGE_FILE)])
        print(f"✅ 成功加载知识库: {KNOWLEDGE_FILE}")
    except Exception as e:
        print(f"❌ 知识库加载失败: {e}")
else:
    print(f"⚠️ 未找到知识库文件: {KNOWLEDGE_FILE}，请检查路径和文件名")

# ==================== LLM 初始化 ====================
llm = LLM(
    model="ollama/qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0.7
)




# ==================== 工厂类 ====================
class TravelCrewFactory:
    def __init__(self, llm):
        self.llm = llm
        self.weather_tool = WeatherTool()

    def _create_agent(self, role, goal, backstory, tools=None):
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=self.llm,
            tools=tools or [],
            allow_delegation=False,  # 关闭委派，避免复杂工具调用
            verbose=True,
            memory=False  # 关闭内存，避免依赖 OpenAI
        )

    def create_weather_crew(self, destination: str):
        """仅负责天气查询的 Crew"""
        agent = self._create_agent(
            role="旅行天气专家",
            goal="提供目的地的实时天气信息",
            backstory="你擅长使用天气工具查询准确数据。",
            tools=[self.weather_tool]
        )
        task = Task(
            description=f"查询{destination}的当前天气、温度、风速，并给出穿衣建议。",
            agent=agent,
            expected_output="简明天气报告"
        )
        return Crew(
            agents=[agent],
            tasks=[task],
            verbose=True
        )

    def create_attractions_crew(self, destination: str, days: int, preferences: str):
        agent = self._create_agent(
            role="景点策划专家",
            goal="推荐最适合用户的景点",
            backstory="你是本地通，能根据偏好推荐景点并安排合理路线。"
        )
        task = Task(
            description=f"为{destination}规划{days}天的景点行程。用户偏好：{preferences}。请输出分天景点列表，含推荐理由。",
            agent=agent,
            expected_output="景点行程规划"
        )
        return Crew(
            agents=[agent],
            tasks=[task],
            knowledge_sources=[knowledge_source] if knowledge_source else None,
            embedder={
                "provider": "ollama",
                "config": {"model": "nomic-embed-text", "base_url": "http://localhost:11434"}
            } if knowledge_source else None,
            verbose=True
        )

    def create_food_crew(self, destination: str, preferences: str):
        agent = self._create_agent(
            role="地道美食向导",
            goal="推荐符合口味的当地美食",
            backstory="你是美食家，推荐地道餐厅并说明理由。"
        )
        task = Task(
            description=f"为{destination}推荐特色美食和餐厅。用户偏好：{preferences}。",
            agent=agent,
            expected_output="美食推荐列表"
        )
        return Crew(
            agents=[agent],
            tasks=[task],
            knowledge_sources=[knowledge_source] if knowledge_source else None,
            embedder={
                "provider": "ollama",
                "config": {"model": "nomic-embed-text", "base_url": "http://localhost:11434"}
            } if knowledge_source else None,
            verbose=True
        )

    def create_itinerary_crew(self, destination: str, days: int, preferences: str,
                              weather_info: str, attractions_info: str, food_info: str):
        """整合所有信息生成最终行程"""
        agent = self._create_agent(
            role="资深旅行规划总指挥",
            goal="整合专家成果，输出完美行程",
            backstory="你是总规划师，善于将天气、景点、美食整合成每日计划。"
        )
        task = Task(
            description=f"""
            目的地：{destination}
            天数：{days}
            用户偏好：{preferences}

            天气信息：{weather_info}
            景点规划：{attractions_info}
            美食推荐：{food_info}

            请整合以上信息，生成一份详细的{days}天行程单（上午、下午、晚上），注意避开用户不喜欢的内容（如怕晒则避免长时间户外）。
            """,
            agent=agent,
            expected_output="完整的行程单"
        )
        return Crew(
            agents=[agent],
            tasks=[task],
            verbose=True
        )


# ==================== Flow 定义 ====================
class TravelFlow(Flow):
    def __init__(self, destination: str, days: int, preferences: str):
        super().__init__()
        self.destination = destination
        self.days = days
        self.preferences = preferences
        self.factory = TravelCrewFactory(llm)

    @start()
    async def collect_weather(self):
        """并行收集天气、景点、美食信息"""
        crew_weather = self.factory.create_weather_crew(self.destination)
        result = await crew_weather.kickoff_async()
        return str(result)

    @start()
    async def collect_attractions(self):
        crew_attractions = self.factory.create_attractions_crew(
            self.destination, self.days, self.preferences
        )
        result = await crew_attractions.kickoff_async()
        return str(result)

    @start()
    async def collect_food(self):
        crew_food = self.factory.create_food_crew(self.destination, self.preferences)
        result = await crew_food.kickoff_async()
        return str(result)

    @listen(and_("collect_weather", "collect_attractions", "collect_food"))
    async def combine(self, combined):  # 注意：参数 combined 可以保留，但实际不使用；也可以去掉参数
        weather = self.state.get("collect_weather", "")
        attractions = self.state.get("collect_attractions", "")
        food = self.state.get("collect_food", "")

        crew_itinerary = self.factory.create_itinerary_crew(
            self.destination, self.days, self.preferences,
            weather, attractions, food
        )
        final_result = await crew_itinerary.kickoff_async()
        return final_result


# ==================== 运行入口 ====================
def run_travel_planning(destination: str, days: int = 3, preferences: str = ""):
    flow = TravelFlow(destination, days, preferences)
    result = asyncio.run(flow.kickoff_async())
    print("\n" + "=" * 50)
    print("📋 最终行程计划")
    print("=" * 50)
    print(result)
    return result


if __name__ == "__main__":
    run_travel_planning("三亚", days=3, preferences="喜欢潜水，怕晒")