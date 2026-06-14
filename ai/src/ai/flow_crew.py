import asyncio
from pathlib import Path
from crewai.flow.flow import Flow, listen, start, router, or_, and_
from crewai import Crew, Agent, Task, LLM
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from tools.weather_tool import WeatherTool

# ==================== 路径配置 ====================
CURRENT_DIR = Path(__file__).parent.resolve()          # src/ai
PROJECT_ROOT = CURRENT_DIR.parent.parent               # D:\crewai\ai
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

# ==================== LLM 初始化 ====================
llm = LLM(
    model="ollama/qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0.7
)

# ==================== embedder 配置（共用） ====================
EMBEDDER_CONFIG = {
    "provider": "ollama",
    "config": {"model": "nomic-embed-text", "base_url": "http://localhost:11434"}
}


# ==================== 工厂类 ====================
class TravelCrewFactory:
    def __init__(self, llm):
        self.llm = llm
        self.weather_tool = WeatherTool()
        self._knowledge_sources = None  # 延迟加载

    # ---------- 知识库：加载 knowledge/ 下所有文档 ----------
    def _get_knowledge_sources(self):
        """加载 knowledge/ 目录下所有支持的文档（docx、pdf、xlsx、pptx、html、md）。
        过滤掉纯用户偏好文本文件（user_preference.txt）。
        如果没有知识库文件，返回 None，Agent 依靠 LLM 自身知识生成。
        """
        if self._knowledge_sources is not None:
            return self._knowledge_sources  # 已加载过，直接返回

        # 支持的文档后缀（Docling 支持的格式）
        supported_exts = {'.docx', '.pdf', '.xlsx', '.pptx', '.html', '.md', '.asciidoc'}
        knowledge_files = []

        if KNOWLEDGE_DIR.exists():
            for f in KNOWLEDGE_DIR.iterdir():
                if f.is_file() and f.suffix.lower() in supported_exts:
                    knowledge_files.append(f)
                    print(f"📄 发现知识库文件: {f.name}")

        if knowledge_files:
            try:
                self._knowledge_sources = [
                    CrewDoclingSource(file_paths=[str(p) for p in knowledge_files])
                ]
                print(f"✅ 成功加载 {len(knowledge_files)} 个知识库文件")
            except Exception as e:
                print(f"❌ 知识库加载失败: {e}")
                self._knowledge_sources = None
        else:
            print(f"⚠️ knowledge/ 目录下未找到知识库文档，Agent 将完全依靠自身知识")
            self._knowledge_sources = None

        return self._knowledge_sources

    # ---------- 创建 Agent ----------
    def _create_agent(self, role, goal, backstory, tools=None):
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=self.llm,
            tools=tools or [],
            allow_delegation=False,
            verbose=True,
            memory=False
        )

    # ---------- 天气 Crew ----------
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

    # ---------- 景点 Crew ----------
    def create_attractions_crew(self, destination: str, days: int,
                                 preferences: str, weather_info: str):
        agent = self._create_agent(
            role="景点策划专家",
            goal="推荐最适合用户的景点",
            backstory="你是本地通，能根据偏好推荐景点并安排合理路线。"
        )
        task = Task(
            description=(
                f"为{destination}规划{days}天的景点行程。\n"
                f"用户偏好：{preferences}\n"
                f"当地天气：{weather_info}\n"
                f"请结合天气状况，推荐适合的景点并输出分天景点列表（含推荐理由）。"
                f"如有雨天，优先推荐室内景点或博物馆。"
            ),
            agent=agent,
            expected_output="景点行程规划"
        )
        knowledge_sources = self._get_knowledge_sources()
        return Crew(
            agents=[agent],
            tasks=[task],
            knowledge_sources=knowledge_sources if knowledge_sources else None,
            embedder=EMBEDDER_CONFIG if knowledge_sources else None,
            verbose=True
        )

    # ---------- 美食 Crew ----------
    def create_food_crew(self, destination: str, preferences: str, weather_info: str):
        agent = self._create_agent(
            role="地道美食向导",
            goal="推荐符合口味的当地美食",
            backstory="你是美食家，推荐地道餐厅并说明理由。"
        )
        task = Task(
            description=(
                f"为{destination}推荐特色美食和餐厅。\n"
                f"用户偏好：{preferences}\n"
                f"当地天气：{weather_info}\n"
                f"请结合天气推荐合适的餐食（如天冷推荐热汤锅、天热推荐清爽食物）。"
            ),
            agent=agent,
            expected_output="美食推荐列表"
        )
        knowledge_sources = self._get_knowledge_sources()
        return Crew(
            agents=[agent],
            tasks=[task],
            knowledge_sources=knowledge_sources if knowledge_sources else None,
            embedder=EMBEDDER_CONFIG if knowledge_sources else None,
            verbose=True
        )

    # ---------- 整合 Crew ----------
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

            请整合以上信息，生成一份详细的{days}天行程单（上午、下午、晚上），
            注意避开用户不喜欢的内容（如怕晒则避免长时间户外）。
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

    # ---- 阶段 1：先获取天气 ----
    @start()
    async def collect_weather(self):
        """第一步：获取目的地实时天气"""
        crew_weather = self.factory.create_weather_crew(self.destination)
        result = await crew_weather.kickoff_async()
        return str(result)

    # ---- 阶段 2：拿到天气后，并行收集景点和美食 ----
    @listen("collect_weather")
    async def collect_attractions(self):
        """第二步（并行A）：根据天气推荐景点"""
        weather = self.state.get("collect_weather", "")
        crew_attractions = self.factory.create_attractions_crew(
            self.destination, self.days, self.preferences, weather
        )
        result = await crew_attractions.kickoff_async()
        return str(result)

    @listen("collect_weather")
    async def collect_food(self):
        """第二步（并行B）：根据天气推荐美食"""
        weather = self.state.get("collect_weather", "")
        crew_food = self.factory.create_food_crew(
            self.destination, self.preferences, weather
        )
        result = await crew_food.kickoff_async()
        return str(result)

    # ---- 阶段 3：景点和美食都完成后，整合行程 ----
    @listen(and_("collect_attractions", "collect_food"))
    async def combine(self, combined):
        """第三步：整合天气、景点、美食，生成最终行程"""
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
