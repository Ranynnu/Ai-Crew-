import asyncio
import yaml
from pathlib import Path
from crewai.flow.flow import Flow, listen, start, router, or_, and_
from crewai import Crew, Agent, Task, LLM
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from tools.weather_tool import WeatherTool

# ==================== 路径配置 ====================
CURRENT_DIR = Path(__file__).parent.resolve()          # src/ai
PROJECT_ROOT = CURRENT_DIR.parent.parent               # D:\crewai\ai
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
CONFIG_DIR = CURRENT_DIR / "config"

# ==================== YAML 配置加载 ====================
def _load_yaml(filename: str) -> dict:
    path = CONFIG_DIR / filename
    if not path.exists():
        print(f"WARNING: YAML config not found: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

AGENTS_CFG = _load_yaml("agents.yaml")
TASKS_CFG = _load_yaml("tasks.yaml")

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
    """
    混合架构：
    - 静态配置（role, goal, backstory, expected_output）从 agents.yaml / tasks.yaml 读取
    - 动态参数（destination, days, preferences, weather）由工厂方法注入
    """

    # Agent key 映射：工厂方法 → YAML 中的 agent名
    AGENT_MAP = {
        "weather":   "weather_expert",
        "attractions": "attractions_expert",
        "food":      "food_expert",
        "itinerary": "coordinator",
    }

    # Task key 映射：工厂方法 → YAML 中的 task名
    TASK_MAP = {
        "weather":   "gather_weather",
        "attractions": "plan_attractions",
        "food":      "plan_food",
        "itinerary": "create_itinerary",
    }

    def __init__(self, llm):
        self.llm = llm
        self.weather_tool = WeatherTool()
        self._knowledge_sources = None  # 延迟加载

    # ========== 从 YAML 取 Agent/Task 静态配置 ==========

    def _get_agent_config(self, key: str) -> dict:
        """读取 YAML 中的 Agent 静态配置"""
        name = self.AGENT_MAP.get(key, key)
        return AGENTS_CFG.get(name, {})

    def _get_task_config(self, key: str) -> dict:
        """读取 YAML 中的 Task 静态配置模板"""
        name = self.TASK_MAP.get(key, key)
        return TASKS_CFG.get(name, {})

    # ========== 知识库 ==========

    def _get_knowledge_sources(self):
        """加载 knowledge/ 目录下所有支持的文档。
        如果没有知识库文件，返回 None，Agent 依靠 LLM 自身知识生成。
        """
        if self._knowledge_sources is not None:
            return self._knowledge_sources  # 已缓存

        supported_exts = {'.docx', '.pdf', '.xlsx', '.pptx', '.html', '.md', '.asciidoc'}
        knowledge_files = []

        if KNOWLEDGE_DIR.exists():
            for f in KNOWLEDGE_DIR.iterdir():
                if f.is_file() and f.suffix.lower() in supported_exts:
                    knowledge_files.append(f)
                    print(f"[knowledge] found: {f.name}")

        if knowledge_files:
            try:
                self._knowledge_sources = [
                    CrewDoclingSource(file_paths=[str(p) for p in knowledge_files])
                ]
                print(f"[knowledge] loaded {len(knowledge_files)} file(s)")
            except Exception as e:
                print(f"[knowledge] load failed: {e}")
                self._knowledge_sources = None
        else:
            print("[knowledge] no documents found, agents will rely on LLM knowledge")
            self._knowledge_sources = None

        return self._knowledge_sources

    # ========== 创建 Agent（YAML 静态属性 + 代码注入 LLM/tools） ==========

    def _create_agent_from_yaml(self, key: str, tools=None):
        """从 YAML 加载 Agent 的 role/goal/backstory，代码注入 LLM 和工具"""
        cfg = self._get_agent_config(key)
        return Agent(
            role=cfg.get("role", ""),
            goal=cfg.get("goal", ""),
            backstory=cfg.get("backstory", ""),
            llm=self.llm,
            tools=tools or [],
            allow_delegation=cfg.get("allow_delegation", False),
            verbose=cfg.get("verbose", True),
            memory=False,
        )

    # ========== 填充 Task 描述中的占位符 ==========

    def _fill_task_description(self, key: str, **kwargs) -> str:
        """从 YAML 读取 Task 描述模板，用动态参数填充占位符"""
        cfg = self._get_task_config(key)
        template = cfg.get("description", "")
        # 支持 {destination}, {days}, {preferences}, {weather_info} 等插值
        try:
            return template.format(**kwargs)
        except KeyError as e:
            print(f"[warn] missing placeholder {e} in task '{key}'")
            return template

    def _get_task_expected_output(self, key: str) -> str:
        cfg = self._get_task_config(key)
        return cfg.get("expected_output", "")

    # ========== 各 Crew 工厂方法 ==========

    def create_weather_crew(self, destination: str):
        agent = self._create_agent_from_yaml("weather", tools=[self.weather_tool])
        description = self._fill_task_description("weather", destination=destination)
        task = Task(
            description=description,
            agent=agent,
            expected_output=self._get_task_expected_output("weather"),
        )
        return Crew(agents=[agent], tasks=[task], verbose=True)

    def create_attractions_crew(self, destination: str, days: int,
                                 preferences: str, weather_info: str):
        agent = self._create_agent_from_yaml("attractions")
        description = self._fill_task_description(
            "attractions",
            destination=destination,
            days=days,
            preferences=preferences,
        )
        # 把天气信息追加到描述末尾
        full_description = (
            f"{description}\n"
            f"[来自天气专家的信息]：{weather_info}\n"
            f"请结合天气状况调整推荐（如雨天优先室内景点）。"
        )
        task = Task(
            description=full_description,
            agent=agent,
            expected_output=self._get_task_expected_output("attractions"),
        )
        knowledge_sources = self._get_knowledge_sources()
        return Crew(
            agents=[agent],
            tasks=[task],
            knowledge_sources=knowledge_sources if knowledge_sources else None,
            embedder=EMBEDDER_CONFIG if knowledge_sources else None,
            verbose=True,
        )

    def create_food_crew(self, destination: str, preferences: str, weather_info: str):
        agent = self._create_agent_from_yaml("food")
        description = self._fill_task_description(
            "food",
            destination=destination,
            preferences=preferences,
        )
        full_description = (
            f"{description}\n"
            f"[来自天气专家的信息]：{weather_info}\n"
            f"请结合天气推荐合适的餐食（如天冷推荐热汤锅、天热推荐清爽食物）。"
        )
        task = Task(
            description=full_description,
            agent=agent,
            expected_output=self._get_task_expected_output("food"),
        )
        knowledge_sources = self._get_knowledge_sources()
        return Crew(
            agents=[agent],
            tasks=[task],
            knowledge_sources=knowledge_sources if knowledge_sources else None,
            embedder=EMBEDDER_CONFIG if knowledge_sources else None,
            verbose=True,
        )

    def create_itinerary_crew(self, destination: str, days: int, preferences: str,
                               weather_info: str, attractions_info: str, food_info: str):
        agent = self._create_agent_from_yaml("itinerary")
        description = self._fill_task_description(
            "itinerary",
            destination=destination,
            days=days,
            preferences=preferences,
        )
        # 整合时把上游结果直接拼进 description
        full_description = (
            f"{description}\n\n"
            f"===== 天气汇总 =====\n{weather_info}\n\n"
            f"===== 景点规划 =====\n{attractions_info}\n\n"
            f"===== 美食推荐 =====\n{food_info}\n\n"
            f"请以上述信息为基础，生成完整 {days} 天行程单（上午/下午/晚上）。"
        )
        task = Task(
            description=full_description,
            agent=agent,
            expected_output=self._get_task_expected_output("itinerary"),
        )
        return Crew(agents=[agent], tasks=[task], verbose=True)


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
        crew_weather = self.factory.create_weather_crew(self.destination)
        result = await crew_weather.kickoff_async()
        return str(result)

    # ---- 阶段 2：拿到天气后，并行收集景点和美食 ----
    @listen("collect_weather")
    async def collect_attractions(self):
        weather = self.state.get("collect_weather", "")
        crew_attractions = self.factory.create_attractions_crew(
            self.destination, self.days, self.preferences, weather
        )
        result = await crew_attractions.kickoff_async()
        return str(result)

    @listen("collect_weather")
    async def collect_food(self):
        weather = self.state.get("collect_weather", "")
        crew_food = self.factory.create_food_crew(
            self.destination, self.preferences, weather
        )
        result = await crew_food.kickoff_async()
        return str(result)

    # ---- 阶段 3：景点和美食都完成后，整合行程 ----
    @listen(and_("collect_attractions", "collect_food"))
    async def combine(self, combined):
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
    print("  Final Itinerary")
    print("=" * 50)
    print(result)
    return result


if __name__ == "__main__":
    run_travel_planning("三亚", days=3, preferences="喜欢潜水，怕晒")
