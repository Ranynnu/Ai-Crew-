import asyncio
import yaml
from pathlib import Path
from crewai.flow.flow import Flow, listen, start, router, or_, and_
from crewai import Crew, Agent, Task, LLM
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from tools.weather_tool import WeatherTool

# ———— 将项目根目录加入 sys.path ————
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ==================== 路径 / 常量 ====================
CURRENT_DIR  = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
CONFIG_DIR    = CURRENT_DIR / "config"

# ==================== YAML 加载 ====================
def _load_yaml(filename: str) -> dict:
    path = CONFIG_DIR / filename
    if not path.exists():
        print(f"WARNING: YAML config not found: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

AGENTS_CFG = _load_yaml("agents.yaml")
TASKS_CFG  = _load_yaml("tasks.yaml")

# ==================== LLM / Embedder ====================
llm = LLM(
    model="ollama/qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0.7,
)

EMBEDDER_CONFIG = {
    "provider": "ollama",
    "config": {"model": "nomic-embed-text", "base_url": "http://localhost:11434"},
}


# ==================== 工厂类 ====================
class TravelCrewFactory:
    """混合架构：YAML 管静态配置（role/goal/backstory/模板），工厂方法注入动态参数。"""

    AGENT_MAP = {
        "weather":     "weather_expert",
        "attractions": "attractions_expert",
        "food":        "food_expert",
        "budget":      "budget_expert",
        "itinerary":   "coordinator",
    }
    TASK_MAP = {
        "weather":     "gather_weather",
        "attractions": "plan_attractions",
        "food":        "plan_food",
        "budget":      "estimate_budget",
        "itinerary":   "create_itinerary",
    }

    def __init__(self, llm):
        self.llm = llm
        self.weather_tool = WeatherTool()
        self._knowledge_sources = None

    # ---------- helpers ----------
    def _get_agent_config(self, key: str) -> dict:
        name = self.AGENT_MAP.get(key, key)
        return AGENTS_CFG.get(name, {})

    def _get_task_config(self, key: str) -> dict:
        name = self.TASK_MAP.get(key, key)
        return TASKS_CFG.get(name, {})

    def _create_agent(self, key: str, tools=None):
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

    def _fill_task_desc(self, key: str, **kwargs) -> str:
        cfg = self._get_task_config(key)
        template = cfg.get("description", "")
        try:
            return template.format(**kwargs)
        except KeyError as e:
            print(f"[warn] missing placeholder {e} in task '{key}'")
            return template

    def _get_expected_output(self, key: str) -> str:
        return self._get_task_config(key).get("expected_output", "")

    # ---------- 知识库 ----------
    def _get_knowledge_sources(self):
        if self._knowledge_sources is not None:
            return self._knowledge_sources
        exts = {".docx", ".pdf", ".xlsx", ".pptx", ".html", ".md", ".asciidoc"}
        files = []
        if KNOWLEDGE_DIR.exists():
            for f in KNOWLEDGE_DIR.iterdir():
                if f.is_file() and f.suffix.lower() in exts:
                    files.append(f)
                    print(f"[knowledge] found: {f.name}")
        if files:
            try:
                self._knowledge_sources = [CrewDoclingSource(file_paths=[str(p) for p in files])]
                print(f"[knowledge] loaded {len(files)} file(s)")
            except Exception as e:
                print(f"[knowledge] load failed: {e}")
                self._knowledge_sources = None
        else:
            print("[knowledge] no documents — agents rely on LLM knowledge")
            self._knowledge_sources = None
        return self._knowledge_sources

    # ---------- 天气 ----------
    def create_weather_crew(self, destination: str):
        agent = self._create_agent("weather", tools=[self.weather_tool])
        desc = self._fill_task_desc("weather", destination=destination)
        task = Task(description=desc, agent=agent,
                    expected_output=self._get_expected_output("weather"))
        return Crew(agents=[agent], tasks=[task], verbose=True)

    # ---------- 预算 ----------
    def create_budget_crew(self, destination: str, days: int, budget: str):
        agent = self._create_agent("budget")
        desc = self._fill_task_desc("budget", destination=destination, days=days, budget=budget)
        task = Task(description=desc, agent=agent,
                    expected_output=self._get_expected_output("budget"))
        return Crew(agents=[agent], tasks=[task], verbose=True)

    # ---------- 景点 ----------
    def create_attractions_crew(self, destination: str, days: int,
                                preferences: str, weather_info: str, budget_info: str):
        agent = self._create_agent("attractions")
        base = self._fill_task_desc(
            "attractions", destination=destination, days=days,
            preferences=preferences, budget_info=budget_info,
        )
        full = (
            f"{base}\n"
            f"[天气]：{weather_info}\n"
            f"[预算]：{budget_info}\n"
            f"要求：每个景点标注所在区域+门票价格+活动强度；门票总和不超过预算。"
        )
        task = Task(description=full, agent=agent,
                    expected_output=self._get_expected_output("attractions"))
        ks = self._get_knowledge_sources()
        return Crew(
            agents=[agent], tasks=[task],
            knowledge_sources=ks if ks else None,
            embedder=EMBEDDER_CONFIG if ks else None,
            verbose=True,
        )

    # ---------- 美食（结构化菜单） ----------
    def create_food_crew(self, destination: str, preferences: str,
                         weather_info: str, budget_info: str):
        agent = self._create_agent("food")
        base = self._fill_task_desc(
            "food", destination=destination, preferences=preferences,
            budget_info=budget_info,
        )
        full = (
            f"{base}\n"
            f"[天气]：{weather_info}\n"
            f"[预算]：{budget_info}\n"
            f"输出要求：为每家餐厅标注区域、场景标签、适合餐段、人均、室内/户外。"
        )
        task = Task(description=full, agent=agent,
                    expected_output=self._get_expected_output("food"))
        ks = self._get_knowledge_sources()
        return Crew(
            agents=[agent], tasks=[task],
            knowledge_sources=ks if ks else None,
            embedder=EMBEDDER_CONFIG if ks else None,
            verbose=True,
        )

    # ---------- 整合（完整耦合指令已在 YAML 中） ----------
    def create_itinerary_crew(self, destination: str, days: int, preferences: str,
                               weather_info: str, attractions_info: str,
                               food_info: str, budget_info: str):
        agent = self._create_agent("itinerary")
        base = self._fill_task_desc(
            "itinerary", destination=destination, days=days,
            preferences=preferences, budget_info=budget_info,
            weather_info=weather_info, attractions_info=attractions_info,
            food_info=food_info,
        )
        task = Task(description=base, agent=agent,
                    expected_output=self._get_expected_output("itinerary"))
        return Crew(agents=[agent], tasks=[task], verbose=True)


# ==================== Flow ====================
class TravelFlow(Flow):
    def __init__(self, destination: str, days: int, preferences: str, budget: str = "中等"):
        super().__init__()
        self.destination  = destination
        self.days         = days
        self.preferences  = preferences
        self.budget       = budget          # 用户预算描述
        self.factory      = TravelCrewFactory(llm)

    # --- 阶段 1：天气先行 ---
    @start()
    async def collect_weather(self):
        result = await self.factory.create_weather_crew(self.destination).kickoff_async()
        return str(result)

    # --- 阶段 2：拿到天气后，4 路并行（预算 / 景点 / 美食） ---
    @listen("collect_weather")
    async def collect_budget(self):
        result = await self.factory.create_budget_crew(
            self.destination, self.days, self.budget
        ).kickoff_async()
        return str(result)

    @listen("collect_weather")
    async def collect_attractions(self):
        weather = self.state.get("collect_weather", "")
        # 预算此时可能还没出（并行），先用用户原始预算描述
        budget_info = self.budget
        result = await self.factory.create_attractions_crew(
            self.destination, self.days, self.preferences, weather, budget_info
        ).kickoff_async()
        return str(result)

    @listen("collect_weather")
    async def collect_food(self):
        weather = self.state.get("collect_weather", "")
        budget_info = self.budget
        result = await self.factory.create_food_crew(
            self.destination, self.preferences, weather, budget_info
        ).kickoff_async()
        return str(result)

    # --- 阶段 3：4 路都完成后，耦合整合 ---
    @listen(and_("collect_budget", "collect_attractions", "collect_food"))
    async def combine(self, combined):
        weather      = self.state.get("collect_weather", "")
        budget       = self.state.get("collect_budget", "")
        attractions  = self.state.get("collect_attractions", "")
        food         = self.state.get("collect_food", "")

        # 如果 budget Agent 未完成，用原始 budget 兜底
        budget_info = budget if budget else self.budget

        result = await self.factory.create_itinerary_crew(
            self.destination, self.days, self.preferences,
            weather, attractions, food, budget_info,
        ).kickoff_async()
        return result


# ==================== 运行入口 ====================
def run_travel_planning(destination: str, days: int = 3,
                        preferences: str = "", budget: str = "中等"):
    flow = TravelFlow(destination, days, preferences, budget)
    result = asyncio.run(flow.kickoff_async())
    print("\n" + "=" * 50)
    print("  Final Itinerary")
    print("=" * 50)
    print(result)
    return result


if __name__ == "__main__":
    run_travel_planning("三亚", days=3, preferences="喜欢潜水，怕晒", budget="中等")
