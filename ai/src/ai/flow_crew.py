#!/usr/bin/env python
"""
============================================================
 flow_crew.py — CrewAI 多智能体旅行规划系统
 架构：天气先行 → 并行(预算+景点+美食) → 耦合整合
============================================================
"""
# ==================== 一劳永逸：禁用 rich 边框 + 强制 UTF-8 ====================
# 必须在所有 import 之前设置
import os, io, sys
os.environ["CREWAI_DISABLE_RICH"] = "1"
# 强制 stdout/stderr 为 UTF-8（解决 Windows GBK 终端乱码）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ==================== 标准库导入 ====================
import asyncio          # 异步编程，支持 Flow 的 kickoff_async
import sys              # sys.path 管理，确保跨目录导入
import yaml             # YAML 配置文件解析
from pathlib import Path  # 跨平台路径处理

# ==================== CrewAI 框架导入 ====================
from crewai.flow.flow import Flow, listen, start, router, or_, and_
from crewai import Crew, Agent, Task, LLM
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource

# ==================== 自定义工具导入 ====================
from tools.weather_tool import WeatherTool  # 调用 Open-Meteo 免费天气 API

# ==================== 路径初始化：确保跨目录导入可用 ====================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # ai 项目根
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ==================== 核心路径常量 ====================
CURRENT_DIR   = Path(__file__).parent.resolve()       # src/ai
PROJECT_ROOT  = CURRENT_DIR.parent.parent              # ai/ (D:\crewai\ai)
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"             # 知识库文档目录
CONFIG_DIR    = CURRENT_DIR / "config"                 # YAML 配置目录

# ==================== YAML 配置加载 ====================
def _load_yaml(filename: str) -> dict:
    """从 config/ 目录加载 YAML 文件，返回字典（文件缺失时返回空字典）"""
    path = CONFIG_DIR / filename
    if not path.exists():
        print(f"WARNING: YAML config not found: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# 全局 YAML 配置（模块级别加载一次，所有工厂方法共享）
AGENTS_CFG = _load_yaml("agents.yaml")   # Agent role/goal/backstory
TASKS_CFG  = _load_yaml("tasks.yaml")    # Task description 模板

# ==================== LLM 大模型初始化 ====================
llm = LLM(
    model="ollama/qwen2.5:7b",         # 本地 Ollama 模型（中文能力强）
    base_url="http://localhost:11434",  # Ollama 本地服务地址
    temperature=0.7,                    # 0.7 平衡创意和一致性
)

# ==================== Embedder 嵌入模型配置（知识库 RAG 用） ====================
# 注意：CrewAI 的 OllamaProvider 使用 model_name 字段（不是 model）
# 且通过 validation_alias 支持 EMBEDDINGS_OLLAMA_MODEL_NAME 等别名
EMBEDDER_CONFIG = {
    "provider": "ollama",
    "config": {
        "model_name": "nomic-embed-text",           # OllamaProvider 要求的字段名
        "url": "http://localhost:11434/api/embeddings",
    },
}


# ===================================================================
#  TravelCrewFactory — 混合架构工厂类
#  YAML 管理静态属性（role/goal/backstory/任务模板）
#  工厂方法注入动态参数（destination, days, budget, weather 等）
# ===================================================================
class TravelCrewFactory:
    """旅行规划工厂：从 YAML 加载静态配置，工厂方法注入动态参数"""

    # ---------- Agent key → YAML 映射 ----------
    AGENT_MAP = {
        "weather":     "weather_expert",     # 天气专家
        "attractions": "attractions_expert", # 景点策划
        "food":        "food_expert",        # 美食向导
        "budget":      "budget_expert",      # 预算规划师（新增）
        "itinerary":   "coordinator",        # 行程总指挥
    }

    # ---------- Task key → YAML 映射 ----------
    TASK_MAP = {
        "weather":     "gather_weather",    # 查询天气
        "attractions": "plan_attractions",  # 规划景点
        "food":        "plan_food",         # 食物菜单
        "budget":      "estimate_budget",   # 估算预算
        "itinerary":   "create_itinerary",  # 耦合整合
    }

    def __init__(self, llm):
        """初始化工厂：注入 LLM 实例和工具"""
        self.llm = llm
        self.weather_tool = WeatherTool()       # 天气查询工具（同步+异步）
        self._knowledge_sources = None           # 延迟加载，首次使用时初始化

    # ==================== YAML 辅助方法 ====================

    def _get_agent_config(self, key: str) -> dict:
        """根据工厂 key 获取 YAML 中对应的 Agent 配置"""
        name = self.AGENT_MAP.get(key, key)
        return AGENTS_CFG.get(name, {})

    def _get_task_config(self, key: str) -> dict:
        """根据工厂 key 获取 YAML 中对应的 Task 配置"""
        name = self.TASK_MAP.get(key, key)
        return TASKS_CFG.get(name, {})

    # ==================== Agent 创建（YAML 静态属性 + 代码注入 LLM/tools） ====================

    def _create_agent(self, key: str, tools=None):
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
            memory=False,  # 关闭记忆，避免依赖 OpenAI
        )

    # ==================== Task 描述插值 ====================

    def _fill_task_desc(self, key: str, **kwargs) -> str:
        """从 YAML 读取 Task 描述模板，用 Python format() 填充动态占位符
        支持的占位符：{destination}, {days}, {preferences}, {weather_info},
        {budget}, {budget_info}, {attractions_info}, {food_info}
        """
        cfg = self._get_task_config(key)
        template = cfg.get("description", "")
        try:
            return template.format(**kwargs)
        except KeyError as e:
            print(f"[warn] missing placeholder {e} in task '{key}'")
            return template

    def _get_expected_output(self, key: str) -> str:
        """获取 YAML 中 Task 的 expected_output 字段"""
        return self._get_task_config(key).get("expected_output", "")

    # ==================== 知识库：扫描 knowledge/ 目录下所有支持的文档 ====================

    def _get_knowledge_sources(self):
        """延迟加载 knowledge/ 目录下所有文档（docx/pdf/xlsx/pptx/html/md）。
        返回 CrewDoclingSource 列表用于向量搜索增强；无文件时返回 None，
        Agent 将完全依靠 LLM 自身知识生成推荐。结果会缓存到 self._knowledge_sources。
        """
        if self._knowledge_sources is not None:
            return self._knowledge_sources  # 已缓存，直接返回

        # Docling 支持的文档格式
        supported_exts = {'.docx', '.pdf', '.xlsx', '.pptx', '.html', '.md', '.asciidoc'}
        knowledge_files = []

        if KNOWLEDGE_DIR.exists():
            for f in KNOWLEDGE_DIR.iterdir():
                if f.is_file() and f.suffix.lower() in supported_exts:
                    knowledge_files.append(f)
                    print(f"[knowledge] found: {f.name}")

        if knowledge_files:
            try:
                # CrewDoclingSource 的 validate_content 会把 str 路径拼接到
                # KNOWLEDGE_DIRECTORY("knowledge") 后面，所以只传文件名即可
                self._knowledge_sources = [
                    CrewDoclingSource(file_paths=[p.name for p in knowledge_files])
                ]
                print(f"[knowledge] loaded {len(knowledge_files)} file(s)")
            except Exception as e:
                print(f"[knowledge] load failed: {e}")
                self._knowledge_sources = None
        else:
            print("[knowledge] no docs — agents rely on LLM knowledge")
            self._knowledge_sources = None

        return self._knowledge_sources

    # ==================== 各 Crew 工厂方法 ====================

    # ---------- 天气 Crew：查询目的地实时天气 ----------
    def create_weather_crew(self, destination: str):
        """创建天气查询 Crew（1 Agent + WeatherTool）"""
        agent = self._create_agent("weather", tools=[self.weather_tool])
        desc = self._fill_task_desc("weather", destination=destination)
        task = Task(
            description=desc,
            agent=agent,
            expected_output=self._get_expected_output("weather"),
        )
        return Crew(agents=[agent], tasks=[task], verbose=True)

    # ---------- 预算 Crew：解析预算描述，输出每日分项预算分配 ----------
    def create_budget_crew(self, destination: str, days: int, budget: str):
        """创建预算规划 Crew：分析用户 budget 字符串（如'经济型3000元'），
        输出结构化的每日分项预算（住宿/门票/餐饮/交通/购物）"""
        agent = self._create_agent("budget")
        desc = self._fill_task_desc(
            "budget", destination=destination, days=days, budget=budget,
        )
        task = Task(
            description=desc,
            agent=agent,
            expected_output=self._get_expected_output("budget"),
        )
        return Crew(agents=[agent], tasks=[task], verbose=True)

    # ---------- 景点 Crew：推荐景点 + 预算约束 ----------
    def create_attractions_crew(self, destination: str, days: int,
                                 preferences: str, weather_info: str,
                                 budget_info: str):
        """创建景点策划 Crew：从 YAML 加载任务模板，注入天气+预算约束。
        要求 Agent 为每个景点标注区域/门票/活动强度。"""
        agent = self._create_agent("attractions")
        base = self._fill_task_desc(
            "attractions",
            destination=destination,
            days=days,
            preferences=preferences,
            budget_info=budget_info,
        )
        # 追加天气和结构要求
        full = (
            f"{base}\n"
            f"[天气信息]：{weather_info}\n"
            f"[预算约束]：{budget_info}\n"
            f"要求：每个景点标注区域位置+门票价格+活动强度；门票总和不超过预算。"
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

    # ---------- 美食 Crew：输出结构化食物菜单 ----------
    def create_food_crew(self, destination: str, preferences: str,
                         weather_info: str, budget_info: str):
        """创建美食向导 Crew：输出结构化食物菜单，每家餐厅标注
        区域/场景标签/适合餐段/人均/室内户外。
        注意：这里产出的是'菜单'而非最终推荐——最终选择在整合阶段按耦合规则完成。"""
        agent = self._create_agent("food")
        base = self._fill_task_desc(
            "food",
            destination=destination,
            preferences=preferences,
            budget_info=budget_info,
        )
        full = (
            f"{base}\n"
            f"[天气信息]：{weather_info}\n"
            f"[预算约束]：{budget_info}\n"
            f"输出要求：每家餐厅标注区域、场景标签、适合餐段、人均、室内/户外。"
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

    # ---------- 整合 Crew：耦合规则在 YAML 模板中 ----------
    def create_itinerary_crew(self, destination: str, days: int, preferences: str,
                               weather_info: str, attractions_info: str,
                               food_info: str, budget_info: str):
        """创建行程整合 Crew：将天气+景点+食物菜单+预算按耦合规则整合。
        耦合规则（在 tasks.yaml 的 create_itinerary 描述中）：
        - 早餐：靠近住宿区域，不与景点耦合
        - 午餐：与上午景点地理耦合（<1km）+ 场景匹配 + 天气响应
        - 下午茶：按上午体力强度条件触发
        - 晚餐：与下午景点耦合 + 回住宿路线逻辑
        - 宵夜：仅当天有夜市/夜游时触发
        - 每日预算上限约束
        """
        agent = self._create_agent("itinerary")
        # 耦合指令已在 YAML 模板中——直接插值填充所有数据
        base = self._fill_task_desc(
            "itinerary",
            destination=destination,
            days=days,
            preferences=preferences,
            budget_info=budget_info,
            weather_info=weather_info,
            attractions_info=attractions_info,
            food_info=food_info,
        )
        task = Task(description=base, agent=agent,
                    expected_output=self._get_expected_output("itinerary"))
        return Crew(agents=[agent], tasks=[task], verbose=True)


# ===================================================================
#  TravelFlow — 多阶段异步流水线
#  阶段1：天气先行（唯一 @start）
#  阶段2：拿到天气后，并行 3 路（预算 + 景点 + 美食）
#  阶段3：3 路都完成后，耦合整合生成最终行程
# ===================================================================
class TravelFlow(Flow):
    """旅行规划 Flow：天气先行 → 并行收集(预算/景点/美食) → 耦合整合"""

    def __init__(self, destination: str, days: int, preferences: str,
                 budget: str = "中等"):
        super().__init__()
        self.destination  = destination
        self.days         = days
        self.preferences  = preferences
        self.budget       = budget           # 用户预算描述（如'经济型3000元'）
        self.factory      = TravelCrewFactory(llm)

    # ---- 阶段 1：天气先行（唯一的 @start 方法） ----
    @start()
    async def collect_weather(self):
        """第一步：获取目的地实时天气（调用 WeatherTool → Open-Meteo API）"""
        result = await self.factory.create_weather_crew(
            self.destination
        ).kickoff_async()
        return str(result)

    # ---- 阶段 2：预算估算（与景点/美食并行） ----
    @listen("collect_weather")
    async def collect_budget(self):
        """第二步（并行A）：解析用户预算描述，输出每日分项预算分配表"""
        result = await self.factory.create_budget_crew(
            self.destination, self.days, self.budget,
        ).kickoff_async()
        return str(result)

    # ---- 阶段 2：景点规划（与预算/美食并行） ----
    @listen("collect_weather")
    async def collect_attractions(self):
        """第二步（并行B）：根据天气+预算+偏好，推荐景点（含区域/门票/强度）"""
        weather = self.state.get("collect_weather", "")
        # 预算可能尚未完成（并行），先用用户原始预算描述
        budget_info = self.budget
        result = await self.factory.create_attractions_crew(
            self.destination, self.days, self.preferences, weather, budget_info,
        ).kickoff_async()
        return str(result)

    # ---- 阶段 2：美食菜单（与预算/景点并行） ----
    @listen("collect_weather")
    async def collect_food(self):
        """第二步（并行C）：根据天气+预算+偏好，输出结构化食物菜单"""
        weather = self.state.get("collect_weather", "")
        budget_info = self.budget
        result = await self.factory.create_food_crew(
            self.destination, self.preferences, weather, budget_info,
        ).kickoff_async()
        return str(result)

    # ---- 阶段 3：耦合整合（三路都完成后触发） ----
    @listen(and_("collect_budget", "collect_attractions", "collect_food"))
    async def combine(self, combined):
        """第三步：整合天气+景点+食物菜单+预算，按耦合规则生成完整行程。
        耦合规则包括：
        - 地理耦合（午餐近上午景点、晚餐顺路回住宿）
        - 场景匹配（爬山后轻食、博物馆后茶馆）
        - 天气响应（雨天室内、晴天露台）
        - 餐段拆分（早餐近酒店、下午茶体力触发、宵夜条件触发）
        - 预算约束（每人每天不超上限）
        """
        weather      = self.state.get("collect_weather", "")
        budget       = self.state.get("collect_budget", "")
        attractions  = self.state.get("collect_attractions", "")
        food         = self.state.get("collect_food", "")

        # 如果 budget Agent 未完成，用用户原始 budget 兜底
        budget_info = budget if budget else self.budget

        result = await self.factory.create_itinerary_crew(
            self.destination, self.days, self.preferences,
            weather, attractions, food, budget_info,
        ).kickoff_async()
        return result


# ===================================================================
#  运行入口
# ===================================================================
def run_travel_planning(destination: str, days: int = 3,
                        preferences: str = "", budget: str = "中等"):
    """主入口：接收用户输入，启动 Flow 并打印最终行程。
    参数：
        destination - 旅行目的地城市名
        days        - 计划天数
        preferences - 用户偏好描述（可选）
        budget      - 预算描述（如'经济型3000元'、'中等'、'豪华'）
    """
    flow = TravelFlow(destination, days, preferences, budget)
    result = asyncio.run(flow.kickoff_async())
    print("\n" + "=" * 50)
    print("  最终行程单")
    print("=" * 50)
    print(str(result))
    return result


# 直接运行此文件时的默认测试参数
if __name__ == "__main__":
    run_travel_planning("三亚", days=3, preferences="喜欢潜水，怕晒", budget="中等")
