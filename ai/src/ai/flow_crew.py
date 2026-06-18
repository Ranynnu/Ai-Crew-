#!/usr/bin/env python
"""
============================================================
 flow_crew.py — CrewAI 多智能体旅行规划系统（自定义知识库工具版）
 架构：天气先行 → 并行(预算+景点+美食) → 耦合整合
 知识库：本地字符串搜索，无 ChromaDB / 嵌入依赖
============================================================
"""
import os, io, sys
os.environ["CREWAI_DISABLE_RICH"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import asyncio
import yaml
import shutil
from pathlib import Path
from typing import Type, Optional

from crewai.flow.flow import Flow, listen, start, and_
from crewai import Crew, Agent, Task, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tools.weather_tool import WeatherTool

# ==================== 路径初始化 ====================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # ai/
CURRENT_DIR   = Path(__file__).parent.resolve()                 # src/ai/
SRC_DIR       = CURRENT_DIR.parent                               # src/
KNOWLEDGE_DIR = SRC_DIR / "knowledge"                           # src/knowledge/
CONFIG_DIR    = CURRENT_DIR / "config"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ==================== 自定义知识库搜索工具 ====================
class KnowledgeSearchInput(BaseModel):
    query: str = Field(..., description="要搜索的关键词或问题")

class KnowledgeSearchTool(BaseTool):
    name: str = "Search Local Knowledge Base"
    description: str = (
        "搜索本地旅行知识库，获取关于目的地景点、餐厅、交通等实用信息。"
        "输入一个或多个关键词，返回相关段落。"
    )
    args_schema: Type[BaseModel] = KnowledgeSearchInput

    # 存储知识库全文
    knowledge_text: str = ""

    def __init__(self, knowledge_dir: Path, **kwargs):
        super().__init__(**kwargs)
        # 读取并合并知识库文件
        contents = []
        if knowledge_dir.exists():
            for f in knowledge_dir.iterdir():
                if f.is_file() and f.suffix.lower() in {'.txt', '.md'}:
                    try:
                        text = f.read_text(encoding='utf-8')
                        contents.append(text)
                        print(f"[knowledge] 已加载: {f.name}")
                    except Exception as e:
                        print(f"[knowledge] 读取失败 {f.name}: {e}")
        self.knowledge_text = "\n\n".join(contents) if contents else ""
        if not self.knowledge_text:
            print("[knowledge] 警告：知识库为空！")

    def _run(self, query: str) -> str:
        """执行关键词搜索，返回包含所有关键词的句子及其上下文"""
        if not self.knowledge_text:
            return "知识库为空，无法搜索。"
        # 将查询拆分为关键词（空格或中文词语）
        keywords = query.replace("，", " ").replace(",", " ").split()
        if not keywords:
            return f"未提供有效的搜索关键词。"
        # 分割为段落（空行分隔）
        paragraphs = self.knowledge_text.split("\n\n")
        results = []
        for p in paragraphs:
            # 段落中必须包含所有关键词（不区分大小写）
            p_lower = p.lower()
            if all(k.lower() in p_lower for k in keywords):
                results.append(p.strip())
        if not results:
            return f"未找到与 '{query}' 相关的信息。"
        # 返回前 3 个最匹配的段落，每个截断 500 字
        output = []
        for i, para in enumerate(results[:3], 1):
            if len(para) > 500:
                para = para[:500] + "..."
            output.append(f"[结果 {i}]\n{para}")
        return "\n\n".join(output)

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

# ==================== LLM ====================
llm = LLM(
    model="ollama/qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0.7,
    api_key="ollama",
)

# ==================== TravelCrewFactory ====================
class TravelCrewFactory:
    AGENT_MAP = {
        "weather": "weather_expert",
        "attractions": "attractions_expert",
        "food": "food_expert",
        "budget": "budget_expert",
        "itinerary": "coordinator",
    }
    TASK_MAP = {
        "weather": "gather_weather",
        "attractions": "plan_attractions",
        "food": "plan_food",
        "budget": "estimate_budget",
        "itinerary": "create_itinerary",
    }

    def __init__(self, llm):
        self.llm = llm
        self.weather_tool = WeatherTool()
        # 自定义知识库搜索工具（无嵌入，无 ChromaDB）
        self.knowledge_tool = KnowledgeSearchTool(knowledge_dir=KNOWLEDGE_DIR)

    def _get_agent_config(self, key: str) -> dict:
        return AGENTS_CFG.get(self.AGENT_MAP.get(key, key), {})

    def _get_task_config(self, key: str) -> dict:
        return TASKS_CFG.get(self.TASK_MAP.get(key, key), {})

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

    # -------------------- Crew 工厂方法 --------------------
    def create_weather_crew(self, destination: str):
        agent = self._create_agent("weather", tools=[self.weather_tool])
        desc = self._fill_task_desc("weather", destination=destination)
        task = Task(description=desc, agent=agent,
                    expected_output=self._get_expected_output("weather"))
        return Crew(agents=[agent], tasks=[task], verbose=True)

    def create_budget_crew(self, destination: str, days: int, budget: str):
        agent = self._create_agent("budget")
        desc = self._fill_task_desc("budget", destination=destination, days=days, budget=budget)
        task = Task(description=desc, agent=agent,
                    expected_output=self._get_expected_output("budget"))
        return Crew(agents=[agent], tasks=[task], verbose=True)

    def create_attractions_crew(self, destination: str, days: int,
                                 preferences: str, weather_info: str,
                                 budget_info: str):
        tools = [self.knowledge_tool] if self.knowledge_tool else []
        agent = self._create_agent("attractions", tools=tools)
        base = self._fill_task_desc("attractions",
            destination=destination, days=days,
            preferences=preferences, budget_info=budget_info)
        full = (f"{base}\n"
                f"[天气信息]：{weather_info}\n"
                f"[预算约束]：{budget_info}\n"
                f"要求：每个景点标注区域位置+门票价格+活动强度+推荐理由。"
                f"你可以使用 Search Local Knowledge Base 工具搜索本地知识库获取更准确的景点信息。")
        task = Task(description=full, agent=agent,
                    expected_output=self._get_expected_output("attractions"))
        return Crew(agents=[agent], tasks=[task], verbose=True)

    def create_food_crew(self, destination: str, preferences: str,
                         weather_info: str, budget_info: str):
        tools = [self.knowledge_tool] if self.knowledge_tool else []
        agent = self._create_agent("food", tools=tools)
        base = self._fill_task_desc("food",
            destination=destination, preferences=preferences,
            budget_info=budget_info)
        full = (f"{base}\n"
                f"[天气信息]：{weather_info}\n"
                f"[预算约束]：{budget_info}\n"
                f"输出要求：每家餐厅标注区域、场景标签、适合餐段、人均、室内/户外。"
                f"你可以使用 Search Local Knowledge Base 工具搜索本地知识库获取准确的餐厅信息。")
        task = Task(description=full, agent=agent,
                    expected_output=self._get_expected_output("food"))
        return Crew(agents=[agent], tasks=[task], verbose=True)

    def create_itinerary_crew(self, destination: str, days: int, preferences: str,
                               weather_info: str, attractions_info: str,
                               food_info: str, budget_info: str):
        agent = self._create_agent("itinerary")
        base = self._fill_task_desc("itinerary",
            destination=destination, days=days, preferences=preferences,
            budget_info=budget_info, weather_info=weather_info,
            attractions_info=attractions_info, food_info=food_info)
        task = Task(description=base, agent=agent,
                    expected_output=self._get_expected_output("itinerary"))
        return Crew(agents=[agent], tasks=[task], verbose=True)


# ==================== TravelFlow ====================
class TravelFlow(Flow):
    def __init__(self, destination: str, days: int, preferences: str,
                 budget: str = "中等"):
        super().__init__()
        self.destination = destination
        self.days = days
        self.preferences = preferences
        self.budget = budget
        self.factory = TravelCrewFactory(llm)

    @start()
    async def collect_weather(self):
        result = await self.factory.create_weather_crew(
            self.destination
        ).kickoff_async()
        self.state["weather"] = str(result)
        return result

    @listen("collect_weather")
    async def collect_budget(self):
        result = await self.factory.create_budget_crew(
            self.destination, self.days, self.budget,
        ).kickoff_async()
        self.state["budget"] = str(result)
        return result

    @listen("collect_weather")
    async def collect_attractions(self):
        weather = self.state.get("weather", "")
        result = await self.factory.create_attractions_crew(
            self.destination, self.days, self.preferences, weather, self.budget,
        ).kickoff_async()
        self.state["attractions"] = str(result)
        return result

    @listen("collect_weather")
    async def collect_food(self):
        weather = self.state.get("weather", "")
        result = await self.factory.create_food_crew(
            self.destination, self.preferences, weather, self.budget,
        ).kickoff_async()
        self.state["food"] = str(result)
        return result

    @listen(and_("collect_budget", "collect_attractions", "collect_food"))
    async def combine(self, combined):
        weather = self.state.get("weather", "")
        budget = self.state.get("budget", "")
        attractions = self.state.get("attractions", "")
        food = self.state.get("food", "")
        result = await self.factory.create_itinerary_crew(
            self.destination, self.days, self.preferences,
            weather, attractions, food, budget,
        ).kickoff_async()
        return result


def run_travel_planning(destination: str, days: int = 3,
                        preferences: str = "", budget: str = "中等"):
    flow = TravelFlow(destination, days, preferences, budget)
    result = asyncio.run(flow.kickoff_async())
    print("\n" + "=" * 50)
    print("  最终行程单")
    print("=" * 50)
    print(str(result))
    return result

if __name__ == "__main__":
    run_travel_planning("三亚", days=3, preferences="喜欢潜水，怕晒", budget="中等")