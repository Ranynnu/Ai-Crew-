# Ai Crew — 多智能体旅行规划

基于 [crewAI](https://crewai.com) Flow 并行架构构建的多智能体旅行规划系统。五位专业化 AI Agent（天气专家、景点策划、美食向导、预算规划师、行程总指挥）分工协作，为用户生成完整的定制旅行攻略。

## 技术栈

- **框架**：crewAI v1.14+（Flow 并行架构）
- **LLM**：[Ollama](https://ollama.com) 本地部署 `qwen2.5:7b`
- **天气 API**：[Open-Meteo](https://open-meteo.com)（免费，无需 API Key）
- **知识库**：本地文本文件（`.txt` / `.md`），关键词匹配搜索，无向量数据库依赖
- **包管理**：[UV](https://docs.astral.sh/uv/)
- **Python**：>=3.10, <3.14

## 快速开始

### 1. 安装 Ollama 并拉取模型

```bash
ollama pull qwen2.5:7b
```

### 2. 安装依赖

```bash
cd ai
uv sync
```

### 3. 运行

```bash
# 使用默认参数（目的地：三亚，天数：3，预算：中等）
uv run ai

# 自定义参数
uv run ai --destination 杭州 --days 4 --preferences "喜欢爬山，不吃辣" --budget "经济型3000元"

# JSON 触发（程序化调用）
uv run plan_with_trigger '{"destination":"杭州","days":2,"preferences":"喜欢爬山","budget":"经济型"}'

# 查看所有选项
uv run ai --help
```

## 项目结构

```
ai/
├── src/ai/
│   ├── main.py                  # CLI 入口（argparse）
│   ├── flow_crew.py             # Flow 编排 + TravelCrewFactory + 知识库工具
│   └── config/
│       ├── agents.yaml          # Agent 静态配置（role / goal / backstory）
│       └── tasks.yaml           # Task 静态配置（描述模板 / 期望输出）
├── src/knowledge/               # 知识库文档（.txt / .md）
│   └── 三亚旅游知识库.txt
├── tools/
│   ├── weather_tool.py          # 天气工具（Open-Meteo API，同步+异步）
│   └── custom_tool.py           # 自定义工具模板
├── pyproject.toml
└── README.md
```

## 架构说明

### Agent 体系（5 位 AI Agent）

| Agent | 角色 | 职责 |
|-------|------|------|
| 🌤️ 天气专家 | Weather Expert | 调用 Open-Meteo API 查询目的地实时天气，给出穿衣建议 |
| 🏛️ 景点策划 | Attractions Expert | 根据偏好和预算推荐景点池，标注区域/门票/强度/耗时 |
| 🍜 美食向导 | Food Expert | 推荐餐厅池，标注区域/场景/餐段/人均/室内外 |
| 💰 预算规划师 | Budget Expert | 解析用户预算，制定每日分项预算上限 |
| 📋 行程总指挥 | Coordinator | 整合所有专家成果，按耦合规则生成完整行程 |

### 执行流程

```
用户输入（destination, days, preferences, budget）
    │
    ▼
┌──────────────────┐
│  阶段1：天气查询  │  @start() → WeatherTool 调用 Open-Meteo API
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  阶段2：并行执行（均拿到天气结果）       │
│  ├─ @listen → 预算规划 Crew            │
│  ├─ @listen → 景点策划 Crew (知识库)    │
│  └─ @listen → 美食向导 Crew (知识库)    │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  阶段3：耦合整合  │  @listen(and_(budget, attractions, food))
└────────┬─────────┘
         │
         ▼
     最终行程攻略（Markdown）
```

### 混合架构

- **agents.yaml / tasks.yaml**：管理 Agent/Task 的**静态属性**（role, goal, backstory, 描述模板）
- **TravelCrewFactory**：工厂方法注入**动态参数**（destination, days, preferences, budget, weather_info）
- 修改 Agent 人设只需编辑 YAML，无需改动 Python 代码

### 自定义知识库搜索

系统使用 **关键词匹配搜索**（非向量检索），无需 embedding 模型或 ChromaDB：

- 将 `.txt` 或 `.md` 文件放入 `src/knowledge/` 目录，启动时自动加载
- 搜索时按段落匹配所有关键词，返回 Top-3 结果（每段截断 500 字）
- 无匹配时 Agent 自动回退到 LLM 自身知识生成

### 行程耦合规则

行程总指挥整合输出时强制执行以下耦合规则：

1. **早餐** → 靠近住宿区域，快捷地道
2. **午餐** → 与上午景点地理邻近（步行<1km），场景+天气匹配
3. **下午茶** → 仅当上午活动强度≥中等时出现
4. **晚餐** → 与下午景点或回程路线耦合
5. **宵夜** → 仅当有夜游/夜市时才推荐
6. 每天附带**花费小计**与预算对比

## 自定义配置

- 修改 `config/agents.yaml` — 调整 Agent 角色/目标/背景故事
- 修改 `config/tasks.yaml` — 调整任务描述、期望输出和耦合规则
- 修改 `flow_crew.py` — 切换 LLM 模型、温度、API 地址

## 添加知识库

在 `src/knowledge/` 目录下放入 `.txt` 或 `.md` 文件，系统启动时自动扫描加载。Agent 可通过 `Search Local Knowledge Base` 工具搜索知识库内容获取更准确的景点和餐厅信息。
