# Ai Crew — 多智能体旅行规划

基于 [crewAI](https://crewai.com) 构建的多智能体旅行规划系统。四位专业化 AI Agent（天气专家、景点策划、美食向导、行程总指挥）分工协作，为用户生成完整的定制旅行行程。

## 技术栈

- **框架**：crewAI v1.14+（Flow 并行架构）
- **LLM**：[Ollama](https://ollama.com) 本地部署 `qwen2.5:7b`
- **嵌入模型**：`nomic-embed-text`（用于知识库 RAG）
- **知识库**：支持 docx / pdf / xlsx / pptx / html / md 文档，由 Docling 解析 + 向量搜索
- **包管理**：[UV](https://docs.astral.sh/uv/)
- **Python**：>=3.10, <3.14

## 快速开始

### 1. 安装 Ollama 并拉取模型

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 2. 安装依赖

```bash
cd ai
uv sync
```

### 3. 运行

```bash
# 使用默认参数（目的地：三亚，天数：3）
uv run ai

# 自定义参数
uv run ai --destination 杭州 --days 4 --preferences "喜欢爬山，不吃辣"

# 查看所有选项
uv run ai --help
```

## 项目结构

```
ai/
├── src/ai/
│   ├── main.py              # CLI 入口（argparse）
│   ├── flow_crew.py         # Flow 编排 + 工厂类 + YAML 混合架构
│   └── config/
│       ├── agents.yaml      # Agent 静态配置（role / goal / backstory）
│       └── tasks.yaml       # Task 静态配置（描述模板 / 期望输出）
├── tools/
│   ├── weather_tool.py      # 天气工具（Open-Meteo API，同步+异步）
│   └── custom_tool.py       # 自定义工具模板
├── knowledge/               # 知识库文档（放 .docx / .pdf 等）
│   ├── 三亚旅游指南.docx
│   └── user_preference.txt
├── pyproject.toml
└── README.md
```

## 架构说明

```
用户输入（destination, days, preferences）
    │
    ▼
┌──────────────┐
│  阶段1：天气  │  @start() → WeatherTool 调用 Open-Meteo API
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────┐
│  阶段2：并行（均拿到天气结果）     │
│  ├─ @listen → 景点策划 Crew       │  ← 知识库 RAG 增强
│  └─ @listen → 美食向导 Crew       │  ← 知识库 RAG 增强
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────┐
│  阶段3：整合  │  @listen(and_(...)) → 行程总指挥
└──────┬───────┘
       │
       ▼
   最终行程单
```

### 混合架构

- **agents.yaml / tasks.yaml**：管 Agent/Task 的**静态属性**（role, goal, backstory, 描述模板）
- **TravelCrewFactory**：工厂方法注入**动态参数**（destination, days, preferences, weather_info）
- 修改 Agent 人设只需编辑 YAML，无需改动 Python 代码

## 添加知识库

在 `knowledge/` 目录下放入任意支持的文档（docx / pdf / xlsx / pptx / html / md），系统会自动扫描加载。无匹配时会自动回退到 Agent 的 LLM 知识生成。

## 自定义配置

- 修改 `config/agents.yaml` — 调整 Agent 角色/目标/背景
- 修改 `config/tasks.yaml` — 调整任务描述和期望输出
- 修改 `flow_crew.py` — 更改 LLM 模型、温度、嵌入模型
