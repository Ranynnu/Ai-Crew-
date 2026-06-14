"""
============================================================
 custom_tool.py — 自定义工具模板
 复制此文件作为新工具的起点
============================================================

CrewAI 工具契约：
- 继承 BaseTool
- 设置 name / description（Agent 据此判断何时调用）
- 实现 _run(self, ...) —— 同步路径
- 实现 _arun(self, ...) —— 异步路径（Flow kickoff_async）
"""
import httpx                        # HTTP 请求库（同步+异步）
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool   # CrewAI 工具基类


class MyCustomToolInput(BaseModel):
    """工具输入参数 Schema —— Agent 会据此传递参数"""
    argument: str = Field(..., description="Description of the argument.")


class MyCustomTool(BaseTool):
    """自定义工具模板 —— 替换 name/description 和 _run/_arun 实现即可使用"""

    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, "
        "your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    # ---------- 同步入口（CrewAI kickoff() 调用） ----------
    def _run(self, argument: str) -> str:
        """同步处理逻辑 —— 替换为实际实现"""
        return f"Processed: {argument}"

    # ---------- 异步入口（Flow kickoff_async() 优先调用） ----------
    async def _arun(self, argument: str) -> str:
        """异步处理逻辑 —— 使用 httpx.AsyncClient 等异步库"""
        return f"Processed (async): {argument}"
