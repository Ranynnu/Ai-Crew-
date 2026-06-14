"""
Custom tool template — add your own tools here.

CrewAI tool contract:
- Extend BaseTool
- Set name / description (used by the agent to decide when to call the tool)
- Implement _run(self, ...) for sync, and _arun(self, ...) for async
"""
import httpx
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")


class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, "
        "your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        """Sync path — replace with real logic."""
        return f"Processed: {argument}"

    async def _arun(self, argument: str) -> str:
        """Async path — replace with real logic (e.g. httpx.AsyncClient)."""
        return f"Processed (async): {argument}"
