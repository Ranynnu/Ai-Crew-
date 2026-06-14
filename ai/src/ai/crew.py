#!/usr/bin/env python
import sys
from pathlib import Path

from tools.weather_tool import WeatherTool
from crewai import Agent, Task, Crew, LLM, Process
from crewai.project import CrewBase, agent, task, crew

my_llm = LLM(
    model="ollama/qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0.7,
)

@CrewBase
class Ai:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def tourism_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["tourism_researcher"],
            llm=my_llm,
            tools=[WeatherTool()],
            verbose=True,
        )

    @agent
    def itinerary_planner(self) -> Agent:
        return Agent(
            config=self.agents_config["itinerary_planner"],
            llm=my_llm,
            verbose=True,
        )

    @task
    def collect_info(self) -> Task:
        return Task(
            config=self.tasks_config["collect_info"],
            agent=self.tourism_researcher(),
        )

    @task
    def plan_itinerary(self) -> Task:
        return Task(
            config=self.tasks_config["plan_itinerary"],
            agent=self.itinerary_planner(),
            context=[self.collect_info()],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
#if __name__ == "__main__":
#     ai_crew = Ai()
#     result = ai_crew.crew().kickoff()
#     print("最终结果：")
#     print(result)