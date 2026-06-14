#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from ai.crew import Ai

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

#保持简洁：不要在 main 文件里添加不必要的逻辑（例如复杂的业务处理、数据预处理等）。它应当只负责配置输入并启动 crew。
#便于本地测试：你需要手动替换文件中的示例输入数据，换成自己真正想测试的数据。
#自动插值：文件会“自动插值”（interpolate）任何任务（tasks）和代理（agents）的信息。这意味着你只需要提供原始输入，框架会自动将输入填充到任务描述、代理提示模板中的占位符（例如 {topic}、{input} 等）里。
def run():
    """
    Run the crew.
    """
    inputs = {
        'destination': '杭州',
        'days': '3'
    }
    try:
        result = Ai().crew().kickoff(inputs=inputs)  # 获取返回值
        save_result_to_file(str(result))  # 调用保存函数
    except Exception as e:
        raise Exception(f"运行 crew 时发生错误{e}")

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        Ai().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Ai().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }

    try:
        Ai().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = Ai().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
def save_result_to_file(content, filename="crew_output.txt"):
    """
    将内容保存到文件。
    - 使用 with open 和指定 encoding='utf-8'
    - 捕获可能的 IOError 或 Exception
    - 保存成功后打印 "结果已保存到 {filename}"
    - 如果失败，打印错误信息
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
            print(f"已经把结果保存在{filename}中")
    except IOError as e:
        print(f"错误原因: {e}")