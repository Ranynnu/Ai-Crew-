#!/usr/bin/env python
import argparse
import sys
import warnings
from pathlib import Path

# ———— 确保项目根目录在 sys.path 中 ————
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai.flow_crew import run_travel_planning

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    运行旅行规划 Crew。支持命令行参数传入目的地、天数、偏好。
    示例:
        ai --destination 三亚 --days 3 --preferences "喜欢潜水，怕晒"
        ai -d 杭州 -n 2
    """
    parser = argparse.ArgumentParser(
        description="CrewAI 多智能体旅行规划 —— 根据目的地、天数、偏好生成完整行程"
    )
    parser.add_argument(
        "--destination", "-d",
        default="三亚",
        help="旅行目的地城市名（默认：三亚）"
    )
    parser.add_argument(
        "--days", "-n",
        type=int,
        default=3,
        help="计划旅行天数（默认：3）"
    )
    parser.add_argument(
        "--preferences", "-p",
        default="",
        help="用户的旅行偏好，如 '喜欢潜水，怕晒'（可选）"
    )
    parser.add_argument(
        "--budget", "-b",
        default="中等",
        help="旅行预算，如 '经济型3000元'、'中等'、'豪华5000元'（默认：中等）"
    )
    args = parser.parse_args()

    try:
        run_travel_planning(args.destination, args.days, args.preferences, args.budget)
    except Exception as e:
        print(f"运行旅行规划时发生错误: {e}", file=sys.stderr)
        sys.exit(1)


def train():
    # 对已废弃命令保留提示
    print("train 命令当前不可用，请使用 'ai -h' 查看可用命令", file=sys.stderr)
    sys.exit(1)


def replay():
    print("replay 命令当前不可用，请使用 'ai -h' 查看可用命令", file=sys.stderr)
    sys.exit(1)


def test():
    print("test 命令当前不可用，请使用 'ai -h' 查看可用命令", file=sys.stderr)
    sys.exit(1)


def run_with_trigger():
    parser = argparse.ArgumentParser(
        description="使用 JSON 触发载荷运行旅行规划"
    )
    parser.add_argument(
        "payload",
        nargs="?",
        help="JSON 格式的触发载荷，如 '{\"destination\":\"三亚\",\"days\":3}'"
    )
    args = parser.parse_args()

    if not args.payload:
        print("错误：需要提供 JSON 参数", file=sys.stderr)
        print("用法: run_with_trigger '{\"destination\":\"儋州\",\"days\":2,\"preferences\":\"喜欢海鲜\"}'", file=sys.stderr)
        sys.exit(1)

    try:
        import json
        payload = json.loads(args.payload)
        destination = payload.get("destination", "三亚")
        days = payload.get("days", 3)
        preferences = payload.get("preferences", "")
        budget = payload.get("budget", "中等")
        run_travel_planning(destination, days, preferences, budget)
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"运行旅行规划时发生错误: {e}", file=sys.stderr)
        sys.exit(1)
