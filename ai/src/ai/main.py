#!/usr/bin/env python
"""
============================================================
 main.py — CLI 入口
 接收命令行参数，调用 flow_crew.run_travel_planning()
============================================================
"""
import argparse          # 命令行参数解析
import sys               # 系统退出/错误输出
import warnings          # 过滤已知无害警告
from pathlib import Path # 路径处理

# ==================== 路径初始化：确保跨目录导入可用 ====================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # ai 项目根
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai.flow_crew import run_travel_planning  # 核心运行入口

# 过滤 pysbd 的 SyntaxWarning（CrewAI 依赖的句子分割库兼容性问题）
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


# ==================== 主命令：ai / run_crew ====================
def run():
    """运行旅行规划 Crew。
    支持命令行参数传入目的地、天数、偏好、预算。
    示例：
        ai --destination 三亚 --days 3 --preferences "喜欢潜水，怕晒" --budget "中等"
        ai -d 杭州 -n 2 -p "喜欢爬山和茶文化" -b "经济型2000元"
    """
    parser = argparse.ArgumentParser(
        description="CrewAI 多智能体旅行规划 — 根据目的地、天数、偏好、预算生成完整行程"
    )
    # --- 目的地 ---
    parser.add_argument(
        "--destination", "-d",
        default="三亚",
        help="旅行目的地城市名（默认：三亚）"
    )
    # --- 天数 ---
    parser.add_argument(
        "--days", "-n",
        type=int,
        default=3,
        help="计划旅行天数（默认：3）"
    )
    # --- 偏好 ---
    parser.add_argument(
        "--preferences", "-p",
        default="",
        help="用户的旅行偏好，如 '喜欢潜水，怕晒'（可选）"
    )
    # --- 预算（新增） ---
    parser.add_argument(
        "--budget", "-b",
        default="中等",
        help="旅行预算，如 '经济型3000元'、'中等'、'豪华5000元'（默认：中等）"
    )
    args = parser.parse_args()

    try:
        run_travel_planning(args.destination, args.days,
                            args.preferences, args.budget)
    except Exception as e:
        print(f"运行旅行规划时发生错误: {e}", file=sys.stderr)
        sys.exit(1)


# ==================== 已废弃命令（保留友好提示） ====================
def train():
    """已废弃：训练命令"""
    print("train 命令当前不可用，请使用 'ai -h' 查看可用命令", file=sys.stderr)
    sys.exit(1)


def replay():
    """已废弃：重放命令"""
    print("replay 命令当前不可用，请使用 'ai -h' 查看可用命令", file=sys.stderr)
    sys.exit(1)


def test():
    """已废弃：测试命令"""
    print("test 命令当前不可用，请使用 'ai -h' 查看可用命令", file=sys.stderr)
    sys.exit(1)


# ==================== JSON 触发入口 ====================
def run_with_trigger():
    """使用 JSON 载荷触发运行（适用于程序化调用场景）。
    示例：
        plan_with_trigger '{"destination":"杭州","days":2,"preferences":"喜欢爬山","budget":"经济型"}'
    """
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
        print("用法: plan_with_trigger '{\"destination\":\"杭州\",\"days\":2}'",
              file=sys.stderr)
        sys.exit(1)

    try:
        import json
        payload = json.loads(args.payload)
        destination = payload.get("destination", "三亚")
        days = payload.get("days", 3)
        preferences = payload.get("preferences", "")
        budget = payload.get("budget", "中等")        # 新增 budget 字段
        run_travel_planning(destination, days, preferences, budget)
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"运行旅行规划时发生错误: {e}", file=sys.stderr)
        sys.exit(1)
