"""Agent 层:LLM 负责"想"(规划、诊断),执行引擎负责"做"。

- llm.py        LLM 客户端抽象(OpenAI 兼容 / Mock)+ 调用追踪
- planner.py    自然语言 → 结构化计划 → 护栏校验 → 失败回灌修复 → 提交引擎
- tools.py      Agent 可调用的受控工具(读/写分级、参数 schema)
- diagnosis.py  失败诊断 Agent(LangGraph:诊断 → 人工确认 interrupt → 执行)
"""
