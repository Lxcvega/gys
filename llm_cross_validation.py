"""
DeepSeek × OpenAI 双模型交叉验证工具
====================================

用途
----
将同一问题同时发给 DeepSeek 与 OpenAI 两个模型（并行），并让它们
互相审查对方的回答，从而实现交叉验证（一致性/可信度校验）。

前置条件
--------
1. 已安装 requests（本工具只用 requests + 标准库，无其他依赖）：
       pip install requests
2. 已配置密钥（二选一）：
   a) 系统环境变量  DEEPSEEK_API_KEY / OPENAI_API_KEY
   b) 项目根目录 .env 文件（参考 .env.example 复制后填写）

用法
----
    python llm_cross_validation.py --prompt "你的问题"
    python llm_cross_validation.py --prompt "..." --system "你是污水处理专家"
    python llm_cross_validation.py --file prompt.txt
    python llm_cross_validation.py --prompt "..." --no-critique --json
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Windows 控制台默认可能不是 UTF-8，强制使用 UTF-8 输出中文
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


# ---------------------------------------------------------------------------
# 提供方配置：DeepSeek 与 OpenAI 均兼容 OpenAI 的 chat/completions 协议
# ---------------------------------------------------------------------------
PROVIDERS = {
    "deepseek": {
        "label": "DeepSeek",
        "endpoint": "https://api.deepseek.com/chat/completions",
        "env_key": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
    },
    "openai": {
        "label": "OpenAI",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "env_key": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
    },
    "ikuncode": {
        "label": "ikuncode",
        "endpoint": "https://api.ikuncode.cc/v1/chat/completions",
        "env_key": "IKUNCODE_API_KEY",
        "model_env": "IKUNCODE_MODEL",
        "default_model": "codex-auto-review",
    },
}


def load_env(path: str = ".env") -> None:
    """极简 .env 加载器（不依赖 python-dotenv）。已存在的环境变量优先。"""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_config(provider: str) -> dict:
    """返回某个提供方解析后的配置（含密钥与模型名）。"""
    cfg = dict(PROVIDERS[provider])
    cfg["api_key"] = os.environ.get(cfg["env_key"], "").strip()
    cfg["model"] = os.environ.get(cfg["model_env"], "").strip() or cfg["default_model"]
    return cfg


def chat(provider: str, messages: list, temperature: float = 0.2,
         max_tokens: int = 2048) -> dict:
    """调用 OpenAI 兼容的 chat/completions 接口，返回统一结构。"""
    if requests is None:
        raise RuntimeError("缺少 requests 库，请先执行：pip install requests")

    cfg = resolve_config(provider)
    if not cfg["api_key"]:
        raise RuntimeError(
            f"未配置 {cfg['label']} 密钥：环境变量 {cfg['env_key']} 为空。"
            "请在 .env 或系统环境变量中设置。"
        )

    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }

    resp = requests.post(cfg["endpoint"], json=payload, headers=headers, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"{cfg['label']} 请求失败 (HTTP {resp.status_code}): {resp.text[:500]}"
        )

    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    return {
        "provider": provider,
        "label": cfg["label"],
        "model": cfg["model"],
        "content": content,
        "usage": data.get("usage", {}),
    }


def _build_messages(system: str, prompt: str) -> list:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def cross_validate(prompt: str, system: str = None,
                   providers: tuple = ("deepseek", "openai"),
                   temperature: float = 0.2,
                   mutual_critique: bool = True) -> dict:
    """同一问题发给多个模型，并让每个模型审查其余模型的回答。

    返回结构:
        {
          "answers":  {provider: {label, model, content, usage}},
          "errors":   {provider: 错误信息},
          "critique": {provider: 审查结论}
        }
    """
    answers, errors = {}, {}
    for p in providers:
        try:
            answers[p] = chat(p, _build_messages(system, prompt), temperature)
        except Exception as exc:  # noqa: BLE001 —— 单个模型失败不阻断整体
            errors[p] = str(exc)

    critique = {}
    if mutual_critique and len(answers) >= 2:
        for p, r in answers.items():
            others = {k: v["content"] for k, v in answers.items() if k != p}
            others_text = "\n\n---\n\n".join(
                f"【{k}】\n{v}" for k, v in others.items()
            )
            review_prompt = (
                "以下是另一个模型对同一问题的回答。请从「事实准确性、完整性、"
                "逻辑一致性」三个方面进行审查，指出其优点、错误或遗漏，"
                "并给出最终的共识结论：\n\n" + others_text
            )
            try:
                critique[p] = chat(
                    p, _build_messages(system, review_prompt), temperature=0.0
                )["content"]
            except Exception as exc:  # noqa: BLE001
                critique[p] = f"(审查失败: {exc})"

    return {"answers": answers, "errors": errors, "critique": critique}


def _print_report(result: dict, prompt: str) -> None:
    bar = "=" * 68
    print(bar)
    print("双模型交叉验证报告")
    print(bar)
    print(f"\n【问题】\n{prompt}\n")

    for p in PROVIDERS:
        label = PROVIDERS[p]["label"]
        if p in result["answers"]:
            r = result["answers"][p]
            print(f"\n◆ {label} ({r['model']}) 回答：")
            print("-" * 68)
            print(r["content"])
        elif p in result["errors"]:
            print(f"\n◆ {label} 调用失败：")
            print("-" * 68)
            print(result["errors"][p])

    if result["critique"]:
        print(f"\n\n{'-' * 68}")
        print("◆ 交叉审查（互相验证）")
        print("-" * 68)
        for p, text in result["critique"].items():
            print(f"\n>>> {PROVIDERS[p]['label']} 对另一模型的审查：")
            print(text)

    ok = len(result["answers"])
    total = ok + len(result["errors"])
    print(f"\n{bar}")
    print(f"完成：{ok}/{total} 个模型成功响应。")
    print(bar)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="DeepSeek × OpenAI 双模型交叉验证工具"
    )
    parser.add_argument("--prompt", "-p", help="要验证的问题")
    parser.add_argument("--file", "-f", help="从文本文件读取问题")
    parser.add_argument("--system", "-s", default=None, help="系统提示词（角色设定）")
    parser.add_argument(
        "--providers", default="deepseek,openai",
        help="参与交叉验证的模型，逗号分隔，默认 deepseek,openai"
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--no-critique", action="store_true",
                        help="跳过互相审查，只做并行回答")
    parser.add_argument("--json", action="store_true",
                        help="以 JSON 形式输出结果（供程序调用）")
    args = parser.parse_args(argv)

    load_env()

    # 组装问题
    if args.file:
        prompt = Path(args.file).read_text(encoding="utf-8").strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        print("请通过 --prompt 或 --file 提供问题。使用 --help 查看用法。")
        return 2

    providers = tuple(
        p.strip() for p in args.providers.split(",") if p.strip()
    )
    unknown = [p for p in providers if p not in PROVIDERS]
    if unknown:
        print(f"未知模型: {unknown}，可选: {list(PROVIDERS)}")
        return 2

    result = cross_validate(
        prompt,
        system=args.system,
        providers=providers,
        temperature=args.temperature,
        mutual_critique=not args.no_critique,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_report(result, prompt)

    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
