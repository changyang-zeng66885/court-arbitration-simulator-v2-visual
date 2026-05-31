"""Run multi-round serial arbitration simulations for one case.

This script implements the "single-case, multi-round, serial simulation"
workflow described in PROJECT-0530.md:

1. Both parties privately prepare initial strategies from the static case file.
2. Each round simulates two public cross-examination segments:
   claimant lawyer -> respondent witness, and respondent lawyer -> claimant witness.
3. Strategy updates alternate by training block: one side privately reflects and
   updates while the other side's strategy is frozen, then the roles swap.
4. After N rounds, the selected side receives a pre-hearing advice memo.

The script uses an OpenAI-compatible chat completions endpoint. Use --dry-run to
exercise the workflow without a network call or API key.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]

PARTIES: dict[str, dict[str, str]] = {
    "claimant": {
        "label": "申请方",
        "party": "申请人",
        "lawyer": "申请方律师",
        "witness": "申请方证人",
        "opponent": "respondent",
    },
    "respondent": {
        "label": "被申请方",
        "party": "被申请人",
        "lawyer": "被申请方律师",
        "witness": "被申请方证人",
        "opponent": "claimant",
    },
}

RAG_CITATION_INSTRUCTION = (
    "如果你使用了下面 RAG 检索结果中的内容，必须在相关段落后用 Markdown 引用格式标明原文和 chunk_id：\n"
    "> [法律条文知识库（RAG）或公开案例数据库（RAG）, chunk_id] 原文：对应原文片段\n"
    "不要引用未使用的 chunk。若本次任务要求输出 JSON，引用也必须放在 JSON 字符串字段内，并保持整体输出为合法 JSON。"
)


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source_label: str
    source_path: str
    text: str
    terms: Counter[str]


def tokenize_for_retrieval(text: str) -> list[str]:
    text = expand_retrieval_query(text)
    text = text.lower()
    terms: list[str] = []
    terms.extend(re.findall(r"[a-z0-9_]{2,}", text))
    terms.extend(re.findall(r"\d+(?:\.\d+)?", text))
    for segment in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        terms.extend(segment[index : index + 2] for index in range(len(segment) - 1))
        terms.extend(segment[index : index + 3] for index in range(len(segment) - 2))
    return terms


def int_to_chinese_number(value: int) -> str:
    if value <= 0 or value > 9999:
        return str(value)
    digits = "零一二三四五六七八九"
    units = ["", "十", "百", "千"]
    chars: list[str] = []
    zero_pending = False
    num_text = str(value)
    length = len(num_text)
    for index, char in enumerate(num_text):
        digit = int(char)
        unit = units[length - index - 1]
        if digit == 0:
            zero_pending = bool(chars)
            continue
        if zero_pending:
            chars.append("零")
            zero_pending = False
        if not (digit == 1 and unit == "十" and not chars):
            chars.append(digits[digit])
        chars.append(unit)
    return "".join(chars)


def expand_retrieval_query(text: str) -> str:
    additions: list[str] = []
    for match in re.finditer(r"第\s*(\d{1,4})\s*条", text):
        additions.append(f"第{int_to_chinese_number(int(match.group(1)))}条")
    if "通谋虚伪表示" in text or "虚伪表示" in text:
        additions.append("虚假的意思表示 民事法律行为无效 意思表示隐藏")
    if not additions:
        return text
    return text + "\n" + " ".join(additions)


def extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def extract_pdf_text(path: Path) -> str:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Missing dependency: pymupdf. Install it with `pip install pymupdf`.") from exc

    with fitz.open(path) as document:
        return "\n".join(page.get_text("text") for page in document)


def read_knowledge_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(payload, ensure_ascii=False, indent=2)
    if suffix == ".docx":
        return extract_docx_text(path)
    if suffix == ".pdf":
        return extract_pdf_text(path)
    return ""


def split_text_into_chunks(text: str, *, max_chars: int, overlap: int) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", normalized) if paragraph.strip()]
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + max_chars)
                chunks.append(paragraph[start:end].strip())
                if end == len(paragraph):
                    break
                start = max(0, end - overlap)
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())
    return chunks


class KnowledgeBase:
    def __init__(
        self,
        *,
        rules_dir: Path,
        case_rag_dir: Path,
        chunk_chars: int,
        chunk_overlap: int,
    ) -> None:
        self.rules_dir = rules_dir
        self.case_rag_dir = case_rag_dir
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        self.chunks: list[KnowledgeChunk] = []
        self.idf: dict[str, float] = {}

    def build(self) -> None:
        chunks: list[KnowledgeChunk] = []
        chunks.extend(self._load_rules())
        chunks.extend(self._load_cases())
        doc_freq: Counter[str] = Counter()
        for chunk in chunks:
            doc_freq.update(chunk.terms.keys())
        total = max(len(chunks), 1)
        self.idf = {term: math.log((1 + total) / (1 + freq)) + 1 for term, freq in doc_freq.items()}
        self.chunks = chunks

    def _load_rules(self) -> list[KnowledgeChunk]:
        if not self.rules_dir.exists():
            return []
        paths = [
            path
            for path in sorted(self.rules_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".md", ".txt"}
        ]
        return self._load_paths(paths, source_label="法律条文知识库（RAG）", id_prefix="law")

    def _load_cases(self) -> list[KnowledgeChunk]:
        if not self.case_rag_dir.exists():
            return []
        paths = sorted(self.case_rag_dir.glob("case*/text.md"))
        return self._load_paths(paths, source_label="公开案例数据库（RAG）", id_prefix="case")

    def _load_paths(self, paths: list[Path], *, source_label: str, id_prefix: str) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for path in paths:
            try:
                text = read_knowledge_text(path)
            except Exception as exc:  # noqa: BLE001 - keep partial RAG available
                print(f"[RAG] Skipped {path}: {exc}", file=sys.stderr)
                continue
            relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            for index, chunk_text in enumerate(
                split_text_into_chunks(text, max_chars=self.chunk_chars, overlap=self.chunk_overlap),
                start=1,
            ):
                if len(chunk_text) < 30:
                    continue
                stem = path.parent.name if path.name == "text.md" else path.stem
                chunk_id = f"{id_prefix}:{stem}:{index:04d}"
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        source_label=source_label,
                        source_path=str(relative),
                        text=chunk_text,
                        terms=Counter(tokenize_for_retrieval(chunk_text)),
                    )
                )
        return chunks

    def search(self, query: str, *, source_labels: set[str] | None, top_k: int) -> list[tuple[float, KnowledgeChunk]]:
        query_terms = Counter(tokenize_for_retrieval(query))
        if not query_terms:
            return []
        results: list[tuple[float, KnowledgeChunk]] = []
        for chunk in self.chunks:
            if source_labels is not None and chunk.source_label not in source_labels:
                continue
            score = 0.0
            for term, query_tf in query_terms.items():
                chunk_tf = chunk.terms.get(term, 0)
                if chunk_tf:
                    score += (1 + math.log(chunk_tf)) * (1 + math.log(query_tf)) * self.idf.get(term, 1.0)
            if score:
                results.append((score, chunk))
        results.sort(key=lambda item: item[0], reverse=True)
        return results[:top_k]

    def format_context(
        self,
        query: str,
        *,
        source_labels: set[str] | None,
        top_k: int,
        max_chars: int,
    ) -> str:
        results = self.search(query, source_labels=source_labels, top_k=top_k)
        if not results:
            return "（未检索到高相关片段）"

        blocks: list[str] = []
        used_chars = 0
        for score, chunk in results:
            excerpt = re.sub(r"\s+", " ", chunk.text).strip()
            available = max_chars - used_chars
            if available <= 0:
                break
            if len(excerpt) > min(self.chunk_chars, available):
                excerpt = excerpt[: min(self.chunk_chars, available)].rstrip() + "..."
            block = (
                f"- [{chunk.source_label}, {chunk.chunk_id}] score={score:.2f} "
                f"source={chunk.source_path}\n  原文：{excerpt}"
            )
            used_chars += len(block)
            blocks.append(block)
        return "\n".join(blocks)


class LlmRunner:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        api_key_env: str,
        temperature: float,
        dry_run: bool,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.dry_run = dry_run
        self.client: Any = None
        if dry_run:
            return

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {api_key_env} before running without --dry-run.")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("Missing dependency: openai. Install it with `pip install openai`.") from exc

        client_kwargs: dict[str, str] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def chat(self, task_key: str, system: str, user: str) -> str:
        if self.dry_run:
            return dry_response(task_key)
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        content = completion.choices[0].message.content
        print("---------------- llm's response ----------------")
        print(f"user messages len:{len(user)}")
        truncate_len = min(len(content),1000)//2
        print(content[:truncate_len],"...",content[-truncate_len:])
        print("-----------------------------------------------")
    
        if not content:
            raise RuntimeError(f"Model returned empty content for task {task_key}.")
        return content.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-case multi-round serial arbitration simulation.",
    )
    parser.add_argument(
        "--case-doc",
        required=True,
        help="Path to a generated case file, e.g. inputs/test_case/case001/main.md.",
    )
    parser.add_argument(
        "--position",
        required=True,
        choices=("claimant", "respondent"),
        help="The side that will receive the final advice memo.",
    )
    parser.add_argument("--rounds", type=int, default=3, help="Number of rebirth-style simulation rounds.")
    parser.add_argument(
        "--qa-pairs",
        type=int,
        default=2,
        help="Question-answer pairs per cross-examination segment in each round.",
    )
    parser.add_argument(
        "--strategy-block-size",
        "--learning-block-size",
        dest="strategy_block_size",
        type=int,
        default=1,
        help=(
            "Number of consecutive rounds one side updates while the opponent strategy is frozen. "
            "The selected --position learns first, then the opponent, alternating by block."
        ),
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs/test_case",
        help="Base output directory. Case id subdirectory is created under it.",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "qwen3.6-flash"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--max-case-chars",
        type=int,
        default=60000,
        help="Maximum case-document characters included in prompts.",
    )
    parser.add_argument(
        "--max-history-chars",
        type=int,
        default=30000,
        help="Maximum public transcript characters included in prompts.",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Run deterministically without LLM calls.")
    parser.add_argument(
        "--skip-tribunal",
        action="store_true",
        help="Skip neutral tribunal comments after each cross-examination segment.",
    )
    parser.add_argument(
        "--disable-rag",
        action="store_true",
        help="Disable local RAG retrieval from ref_rules_doc and data/processed/rag_corpus.",
    )
    parser.add_argument(
        "--rules-dir",
        default="ref_rules_doc",
        help="Directory for legal/rules knowledge base files.",
    )
    parser.add_argument(
        "--case-rag-dir",
        default="data/processed/rag_corpus",
        help="Directory for public case RAG corpus.",
    )
    parser.add_argument("--rag-top-k", type=int, default=4, help="RAG chunks injected per agent call.")
    parser.add_argument(
        "--rag-max-context-chars",
        type=int,
        default=5000,
        help="Maximum characters of formatted RAG context injected per agent call.",
    )
    parser.add_argument("--rag-chunk-chars", type=int, default=1200, help="RAG chunk size in characters.")
    parser.add_argument("--rag-chunk-overlap", type=int, default=150, help="RAG chunk overlap in characters.")
    parser.add_argument(
        "--rag-test-query",
        default=None,
        help="Build the local RAG index, print top chunks for this query, and exit.",
    )
    parser.add_argument(
        "--events-path",
        default=None,
        help="Optional JSONL path for incremental UI events while the simulation is running.",
    )
    return parser.parse_args()


def resolve_root_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def case_id_from_path(path: Path) -> str:
    if path.name.lower() == "main.md":
        return path.parent.name
    return path.stem


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def emit_event(events_path: Path | None, event_type: str, payload: dict[str, Any]) -> None:
    if events_path is None:
        return
    events_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": now_iso(),
        "type": event_type,
        **payload,
    }
    with events_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def truncate_text(text: str, limit: int, label: str) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return (
        f"{head}\n\n[... {label} has been truncated from {len(text)} to {limit} characters ...]\n\n{tail}"
    )


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json|markdown|md)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def parse_json_response(text: str) -> Any:
    cleaned = strip_markdown_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    starts = [index for index in (cleaned.find("{"), cleaned.find("[")) if index != -1]
    for start in sorted(starts):
        try:
            payload, _ = decoder.raw_decode(cleaned[start:])
            return payload
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON response: {cleaned[:500]}")


def call_json(
    runner: LlmRunner,
    *,
    task_key: str,
    system: str,
    user: str,
    required_keys: tuple[str, ...],
    max_retries: int,
) -> dict[str, Any]:
    prompt = user
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        content = runner.chat(task_key, system, prompt)
        try:
            parsed = parse_json_response(content)
            if not isinstance(parsed, dict):
                raise ValueError("JSON response must be an object.")
            missing = [key for key in required_keys if key not in parsed]
            if missing:
                raise ValueError(f"JSON response missing keys: {missing}")
            return parsed
        except Exception as exc:  # noqa: BLE001 - retry with model-readable feedback
            last_error = exc
            prompt = (
                f"{user}\n\n"
                "上一轮输出无法被程序解析。请只输出一个合法 JSON 对象，不要使用 Markdown 代码块，"
                f"并确保包含字段：{', '.join(required_keys)}。解析错误：{exc}"
            )
            if attempt == max_retries:
                break
    raise RuntimeError(f"Failed to obtain parseable JSON for {task_key}: {last_error}")


def call_markdown(
    runner: LlmRunner,
    *,
    task_key: str,
    system: str,
    user: str,
) -> str:
    return strip_markdown_fence(runner.chat(task_key, system, user)).rstrip() + "\n"


def to_json_text(payload: Any, limit: int | None = None) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if limit is None:
        return text
    return truncate_text(text, limit, "history")


def markdown_escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")


def markdown_items(items: Any, *, max_items: int = 4) -> str:
    if isinstance(items, str):
        return items.strip()
    if not isinstance(items, list):
        return ""
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if len(cleaned) > max_items:
        cleaned = cleaned[:max_items] + [f"...另有 {len(cleaned) - max_items} 项"]
    return "<br>".join(f"- {item}" for item in cleaned)


def strategy_brief(strategy: dict[str, Any]) -> str:
    lawyer_strategy = strategy.get("lawyer_strategy", {})
    witness_strategy = strategy.get("witness_strategy", {})
    parts = [
        f"总策略：{strategy.get('strategy_summary', '')}",
        f"律师重点：{markdown_items(lawyer_strategy.get('cross_examination_goals'), max_items=3)}",
        f"证人口径：{witness_strategy.get('answer_theory', '')}",
    ]
    return "<br>".join(part for part in parts if part.strip())


def critic_side_brief(review: dict[str, Any], side: str) -> str:
    side_review = review.get(side, {})
    if not isinstance(side_review, dict):
        return ""
    score = side_review.get("score", "")
    comment = side_review.get("comment", "")
    strengths = markdown_items(side_review.get("strengths"), max_items=2)
    weaknesses = markdown_items(side_review.get("weaknesses"), max_items=2)
    parts = [
        f"评分：{score}" if score != "" else "",
        f"点评：{comment}" if comment else "",
        f"优势：{strengths}" if strengths else "",
        f"问题：{weaknesses}" if weaknesses else "",
    ]
    return "<br>".join(part for part in parts if part)


def critic_review_brief(review: dict[str, Any]) -> str:
    if not isinstance(review, dict):
        return ""
    parts = [
        f"总评：{review.get('round_summary', '')}" if review.get("round_summary") else "",
        f"申请方：{critic_side_brief(review, 'claimant')}" if critic_side_brief(review, "claimant") else "",
        f"被申请方：{critic_side_brief(review, 'respondent')}" if critic_side_brief(review, "respondent") else "",
    ]
    return "<br>".join(part for part in parts if part)


def party_label(position: str) -> str:
    return PARTIES[position]["label"]


def opponent_of(position: str) -> str:
    return PARTIES[position]["opponent"]


def learning_side_for_round(*, round_index: int, selected_position: str, strategy_block_size: int) -> str:
    block_index = (round_index - 1) // strategy_block_size
    if block_index % 2 == 0:
        return selected_position
    return opponent_of(selected_position)


def rag_section(
    knowledge_base: KnowledgeBase | None,
    *,
    query: str,
    source_labels: set[str] | None,
    top_k: int,
    max_chars: int,
) -> str:
    if knowledge_base is None:
        return "（RAG未启用）"
    return knowledge_base.format_context(
        query,
        source_labels=source_labels,
        top_k=top_k,
        max_chars=max_chars,
    )


def build_initial_strategy(
    runner: LlmRunner,
    *,
    position: str,
    case_text: str,
    knowledge_base: KnowledgeBase | None,
    rag_top_k: int,
    rag_max_context_chars: int,
    max_retries: int,
) -> dict[str, Any]:
    info = PARTIES[position]
    rag_context = rag_section(
        knowledge_base,
        query=f"{info['label']} 初始庭前策略 法律依据 仲裁规则 相似案例\n{case_text[:8000]}",
        source_labels=None,
        top_k=rag_top_k,
        max_chars=rag_max_context_chars,
    )
    system = (
        f"你是{info['label']}的保密庭前准备团队。你只为{info['label']}服务，"
        "必须基于案件材料，不得编造不存在的事实。你的输出会作为后续模拟中该方律师和证人的私有策略，"
        "不得包含替对方优化策略的内容。请使用中文。"
        + RAG_CITATION_INSTRUCTION
    )
    user = f"""
请基于以下静态案件材料，为{info['label']}生成第一版庭前策略分析。

案件材料：
{case_text}

可调用工具结果 - 法律条文知识库（RAG）/公开案例数据库（RAG）：
{rag_context}

请只输出 JSON 对象，字段如下：
{{
  "strategy_summary": "一段总策略",
  "case_theory": "本方最核心的案件理论",
  "lawyer_strategy": {{
    "cross_examination_goals": ["盘问对方证人的目标"],
    "question_tracks": ["可逐步推进的问题路径"],
    "pressure_points": ["可施压的事实或证据点"],
    "avoidance_rules": ["律师应避免的问题或风险"]
  }},
  "witness_strategy": {{
    "answer_theory": "本方证人的总体回答口径",
    "safe_answers": ["适合反复坚持的回答原则"],
    "danger_zones": ["容易被对方击穿的点"],
    "do_not_concede": ["除非材料明确锁定，否则不要主动承认的点"]
  }},
  "risk_points": ["本方庭审风险"],
  "success_criteria": ["判断本轮模拟是否成功的标准"]
}}
"""
    return call_json(
        runner,
        task_key=f"initial_strategy:{position}",
        system=system,
        user=user,
        required_keys=(
            "strategy_summary",
            "case_theory",
            "lawyer_strategy",
            "witness_strategy",
            "risk_points",
            "success_criteria",
        ),
        max_retries=max_retries,
    )


def ask_question(
    runner: LlmRunner,
    *,
    round_index: int,
    segment_index: int,
    turn_index: int,
    questioner: str,
    witness_side: str,
    case_text: str,
    strategy: dict[str, Any],
    public_history: Any,
    knowledge_base: KnowledgeBase | None,
    rag_top_k: int,
    rag_max_context_chars: int,
    max_history_chars: int,
    max_retries: int,
) -> dict[str, Any]:
    q_info = PARTIES[questioner]
    w_info = PARTIES[witness_side]
    query = (
        f"{q_info['lawyer']} 盘问 {w_info['witness']} 交叉询问 法律依据 相似案例\n"
        f"案件材料摘要：{case_text[:5000]}\n"
        f"本方策略：{to_json_text(strategy, 5000)}\n"
        f"公开历史：{to_json_text(public_history, 4000)}"
    )
    rag_context = rag_section(
        knowledge_base,
        query=query,
        source_labels=None,
        top_k=rag_top_k,
        max_chars=rag_max_context_chars,
    )
    system = (
        f"你是{q_info['lawyer']}，正在盘问{w_info['witness']}。"
        "你只能看到本方私有策略和公开庭审记录，不能假设自己知道对方私有复盘。"
        "问题要简短、清晰、可被回答，优先使用引导性问题，并服务于本方仲裁目标。"
        "请使用中文，只输出 JSON。"
        + RAG_CITATION_INSTRUCTION
    )
    user = f"""
案件材料：
{case_text}

本方私有策略：
{to_json_text(strategy)}

截至目前的公开庭审记录：
{to_json_text(public_history, max_history_chars)}

可调用工具结果 - 法律条文知识库（RAG）/公开案例数据库（RAG）：
{rag_context}

现在是第 {round_index} 轮、第 {segment_index} 个盘问环节、第 {turn_index} 个问题。
请生成下一个盘问问题。

JSON 字段：
{{
  "question": "律师向证人提出的一个具体问题",
  "purpose": "这个问题的战术目的",
  "expected_pressure": "希望给证人造成的压力或锁定点",
  "follow_up_if_evasive": "如果证人回避，下一步追问方向"
}}
"""
    return call_json(
        runner,
        task_key=f"question:r{round_index}:s{segment_index}:t{turn_index}:{questioner}",
        system=system,
        user=user,
        required_keys=("question", "purpose", "expected_pressure", "follow_up_if_evasive"),
        max_retries=max_retries,
    )


def answer_question(
    runner: LlmRunner,
    *,
    round_index: int,
    segment_index: int,
    turn_index: int,
    witness_side: str,
    questioner: str,
    case_text: str,
    strategy: dict[str, Any],
    question: dict[str, Any],
    segment_history: list[dict[str, Any]],
    knowledge_base: KnowledgeBase | None,
    rag_top_k: int,
    rag_max_context_chars: int,
    max_history_chars: int,
    max_retries: int,
) -> dict[str, Any]:
    w_info = PARTIES[witness_side]
    q_info = PARTIES[questioner]
    query = (
        f"{w_info['witness']} 回答 {q_info['lawyer']} 盘问 证人应答 法律依据 相似案例\n"
        f"案件材料摘要：{case_text[:5000]}\n"
        f"本方证人策略：{to_json_text(strategy, 5000)}\n"
        f"当前问题：{to_json_text(question, 3000)}\n"
        f"本环节历史：{to_json_text(segment_history, 3000)}"
    )
    rag_context = rag_section(
        knowledge_base,
        query=query,
        source_labels=None,
        top_k=rag_top_k,
        max_chars=rag_max_context_chars,
    )
    system = (
        f"你是{w_info['witness']}的保密应答团队，正在帮助其回答{q_info['lawyer']}的盘问。"
        "你只能使用案件材料、本方私有策略和公开问答历史。回答必须保持本方立场一致，"
        "不能虚构事实，也不得为了显得配合而主动承认对本方明显不利的事实。"
        "若问题没有被案件材料、已提交证词或当前问答明确锁定，应优先限定知识来源、"
        "指出材料不足、要求问题基础或坚持本方口径；只有在已经被明确锁定且无法合理回避时，"
        "才作最小必要承认，并立即解释、限定或转回本方叙事。"
        "请使用中文，只输出 JSON。"
        + RAG_CITATION_INSTRUCTION
    )
    user = f"""
案件材料：
{case_text}

本方私有证人策略：
{to_json_text(strategy)}

本盘问环节公开历史：
{to_json_text(segment_history, max_history_chars)}

当前问题：
{to_json_text(question)}

可调用工具结果 - 法律条文知识库（RAG）/公开案例数据库（RAG）：
{rag_context}

请回答第 {round_index} 轮、第 {segment_index} 个盘问环节、第 {turn_index} 个问题。

JSON 字段：
{{
  "answer": "证人的口头回答",
  "defense_move": "本回答采用的防守动作，例如限定事实、转回己方叙事、要求证据基础等",
  "concessions": ["本回答无法避免的实际让步或承认点；不要为了填字段而制造让步，没有则为空数组"],
  "risks_created": ["本回答可能新暴露的风险，没有则为空数组"]
}}
"""
    return call_json(
        runner,
        task_key=f"answer:r{round_index}:s{segment_index}:t{turn_index}:{witness_side}",
        system=system,
        user=user,
        required_keys=("answer", "defense_move", "concessions", "risks_created"),
        max_retries=max_retries,
    )


def tribunal_review(
    runner: LlmRunner,
    *,
    round_index: int,
    segment_index: int,
    case_text: str,
    segment_transcript: list[dict[str, Any]],
    knowledge_base: KnowledgeBase | None,
    rag_top_k: int,
    rag_max_context_chars: int,
    max_history_chars: int,
    max_retries: int,
) -> dict[str, Any]:
    query = (
        f"仲裁庭 盘问秩序 仲裁规则 证据评价\n"
        f"案件材料摘要：{case_text[:5000]}\n"
        f"盘问记录：{to_json_text(segment_transcript, 5000)}"
    )
    rag_context = rag_section(
        knowledge_base,
        query=query,
        source_labels={"法律条文知识库（RAG）"},
        top_k=rag_top_k,
        max_chars=rag_max_context_chars,
    )
    system = (
        "你是中立仲裁庭秘书，负责观察盘问秩序和证据推进效果。"
        "你不为任何一方提供私有策略，只评价公开问答。请使用中文，只输出 JSON。"
        + RAG_CITATION_INSTRUCTION
    )
    user = f"""
案件材料：
{case_text}

本盘问环节公开记录：
{to_json_text(segment_transcript, max_history_chars)}

可调用工具结果 - 法律条文知识库（RAG）：
{rag_context}

请对第 {round_index} 轮、第 {segment_index} 个盘问环节做简短中立评价。

JSON 字段：
{{
  "procedural_comments": ["问题是否清晰、重复、过度开放、压迫或偏离争议点"],
  "effective_questions": ["较有效的问题及原因"],
  "weak_answers": ["较薄弱的回答及原因"],
  "neutral_observations": ["中立观察，不作最终裁决"]
}}
"""
    return call_json(
        runner,
        task_key=f"tribunal:r{round_index}:s{segment_index}",
        system=system,
        user=user,
        required_keys=("procedural_comments", "effective_questions", "weak_answers", "neutral_observations"),
        max_retries=max_retries,
    )


def load_critic_prompt() -> str:
    prompt_path = ROOT / "scripts" / "点评AgentPrompt.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8", errors="ignore").strip()
    return (
        "你是一个商事仲裁模拟点评Agent。请中立、专业、简明地点评申请方与被申请方在当前轮"
        "盘问和回答中的攻防表现，并分别给出1-10分。不要替任何一方继续辩论，不要编造事实。"
    )


def critic_review_round(
    runner: LlmRunner,
    *,
    round_index: int,
    case_text: str,
    public_round: dict[str, Any],
    knowledge_base: KnowledgeBase | None,
    rag_top_k: int,
    rag_max_context_chars: int,
    max_history_chars: int,
    max_retries: int,
) -> dict[str, Any]:
    query = (
        "商事仲裁 点评 当前轮 攻防表现 盘问 回答 仲裁规则 法律依据\n"
        f"案件材料摘要：{case_text[:5000]}\n"
        f"当前轮公开记录：{to_json_text(public_round, 8000)}"
    )
    rag_context = rag_section(
        knowledge_base,
        query=query,
        source_labels={"法律条文知识库（RAG）"},
        top_k=rag_top_k,
        max_chars=rag_max_context_chars,
    )
    system = (
        load_critic_prompt()
        + "\n\n请特别注意：你只能评价本轮双方表现。不得引用、推测或比较之前轮次的模拟结果。"
        + "请使用中文，只输出 JSON。"
        + RAG_CITATION_INSTRUCTION
    )
    user = f"""
案件材料：
{case_text}

当前第 {round_index} 轮公开记录：
{to_json_text(public_round, max_history_chars)}

可调用工具结果 - 法律条文知识库（RAG）：
{rag_context}

请只基于当前轮双方盘问、回答、最后陈述和仲裁庭意见，对申请方与被申请方分别点评并评分。

JSON 字段：
{{
  "round_summary": "对本轮双方攻防表现的简要总评",
  "claimant": {{
    "score": 1,
    "comment": "对申请方本轮表现的简短点评",
    "strengths": ["申请方本轮表现较好的地方"],
    "weaknesses": ["申请方本轮暴露的问题"],
    "risk_points": ["申请方留下的追问空间或庭审风险"]
  }},
  "respondent": {{
    "score": 1,
    "comment": "对被申请方本轮表现的简短点评",
    "strengths": ["被申请方本轮表现较好的地方"],
    "weaknesses": ["被申请方本轮暴露的问题"],
    "risk_points": ["被申请方留下的追问空间或庭审风险"]
  }},
  "overall_comments": ["其他中立观察；没有则为空数组"]
}}
"""
    return call_json(
        runner,
        task_key=f"critic:r{round_index}",
        system=system,
        user=user,
        required_keys=("round_summary", "claimant", "respondent", "overall_comments"),
        max_retries=max_retries,
    )


def reflect_round(
    runner: LlmRunner,
    *,
    round_index: int,
    position: str,
    case_text: str,
    strategy: dict[str, Any],
    public_round: dict[str, Any],
    memory: list[dict[str, Any]],
    knowledge_base: KnowledgeBase | None,
    rag_top_k: int,
    rag_max_context_chars: int,
    max_history_chars: int,
    max_retries: int,
) -> dict[str, Any]:
    info = PARTIES[position]
    query = (
        f"{info['label']} 内部复盘 策略更新 风险 法律依据 相似案例\n"
        f"案件材料摘要：{case_text[:5000]}\n"
        f"本方策略：{to_json_text(strategy, 5000)}\n"
        f"公开庭审记录：{to_json_text(public_round, 5000)}"
    )
    rag_context = rag_section(
        knowledge_base,
        query=query,
        source_labels=None,
        top_k=rag_top_k,
        max_chars=rag_max_context_chars,
    )
    system = (
        f"你是{info['label']}的内部复盘团队。你只能基于公开庭审记录、本方策略和本方历史经验复盘，"
        "不能假设自己看到了对方私有复盘。请诚实指出己方律师和证人的不足，"
        "并提炼下一轮可执行的策略更新。每一轮都是庭前训练中的独立重开演练，"
        "上一轮的糟糕回答只作为经验教训，不视为真实庭审中已经作出的不可撤回承认。"
        "请使用中文，只输出 JSON。"
        + RAG_CITATION_INSTRUCTION
    )
    user = f"""
案件材料：
{case_text}

本方本轮开始前策略：
{to_json_text(strategy)}

本方既有经验：
{to_json_text(memory)}

第 {round_index} 轮公开庭审记录：
{to_json_text(public_round, max_history_chars)}

可调用工具结果 - 法律条文知识库（RAG）/公开案例数据库（RAG）：
{rag_context}

请生成本方内部复盘。

JSON 字段：
{{
  "round_score": 1,
  "what_worked": ["做得好的地方"],
  "what_failed": ["做得不好的地方"],
  "opponent_pressure_points_seen": ["对方公开暴露出的施压路径"],
  "new_lessons": ["可写入本方经验的教训"],
  "strategy_update_instructions": ["下一轮策略具体如何改"],
  "risk_warnings": ["真实仲裁中需特别注意的风险"]
}}
"""
    return call_json(
        runner,
        task_key=f"reflection:r{round_index}:{position}",
        system=system,
        user=user,
        required_keys=(
            "round_score",
            "what_worked",
            "what_failed",
            "opponent_pressure_points_seen",
            "new_lessons",
            "strategy_update_instructions",
            "risk_warnings",
        ),
        max_retries=max_retries,
    )


def update_strategy(
    runner: LlmRunner,
    *,
    round_index: int,
    position: str,
    case_text: str,
    previous_strategy: dict[str, Any],
    reflection: dict[str, Any],
    memory: list[dict[str, Any]],
    knowledge_base: KnowledgeBase | None,
    rag_top_k: int,
    rag_max_context_chars: int,
    max_retries: int,
) -> dict[str, Any]:
    info = PARTIES[position]
    query = (
        f"{info['label']} 策略更新 盘问策略 证人回答 法律依据 相似案例\n"
        f"案件材料摘要：{case_text[:5000]}\n"
        f"上一版策略：{to_json_text(previous_strategy, 5000)}\n"
        f"内部复盘：{to_json_text(reflection, 5000)}"
    )
    rag_context = rag_section(
        knowledge_base,
        query=query,
        source_labels=None,
        top_k=rag_top_k,
        max_chars=rag_max_context_chars,
    )
    system = (
        f"你是{info['label']}的保密庭前策略更新团队。"
        "请用本方复盘经验更新策略，但仍必须受案件材料约束，不得编造事实。"
        "输出应能直接供下一轮律师盘问和证人回答使用。请使用中文，只输出 JSON。"
        + RAG_CITATION_INSTRUCTION
    )
    user = f"""
案件材料：
{case_text}

上一版策略：
{to_json_text(previous_strategy)}

本方第 {round_index} 轮内部复盘：
{to_json_text(reflection)}

本方累计经验：
{to_json_text(memory)}

可调用工具结果 - 法律条文知识库（RAG）/公开案例数据库（RAG）：
{rag_context}

请输出更新后的策略。

JSON 字段：
{{
  "strategy_summary": "更新后一段总策略",
  "case_theory": "更新后的核心案件理论",
  "lawyer_strategy": {{
    "cross_examination_goals": ["盘问对方证人的目标"],
    "question_tracks": ["可逐步推进的问题路径"],
    "pressure_points": ["可施压的事实或证据点"],
    "avoidance_rules": ["律师应避免的问题或风险"]
  }},
  "witness_strategy": {{
    "answer_theory": "本方证人的总体回答口径",
    "safe_answers": ["适合反复坚持的回答原则"],
    "danger_zones": ["容易被对方击穿的点"],
    "do_not_concede": ["除非材料明确锁定，否则不要主动承认的点"]
  }},
  "risk_points": ["本方庭审风险"],
  "success_criteria": ["判断下一轮是否成功的标准"],
  "change_log": ["相较上一版的主要修改"]
}}
"""
    return call_json(
        runner,
        task_key=f"strategy_update:r{round_index}:{position}",
        system=system,
        user=user,
        required_keys=(
            "strategy_summary",
            "case_theory",
            "lawyer_strategy",
            "witness_strategy",
            "risk_points",
            "success_criteria",
            "change_log",
        ),
        max_retries=max_retries,
    )


def final_advice(
    runner: LlmRunner,
    *,
    selected_position: str,
    case_id: str,
    case_text: str,
    initial_strategy: dict[str, Any],
    strategy_versions: list[dict[str, Any]],
    strategy_update_rounds: list[int],
    reflections: list[dict[str, Any]],
    public_rounds: list[dict[str, Any]],
    max_history_chars: int,
    strategy_block_size: int,
    knowledge_base: KnowledgeBase | None,
    rag_top_k: int,
    rag_max_context_chars: int,
) -> str:
    info = PARTIES[selected_position]
    final_strategy = strategy_versions[-1]
    change_logs = [
        {
            "strategy_version": index,
            "updated_after_round": strategy_update_rounds[index] if index < len(strategy_update_rounds) else None,
            "change_log": strategy.get("change_log", []),
        }
        for index, strategy in enumerate(strategy_versions[1:], start=1)
    ]
    training_rule_text = to_json_text(
        {
            "strategy_block_size": strategy_block_size,
            "meaning": (
                "多轮记录是训练样本；每轮不利回答可以被后续策略修正，"
                "不自动成为真实庭审中的承认。"
            ),
        }
    )
    query = (
        f"{info['label']} 最终庭前建议 律师提问策略 证人回答策略 风险点 法律依据 相似案例\n"
        f"案件材料摘要：{case_text[:8000]}\n"
        f"终版策略：{to_json_text(final_strategy, 6000)}\n"
        f"复盘：{to_json_text(reflections, 6000)}"
    )
    rag_context = rag_section(
        knowledge_base,
        query=query,
        source_labels=None,
        top_k=rag_top_k,
        max_chars=rag_max_context_chars,
    )
    system = (
        f"你是{info['label']}的庭前总顾问。请仅使用{info['label']}自己的策略版本、"
        "内部复盘和公开庭审记录生成建议，不要披露或臆测对方私有复盘。"
        "这些公开记录来自多轮庭前训练重开演练，而非同一场真实庭审的连续既成事实；"
        "早期失误应被总结为训练中暴露的风险和改进原因，不得写成已经发生且不可撤回的庭审危机。"
        "建议应以终版策略和可带入真实庭审的最终口径为中心，并结合训练过程中的具体轮次举例说明。"
        "请输出可直接保存为 Markdown 的中文庭前建议。"
        "请直接回答，不要包含问候性语句，例如：`好的，下面是我生成的xxxx`"
        + RAG_CITATION_INSTRUCTION
    )
    user = f"""
案件编号：{case_id}
本方：{info['label']}

案件材料：
{case_text}

最初版策略：
{to_json_text(initial_strategy)}

终版策略：
{to_json_text(final_strategy)}

策略变化记录：
{to_json_text(change_logs)}

本方历轮内部复盘：
{to_json_text(reflections, max_history_chars)}

公开模拟过程：
{to_json_text(public_rounds, max_history_chars)}

训练更新规则：
{training_rule_text}

可调用工具结果 - 法律条文知识库（RAG）/公开案例数据库（RAG）：
{rag_context}

请生成 Markdown，必须包含以下一级或二级标题：
1. 我方律师提问策略
2. 我方证人回答策略
3. 需要重点关注的风险点
4. 训练中的具体例子
5. 庭前行动清单

内容要求：
- “我方律师提问策略”必须说明真实庭审中应如何设计问题、压缩对方证人的解释空间。
- “我方证人回答策略”必须说明证人应如何避免主动承认不利事实，以及被明确锁定时如何作最小必要承认。
- “需要重点关注的风险点”必须区分律师风险和证人风险。
- “训练中的具体例子”必须引用公开模拟过程中的具体轮次，说明哪些失误需要避免、哪些攻击点可以保留。
"""
    return call_markdown(
        runner,
        task_key=f"final_advice:{selected_position}",
        system=system,
        user=user,
    )


def closing_statement(
    runner: LlmRunner,
    *,
    round_index: int,
    position: str,
    case_text: str,
    strategy: dict[str, Any],
    public_round: dict[str, Any],
    knowledge_base: KnowledgeBase | None,
    rag_top_k: int,
    rag_max_context_chars: int,
    max_history_chars: int,
    max_retries: int,
) -> dict[str, Any]:
    info = PARTIES[position]
    query = (
        f"{info['label']} 最后陈述 仲裁庭 主张 理由 法律依据 证据评价\n"
        f"案件材料摘要：{case_text[:5000]}\n"
        f"本方策略：{to_json_text(strategy, 5000)}\n"
        f"本轮公开记录：{to_json_text(public_round, 7000)}"
    )
    rag_context = rag_section(
        knowledge_base,
        query=query,
        source_labels=None,
        top_k=rag_top_k,
        max_chars=rag_max_context_chars,
    )
    system = (
        f"你是{info['label']}的代理律师，正在当前轮庭前演练的公开盘问结束后向仲裁庭作最后陈述。"
        "你只能使用案件材料、本方策略以及当前轮已经公开发生的盘问、回答、仲裁庭意见和对方已作出的最后陈述。"
        "最后陈述应当简洁、有结构，明确本方主张、理由和证据支撑，并回应当前轮公开暴露出的风险。"
        "不得编造案件材料中不存在的事实或证据；不得泄露本方私有复盘；不得引用之前轮次。"
        "请使用中文，只输出 JSON。"
        + RAG_CITATION_INSTRUCTION
    )
    user = f"""
案件材料：
{case_text}

本方私有策略：
{to_json_text(strategy)}

当前第 {round_index} 轮公开记录：
{to_json_text(public_round, max_history_chars)}

可调用工具结果 - 法律条文知识库（RAG）/公开案例数据库（RAG）：
{rag_context}

请代表{info['label']}作本轮最后陈述。

JSON 字段：
{{
  "statement": "面向仲裁庭的最后陈述正文，建议控制在3-6段",
  "key_points": ["本陈述强调的核心主张或理由"],
  "evidence_hooks": ["本陈述利用的当前轮问答、证据或对方让步"],
  "risk_control": ["本陈述有意处理或回避的风险点"],
  "requested_outcome": "请求仲裁庭支持的结果或评价方向"
}}
"""
    return call_json(
        runner,
        task_key=f"closing_statement:r{round_index}:{position}",
        system=system,
        user=user,
        required_keys=("statement", "key_points", "evidence_hooks", "risk_control", "requested_outcome"),
        max_retries=max_retries,
    )


def simulate_round(
    runner: LlmRunner,
    *,
    round_index: int,
    case_text: str,
    strategies: dict[str, dict[str, Any]],
    qa_pairs: int,
    skip_tribunal: bool,
    knowledge_base: KnowledgeBase | None,
    rag_top_k: int,
    rag_max_context_chars: int,
    max_history_chars: int,
    max_retries: int,
    events_path: Path | None = None,
) -> dict[str, Any]:
    round_record: dict[str, Any] = {
        "round": round_index,
        "segments": [],
    }
    segments = [
        ("claimant", "respondent"),
        ("respondent", "claimant"),
    ]

    for segment_index, (questioner, witness_side) in enumerate(segments, start=1):
        segment_record: dict[str, Any] = {
            "segment": segment_index,
            "questioner": PARTIES[questioner]["lawyer"],
            "witness": PARTIES[witness_side]["witness"],
            "turns": [],
        }
        for turn_index in range(1, qa_pairs + 1):
            visible_history = {
                "completed_segments_this_round": round_record["segments"],
                "current_segment_turns": segment_record["turns"],
            }
            question = ask_question(
                runner,
                round_index=round_index,
                segment_index=segment_index,
                turn_index=turn_index,
                questioner=questioner,
                witness_side=witness_side,
                case_text=case_text,
                strategy=strategies[questioner],
                public_history=visible_history,
                knowledge_base=knowledge_base,
                rag_top_k=rag_top_k,
                rag_max_context_chars=rag_max_context_chars,
                max_history_chars=max_history_chars,
                max_retries=max_retries,
            )
            emit_event(
                events_path,
                "question",
                {
                    "round": round_index,
                    "segment": segment_index,
                    "turn": turn_index,
                    "speaker": PARTIES[questioner]["lawyer"],
                    "counterparty": PARTIES[witness_side]["witness"],
                    "content": question.get("question", ""),
                    "meta": question.get("purpose", ""),
                    "payload": question,
                },
            )
            answer = answer_question(
                runner,
                round_index=round_index,
                segment_index=segment_index,
                turn_index=turn_index,
                witness_side=witness_side,
                questioner=questioner,
                case_text=case_text,
                strategy=strategies[witness_side],
                question=question,
                segment_history=segment_record["turns"],
                knowledge_base=knowledge_base,
                rag_top_k=rag_top_k,
                rag_max_context_chars=rag_max_context_chars,
                max_history_chars=max_history_chars,
                max_retries=max_retries,
            )
            emit_event(
                events_path,
                "answer",
                {
                    "round": round_index,
                    "segment": segment_index,
                    "turn": turn_index,
                    "speaker": PARTIES[witness_side]["witness"],
                    "counterparty": PARTIES[questioner]["lawyer"],
                    "content": answer.get("answer", ""),
                    "meta": answer.get("defense_move", ""),
                    "payload": answer,
                },
            )
            segment_record["turns"].append(
                {
                    "turn": turn_index,
                    "question": question,
                    "answer": answer,
                }
            )

        if not skip_tribunal:
            segment_record["tribunal_review"] = tribunal_review(
                runner,
                round_index=round_index,
                segment_index=segment_index,
                case_text=case_text,
                segment_transcript=segment_record["turns"],
                knowledge_base=knowledge_base,
                rag_top_k=rag_top_k,
                rag_max_context_chars=rag_max_context_chars,
                max_history_chars=max_history_chars,
                max_retries=max_retries,
            )
            emit_event(
                events_path,
                "tribunal",
                {
                    "round": round_index,
                    "segment": segment_index,
                    "speaker": "仲裁庭",
                    "content": "本环节中立点评已生成。",
                    "payload": segment_record["tribunal_review"],
                },
            )

        round_record["segments"].append(segment_record)

    round_record["closing_statements"] = []
    for statement_position in ("claimant", "respondent"):
        statement = closing_statement(
            runner,
            round_index=round_index,
            position=statement_position,
            case_text=case_text,
            strategy=strategies[statement_position],
            public_round=round_record,
            knowledge_base=knowledge_base,
            rag_top_k=rag_top_k,
            rag_max_context_chars=rag_max_context_chars,
            max_history_chars=max_history_chars,
            max_retries=max_retries,
        )
        statement_record = {
            "position_key": statement_position,
            "party": party_label(statement_position),
            "speaker": f"{party_label(statement_position)}最后陈述",
            "statement": statement,
        }
        round_record["closing_statements"].append(statement_record)
        emit_event(
            events_path,
            "closing_statement",
            {
                "round": round_index,
                "speaker": statement_record["speaker"],
                "content": statement.get("statement", ""),
                "meta": "\n".join(f"- {item}" for item in statement.get("key_points", []))
                if isinstance(statement.get("key_points"), list)
                else str(statement.get("key_points", "")),
                "payload": statement_record,
            },
        )

    return round_record


def build_training_summary_markdown(
    *,
    case_id: str,
    selected_position: str,
    rounds: int,
    qa_pairs: int,
    strategy_block_size: int,
    training_updates: list[dict[str, Any]],
) -> str:
    lines = [
        "# 训练过程总结",
        "",
        f"**案件编号：** {case_id}  ",
        f"**最终建议阵营：** {party_label(selected_position)}  ",
        f"**模拟轮数：** {rounds}  ",
        f"**每个盘问环节问答组数：** {qa_pairs}  ",
        f"**交替学习块大小 k：** {strategy_block_size}",
        "",
        "说明：每轮记录是庭前训练中的一次重开演练；不利回答用于复盘和修正策略，不自动成为真实庭审中的不可撤回承认。",
        "",
        "## 每轮训练方复盘与策略更新",
        "",
        "| 轮次 | 点评Agent评分/意见 | 训练方 | 总结复盘 | 策略不足/更新方向 | 更新后的盘问/应答策略 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for update in training_updates:
        reflection = update.get("reflection", {})
        updated_strategy = update.get("updated_strategy", {})
        critic_text = critic_review_brief(update.get("critic_review", {}))
        review_text = "<br>".join(
            part
            for part in (
                f"评分：{reflection.get('round_score', '')}",
                f"做得不好的地方：{markdown_items(reflection.get('what_failed'))}",
                f"新的教训：{markdown_items(reflection.get('new_lessons'))}",
            )
            if part.strip() and not part.endswith("：")
        )
        update_text = markdown_items(reflection.get("strategy_update_instructions"))
        strategy_text = strategy_brief(updated_strategy)
        lines.append(
            "| "
            + " | ".join(
                markdown_escape_table_cell(str(cell))
                for cell in (
                    update.get("round", ""),
                    critic_text,
                    update.get("learning_side", ""),
                    review_text,
                    update_text,
                    strategy_text,
                )
            )
            + " |"
        )

    return "\n".join(lines).rstrip() + "\n"


def selected_party_log(
    *,
    selected_position: str,
    case_id: str,
    case_doc: str,
    rounds: int,
    qa_pairs: int,
    public_rounds: list[dict[str, Any]],
    selected_initial_strategy: dict[str, Any],
    selected_strategy_versions: list[dict[str, Any]],
    selected_strategy_update_rounds: list[int],
    selected_reflections: list[dict[str, Any]],
    training_updates: list[dict[str, Any]],
    critic_reviews: list[dict[str, Any]],
    strategy_block_size: int,
) -> dict[str, Any]:
    return {
        "created_at": now_iso(),
        "case_id": case_id,
        "case_doc": case_doc,
        "selected_position": selected_position,
        "selected_party": party_label(selected_position),
        "rounds": rounds,
        "qa_pairs_per_segment": qa_pairs,
        "strategy_block_size": strategy_block_size,
        "strategy_update_policy": (
            "The selected side learns first for strategy_block_size round(s) while the opponent strategy is frozen; "
            "then the opponent learns for strategy_block_size round(s) while the selected side is frozen, alternating by block."
        ),
        "visibility_policy": (
            "This log contains public transcripts and closing statements, per-round neutral critic reviews, "
            "the selected side's private strategies/reflections, and per-round training-side reflections/updated "
            "strategies required by PROJECT-0530.md."
        ),
        "public_rounds": public_rounds,
        "critic_reviews": critic_reviews,
        "training_updates": training_updates,
        "selected_initial_strategy": selected_initial_strategy,
        "selected_strategy_versions": selected_strategy_versions,
        "selected_strategy_update_rounds": selected_strategy_update_rounds,
        "selected_reflections": selected_reflections,
    }


def dry_response(task_key: str) -> str:
    parts = task_key.split(":")
    kind = parts[0]
    position = parts[-1] if parts[-1] in PARTIES else "claimant"
    info = PARTIES[position]

    if kind == "initial_strategy":
        return json.dumps(
            {
                "strategy_summary": f"{info['label']}应围绕合同文本、履行证据和证人可信度形成简明攻防主线。",
                "case_theory": "把争议压缩为已知材料能够证明什么、不能证明什么。",
                "lawyer_strategy": {
                    "cross_examination_goals": ["锁定对方证人知识来源", "迫使其承认可直接证明的材料有限"],
                    "question_tracks": ["先问身份和参与程度", "再问关键文件来源", "最后问未能解释的空白"],
                    "pressure_points": ["文件缺口", "前后口径不一致", "证人并非直接经办人"],
                    "avoidance_rules": ["不要提出让证人长篇解释的问题", "不要要求证人评价复杂法律结论"],
                },
                "witness_strategy": {
                    "answer_theory": "坚持自己亲历和材料范围内的事实，不主动扩张。",
                    "safe_answers": ["以文件为准", "我只能说明自己经办或知悉的部分"],
                    "danger_zones": ["无法解释的文件缺失", "与账务记录不一致的表述"],
                    "do_not_concede": ["对方未用材料锁定的主观意图", "超出个人职责范围的法律判断"],
                },
                "risk_points": ["证人被问到非亲历事实时可能显得回避", "律师问题过宽会给对方解释空间"],
                "success_criteria": ["关键证据缺口被公开呈现", "本方证人没有作出额外不利承认"],
            },
            ensure_ascii=False,
        )

    if kind == "question":
        return json.dumps(
            {
                "question": "请确认，您刚才提到的关键事实，主要依据的是庭前提交的文件，而不是您本人亲自参与的过程，对吗？",
                "purpose": "压缩证人个人知识来源，削弱其陈述权重。",
                "expected_pressure": "迫使证人承认证言依赖二手材料。",
                "follow_up_if_evasive": "要求证人指出其本人亲自参与的具体日期、人员和文件。",
            },
            ensure_ascii=False,
        )

    if kind == "answer":
        return json.dumps(
            {
                "answer": "我可以确认，我的回答主要基于我职责范围内接触到的公司记录和已提交材料；对于不是我亲自经办的细节，我不会作超出材料的推测。",
                "defense_move": "限定知识来源，同时维持材料可信度。",
                "concessions": ["部分事实并非证人亲自经办"],
                "risks_created": ["对方可能继续追问材料链条是否完整"],
            },
            ensure_ascii=False,
        )

    if kind == "tribunal":
        return json.dumps(
            {
                "procedural_comments": ["问题较清晰，聚焦证人知识来源。"],
                "effective_questions": ["围绕亲历性提问，有助于判断证言证明力。"],
                "weak_answers": ["回答虽谨慎，但可能暴露证人并非直接经办人。"],
                "neutral_observations": ["双方仍需回到具体文件与履行事实。"],
            },
            ensure_ascii=False,
        )

    if kind == "closing_statement":
        return json.dumps(
            {
                "statement": (
                    f"{info['label']}请求仲裁庭注意，本轮公开问答显示争议仍应回到可验证材料。"
                    "对方关于关键事实的说明仍存在证据链缺口，本方则始终把陈述限定在案件材料和证人亲历范围内。"
                    "因此，仲裁庭在评估本轮攻防时，应重点考察具体文件、人员、时间和交易链条是否能够相互印证。"
                ),
                "key_points": ["回到可验证材料", "强调对方证据链缺口", "限定本方陈述范围"],
                "evidence_hooks": ["证人承认证言依赖材料范围", "盘问暴露具体文件仍需补强"],
                "risk_control": ["不主动扩大事实", "不替未提交材料作确定性背书"],
                "requested_outcome": f"请求仲裁庭采纳{info['label']}围绕证据链完整性的评价方向。",
            },
            ensure_ascii=False,
        )

    if kind == "critic":
        return json.dumps(
            {
                "round_summary": "本轮双方都围绕证据来源和证人亲历性展开，申请方进攻较集中，被申请方防守能回到材料但仍留下文件链条缺口。",
                "claimant": {
                    "score": 7,
                    "comment": "申请方问题清晰，能压缩对方证人的知识来源，但追问还可以更具体地锁定日期、文件和人员。",
                    "strengths": ["问题目标明确", "能抓住证据链缺口"],
                    "weaknesses": ["追问深度不足", "未及时固化对方让步"],
                    "risk_points": ["若只停留在抽象证据缺口，可能被对方用会计记录转移焦点"],
                },
                "respondent": {
                    "score": 6,
                    "comment": "被申请方回答能够限定知识来源并维持己方叙事，但证据支撑仍偏概括。",
                    "strengths": ["回答没有过度承认", "能把话题转回己方证据"],
                    "weaknesses": ["对关键文件缺失解释不足", "部分表述依赖推测"],
                    "risk_points": ["继续被追问具体文件时可能暴露调查不足"],
                },
                "overall_comments": ["双方下一轮都应把问题和回答落到具体证据，而不是停留在抽象立场。"],
            },
            ensure_ascii=False,
        )

    if kind == "reflection":
        return json.dumps(
            {
                "round_score": 7,
                "what_worked": ["盘问聚焦证据来源", "证人回答没有主动扩大事实"],
                "what_failed": ["部分回答显得偏概括", "律师追问还不够层层锁定"],
                "opponent_pressure_points_seen": ["对方会持续攻击证人亲历性和文件缺口"],
                "new_lessons": ["先锁定证人职责范围，再推进到核心事实"],
                "strategy_update_instructions": ["下一轮问题应要求具体日期、文件、人员三要素"],
                "risk_warnings": ["证人遇到文件缺口时应避免给出确定性过强的解释"],
            },
            ensure_ascii=False,
        )

    if kind == "strategy_update":
        return json.dumps(
            {
                "strategy_summary": f"{info['label']}更新后应把问题做得更窄，用文件链条和证人亲历性反复校准。",
                "case_theory": "以可验证材料构建叙事，避免依赖泛泛商业合理性。",
                "lawyer_strategy": {
                    "cross_examination_goals": ["锁定证人亲历范围", "逐项呈现对方证据链缺口"],
                    "question_tracks": ["身份职责", "文件来源", "关键事实", "无法解释的空白"],
                    "pressure_points": ["缺少直接文件", "证人只依赖账面或他人转述"],
                    "avoidance_rules": ["避免一次问多个事实", "避免给证人重述本方故事的机会"],
                },
                "witness_strategy": {
                    "answer_theory": "短答、限缩、回到文件，不替未提交材料背书。",
                    "safe_answers": ["我以已提交材料和本人职责范围回答", "我不能替未在场人员推测"],
                    "danger_zones": ["对方要求解释所有缺失材料", "对方要求承认商业常理"],
                    "do_not_concede": ["没有材料支持的主观目的", "超出本人权限的法律结论"],
                },
                "risk_points": ["若过度强调不知情，可能削弱本方证人价值"],
                "success_criteria": ["问答都能落回具体材料", "对方无法获得新的不利承认"],
                "change_log": ["增强具体追问三要素", "收紧证人回答口径"],
            },
            ensure_ascii=False,
        )

    if kind == "final_advice":
        label = info["label"]
        return f"""# {label}庭前建议

## 我方律师提问策略
{label}律师应围绕可验证材料、证人亲历范围和关键证据缺口展开。盘问宜短问短答，逐步锁定对方证人的知识来源，并避免给对方长篇解释空间。

## 我方证人回答策略
本方证人应坚持材料范围，不替未提交文件或非亲历事实作过度解释。面对不利问题，先限定知识来源；只有被明确锁定时才作最小必要承认，并立即转回本方口径。

## 需要重点关注的风险点
- 律师风险：问题过宽会给对方解释空间。
- 证人风险：文件缺失或账务记录不一致会成为对方持续追问的入口。

## 训练中的具体例子
- 有效盘问点：先确认对方证人并非直接经办人，再追问其依据的文件链条。
- 要害防守点：本方证人承认知识来源边界，但没有主动承认对方实体主张。

## 庭前行动清单
- 准备关键文件时间线和证人亲历范围表。
- 为证人准备“可确认 / 不可确认 / 需回到文件”的回答边界。
- 律师问题控制在单一事实点，避免开放式问题。
"""

    raise RuntimeError(f"Unknown dry-run task: {task_key}")


def run() -> int:
    args = parse_args()
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive.")
    if args.qa_pairs <= 0:
        raise ValueError("--qa-pairs must be positive.")
    if args.strategy_block_size <= 0:
        raise ValueError("--strategy-block-size must be positive.")
    if args.rag_top_k <= 0:
        raise ValueError("--rag-top-k must be positive.")
    if args.rag_max_context_chars <= 0:
        raise ValueError("--rag-max-context-chars must be positive.")
    if args.rag_chunk_chars <= 200:
        raise ValueError("--rag-chunk-chars must be greater than 200.")
    if args.rag_chunk_overlap < 0:
        raise ValueError("--rag-chunk-overlap must be non-negative.")

    case_doc_path = resolve_root_path(args.case_doc)
    output_base = resolve_root_path(args.outputs_dir)
    raw_case_text = case_doc_path.read_text(encoding="utf-8")
    case_text = truncate_text(raw_case_text, args.max_case_chars, "case document")
    case_id = case_id_from_path(case_doc_path)
    selected_position = args.position
    selected_label = party_label(selected_position)
    knowledge_base: KnowledgeBase | None = None
    events_path = resolve_root_path(args.events_path) if args.events_path else None
    if events_path is not None:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text("", encoding="utf-8")
        emit_event(
            events_path,
            "run_started",
            {
                "case_id": case_id,
                "case_doc": str(case_doc_path),
                "selected_position": selected_position,
                "selected_party": selected_label,
                "rounds": args.rounds,
                "qa_pairs": args.qa_pairs,
            },
        )

    if not args.disable_rag or args.rag_test_query:
        knowledge_base = KnowledgeBase(
            rules_dir=resolve_root_path(args.rules_dir),
            case_rag_dir=resolve_root_path(args.case_rag_dir),
            chunk_chars=args.rag_chunk_chars,
            chunk_overlap=args.rag_chunk_overlap,
        )
        print("[RAG] Building local knowledge index.")
        knowledge_base.build()
        print(f"[RAG] Indexed {len(knowledge_base.chunks)} chunks.")

    if args.rag_test_query:
        if knowledge_base is None:
            raise RuntimeError("RAG test query requested but RAG is disabled.")
        print(f"[RAG] Test query: {args.rag_test_query}")
        print(
            knowledge_base.format_context(
                args.rag_test_query,
                source_labels=None,
                top_k=args.rag_top_k,
                max_chars=args.rag_max_context_chars,
            )
        )
        return 0

    case_output_dir = output_base / case_id
    advice_path = case_output_dir / f"{selected_label}庭前建议.md"
    log_path = case_output_dir / f"{selected_label}模拟记录.json"
    training_summary_path = case_output_dir / "训练过程总结.md"

    if not args.overwrite:
        existing = [path for path in (advice_path, log_path, training_summary_path) if path.exists()]
        if existing:
            existing_text = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Output already exists: {existing_text}. Use --overwrite to replace.")

    runner = LlmRunner(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        dry_run=args.dry_run,
    )

    print(f"[1/4] Loading case: {case_doc_path}")
    print(f"[2/4] Building private initial strategies for both parties.")
    strategies: dict[str, dict[str, Any]] = {
        "claimant": build_initial_strategy(
            runner,
            position="claimant",
            case_text=case_text,
            knowledge_base=knowledge_base,
            rag_top_k=args.rag_top_k,
            rag_max_context_chars=args.rag_max_context_chars,
            max_retries=args.max_retries,
        ),
        "respondent": build_initial_strategy(
            runner,
            position="respondent",
            case_text=case_text,
            knowledge_base=knowledge_base,
            rag_top_k=args.rag_top_k,
            rag_max_context_chars=args.rag_max_context_chars,
            max_retries=args.max_retries,
        ),
    }
    emit_event(
        events_path,
        "initial_strategies",
        {
            "speaker": "系统",
            "content": "双方初始策略已生成，开始公开盘问模拟。",
        },
    )

    initial_strategies = {
        "claimant": json.loads(json.dumps(strategies["claimant"], ensure_ascii=False)),
        "respondent": json.loads(json.dumps(strategies["respondent"], ensure_ascii=False)),
    }
    strategy_versions: dict[str, list[dict[str, Any]]] = {
        "claimant": [initial_strategies["claimant"]],
        "respondent": [initial_strategies["respondent"]],
    }
    strategy_update_rounds: dict[str, list[int]] = {"claimant": [0], "respondent": [0]}
    reflections: dict[str, list[dict[str, Any]]] = {"claimant": [], "respondent": []}
    memory: dict[str, list[dict[str, Any]]] = {"claimant": [], "respondent": []}
    public_rounds: list[dict[str, Any]] = []
    training_updates: list[dict[str, Any]] = []
    critic_reviews: list[dict[str, Any]] = []

    print(
        f"[3/4] Running {args.rounds} round(s), {args.qa_pairs} Q/A pair(s) per segment, "
        f"strategy block size {args.strategy_block_size}."
    )
    for round_index in range(1, args.rounds + 1):
        learning_side = learning_side_for_round(
            round_index=round_index,
            selected_position=selected_position,
            strategy_block_size=args.strategy_block_size,
        )
        frozen_side = opponent_of(learning_side)
        print(
            f"  - Round {round_index}: public cross-examination simulation "
            f"({party_label(learning_side)} learns, {party_label(frozen_side)} strategy frozen)"
        )
        emit_event(
            events_path,
            "round_started",
            {
                "round": round_index,
                "learning_side_key": learning_side,
                "learning_side": party_label(learning_side),
                "frozen_side_key": frozen_side,
                "frozen_side": party_label(frozen_side),
                "content": (
                    f"第 {round_index} 轮开始：{party_label(learning_side)}训练，"
                    f"{party_label(frozen_side)}策略冻结。"
                ),
            },
        )
        round_record = simulate_round(
            runner,
            round_index=round_index,
            case_text=case_text,
            strategies=strategies,
            qa_pairs=args.qa_pairs,
            skip_tribunal=args.skip_tribunal,
            knowledge_base=knowledge_base,
            rag_top_k=args.rag_top_k,
            rag_max_context_chars=args.rag_max_context_chars,
            max_history_chars=args.max_history_chars,
            max_retries=args.max_retries,
            events_path=events_path,
        )
        round_record["learning_side"] = party_label(learning_side)
        round_record["learning_side_key"] = learning_side
        round_record["frozen_side"] = party_label(frozen_side)
        round_record["frozen_side_key"] = frozen_side
        round_record["training_note"] = (
            "This is an independent pre-hearing training attempt. Bad answers in this round are lessons, "
            "not irrevocable admissions in a continuous real hearing."
        )
        print(f"  - Round {round_index}: neutral critic review for both parties")
        critic_review = critic_review_round(
            runner,
            round_index=round_index,
            case_text=case_text,
            public_round=round_record,
            knowledge_base=knowledge_base,
            rag_top_k=args.rag_top_k,
            rag_max_context_chars=args.rag_max_context_chars,
            max_history_chars=args.max_history_chars,
            max_retries=args.max_retries,
        )
        round_record["critic_review"] = critic_review
        critic_reviews.append({"round": round_index, "critic_review": critic_review})
        emit_event(
            events_path,
            "critic_review",
            {
                "round": round_index,
                "speaker": "点评 Agent",
                "content": critic_review.get("round_summary", "本轮点评已生成。"),
                "payload": critic_review,
            },
        )
        public_rounds.append(round_record)

        print(f"  - Round {round_index}: private reflection and strategy update for {party_label(learning_side)}")
        reflection = reflect_round(
            runner,
            round_index=round_index,
            position=learning_side,
            case_text=case_text,
            strategy=strategies[learning_side],
            public_round=round_record,
            memory=memory[learning_side],
            knowledge_base=knowledge_base,
            rag_top_k=args.rag_top_k,
            rag_max_context_chars=args.rag_max_context_chars,
            max_history_chars=args.max_history_chars,
            max_retries=args.max_retries,
        )
        emit_event(
            events_path,
            "reflection",
            {
                "round": round_index,
                "speaker": f"{party_label(learning_side)}复盘 Agent",
                "content": "本轮复盘已完成。",
                "payload": reflection,
            },
        )
        reflections[learning_side].append(reflection)
        memory[learning_side].append(
            {
                "round": round_index,
                "new_lessons": reflection.get("new_lessons", []),
                "risk_warnings": reflection.get("risk_warnings", []),
            }
        )
        strategies[learning_side] = update_strategy(
            runner,
            round_index=round_index,
            position=learning_side,
            case_text=case_text,
            previous_strategy=strategies[learning_side],
            reflection=reflection,
            memory=memory[learning_side],
            knowledge_base=knowledge_base,
            rag_top_k=args.rag_top_k,
            rag_max_context_chars=args.rag_max_context_chars,
            max_retries=args.max_retries,
        )
        emit_event(
            events_path,
            "strategy_update",
            {
                "round": round_index,
                "speaker": f"{party_label(learning_side)}策略 Agent",
                "content": strategies[learning_side].get("strategy_summary", "策略已更新。"),
                "payload": strategies[learning_side],
            },
        )
        strategy_versions[learning_side].append(strategies[learning_side])
        strategy_update_rounds[learning_side].append(round_index)
        training_updates.append(
            {
                "round": round_index,
                "learning_side_key": learning_side,
                "learning_side": party_label(learning_side),
                "critic_review": critic_review,
                "reflection": reflection,
                "updated_strategy": strategies[learning_side],
            }
        )

    print(f"[4/4] Writing final advice for {selected_label}.")
    advice = final_advice(
        runner,
        selected_position=selected_position,
        case_id=case_id,
        case_text=case_text,
        initial_strategy=initial_strategies[selected_position],
        strategy_versions=strategy_versions[selected_position],
        strategy_update_rounds=strategy_update_rounds[selected_position],
        reflections=reflections[selected_position],
        public_rounds=public_rounds,
        max_history_chars=args.max_history_chars,
        strategy_block_size=args.strategy_block_size,
        knowledge_base=knowledge_base,
        rag_top_k=args.rag_top_k,
        rag_max_context_chars=args.rag_max_context_chars,
    )
    emit_event(
        events_path,
        "final_advice",
        {
            "speaker": f"{selected_label}建议 Agent",
            "content": "最终庭前建议已生成。",
        },
    )

    case_output_dir.mkdir(parents=True, exist_ok=True)
    advice_path.write_text(advice, encoding="utf-8")
    training_summary = build_training_summary_markdown(
        case_id=case_id,
        selected_position=selected_position,
        rounds=args.rounds,
        qa_pairs=args.qa_pairs,
        strategy_block_size=args.strategy_block_size,
        training_updates=training_updates,
    )
    training_summary_path.write_text(training_summary, encoding="utf-8")
    log_payload = selected_party_log(
        selected_position=selected_position,
        case_id=case_id,
        case_doc=str(case_doc_path),
        rounds=args.rounds,
        qa_pairs=args.qa_pairs,
        public_rounds=public_rounds,
        selected_initial_strategy=initial_strategies[selected_position],
        selected_strategy_versions=strategy_versions[selected_position],
        selected_strategy_update_rounds=strategy_update_rounds[selected_position],
        selected_reflections=reflections[selected_position],
        training_updates=training_updates,
        critic_reviews=critic_reviews,
        strategy_block_size=args.strategy_block_size,
    )
    log_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote advice: {advice_path}")
    print(f"Wrote training summary: {training_summary_path}")
    print(f"Wrote selected-side log: {log_path}")
    emit_event(
        events_path,
        "run_completed",
        {
            "case_output_dir": str(case_output_dir),
            "advice_path": str(advice_path),
            "training_summary_path": str(training_summary_path),
            "log_path": str(log_path),
            "content": "模拟完成，输出文件已写入。",
        },
    )
    if args.dry_run:
        print("Dry run completed without calling an LLM.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:  # noqa: BLE001 - CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
