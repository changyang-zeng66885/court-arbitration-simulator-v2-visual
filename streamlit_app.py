from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
INPUT_CASES_DIR = ROOT / "inputs" / "test_case"
RULES_DIR = ROOT / "ref_rules_doc"
RUNS_DIR = ROOT / "outputs" / "streamlit_runs"
CASE_RAG_DIR = ROOT / "data" / "processed" / "rag_corpus"

POSITION_OPTIONS = {"申请方": "claimant", "被申请方": "respondent"}
POSITION_LABELS = {value: label for label, value in POSITION_OPTIONS.items()}
AUTO_RULE_OPTION = "自动识别"
ALL_RULES_OPTION = "全部规则文档"
RESULT_RAG_SOURCE_LABEL = "本案模拟结果（RAG）"
LAW_RAG_SOURCE_LABEL = "法律条文知识库（RAG）"
CASE_RAG_SOURCE_LABEL = "公开案例数据库（RAG）"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(read_text(path))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def list_case_docs() -> list[Path]:
    if not INPUT_CASES_DIR.exists():
        return []
    return sorted(INPUT_CASES_DIR.glob("case*/main.md"))


def list_rule_docs() -> list[Path]:
    if not RULES_DIR.exists():
        return []
    return sorted(
        path
        for path in RULES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".md", ".txt"}
    )


def case_id_from_path(path: Path) -> str:
    if path.name.lower() == "main.md":
        return path.parent.name
    return path.stem


def compact_filename(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def infer_rule_docs(case_text: str, rule_docs: list[Path]) -> list[Path]:
    normalized_case = case_text.replace(" ", "").replace("\n", "")
    matches: list[Path] = []
    for path in rule_docs:
        stem = path.stem
        candidates = {stem, stem.split("-")[0], stem.replace("-", "")}
        if any(candidate and candidate in normalized_case for candidate in candidates):
            matches.append(path)
    return matches


def make_selected_rules_dir(run_dir: Path, selected_rule_docs: list[Path]) -> Path:
    selected_dir = run_dir / "selected_rules"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for source in selected_rule_docs:
        shutil.copy2(source, selected_dir / source.name)
    return selected_dir


def save_uploaded_case(run_dir: Path, uploaded_file: Any) -> Path:
    case_dir = run_dir / "uploaded_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    case_path = case_dir / "main.md"
    case_path.write_bytes(uploaded_file.getvalue())
    return case_path


def build_command(
    *,
    case_doc: Path,
    output_root: Path,
    position: str,
    rounds: int,
    qa_pairs: int,
    strategy_block_size: int,
    dry_run: bool,
    skip_tribunal: bool,
    disable_rag: bool,
    rules_dir: Path,
    model: str,
    base_url: str,
    temperature: float,
    events_path: Path | None = None,
) -> tuple[list[str], dict[str, str]]:
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "simulate_case.py"),
        "--case-doc",
        str(case_doc),
        "--position",
        position,
        "--rounds",
        str(rounds),
        "--qa-pairs",
        str(qa_pairs),
        "--strategy-block-size",
        str(strategy_block_size),
        "--outputs-dir",
        str(output_root),
        "--temperature",
        str(temperature),
        "--overwrite",
    ]

    if events_path is not None:
        command.extend(["--events-path", str(events_path)])

    if dry_run:
        command.append("--dry-run")
    if skip_tribunal:
        command.append("--skip-tribunal")
    if disable_rag:
        command.append("--disable-rag")
    else:
        command.extend(["--rules-dir", str(rules_dir), "--case-rag-dir", str(CASE_RAG_DIR)])

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if model.strip():
        env["OPENAI_MODEL"] = model.strip()
    if base_url.strip():
        env["OPENAI_BASE_URL"] = base_url.strip()
    return command, env


def load_event_file(events_path: Path) -> list[dict[str, Any]]:
    if not events_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in read_text(events_path).splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def enqueue_output(stream: Any, output_queue: queue.Queue[str]) -> None:
    for line in stream:
        output_queue.put(line.rstrip())


def run_simulation(command: list[str], env: dict[str, str], events_path: Path | None = None) -> tuple[int, list[str]]:
    log_lines: list[str] = []
    st.markdown("### 实时模拟")
    live_chat_box = st.empty()
    log_box = st.empty()
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None
    output_queue: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=enqueue_output, args=(process.stdout, output_queue), daemon=True)
    reader.start()

    last_event_count = -1
    while process.poll() is None:
        while True:
            try:
                log_lines.append(output_queue.get_nowait())
            except queue.Empty:
                break
        if events_path is not None:
            events = load_event_file(events_path)
            if len(events) != last_event_count:
                with live_chat_box.container():
                    render_live_events(events)
                last_event_count = len(events)
        log_box.code("\n".join(log_lines[-80:]) or "等待模拟脚本输出...", language="text")
        time.sleep(0.35)

    return_code = process.wait()
    reader.join(timeout=1)
    while True:
        try:
            log_lines.append(output_queue.get_nowait())
        except queue.Empty:
            break
    if events_path is not None:
        with live_chat_box.container():
            render_live_events(load_event_file(events_path))
    log_box.code("\n".join(log_lines[-120:]), language="text")
    return return_code, log_lines


def load_result(case_doc: Path, output_root: Path, position: str) -> dict[str, Any]:
    case_id = case_id_from_path(case_doc)
    selected_label = POSITION_LABELS[position]
    case_output_dir = output_root / case_id
    log_path = case_output_dir / f"{selected_label}模拟记录.json"
    return load_result_from_log(log_path, output_root.parent)


def load_result_from_log(log_path: Path, run_dir: Path | None = None) -> dict[str, Any]:
    payload = json.loads(read_text(log_path))
    case_output_dir = log_path.parent
    selected_party = payload.get("selected_party", "申请方")
    case_id = payload.get("case_id", case_output_dir.name)
    advice_path = case_output_dir / f"{selected_party}庭前建议.md"
    summary_path = case_output_dir / "训练过程总结.md"
    if run_dir is None:
        run_dir = case_output_dir.parents[1] if len(case_output_dir.parents) > 1 else case_output_dir
    metadata_path = run_dir / "run_metadata.json"
    events_path = run_dir / "events.jsonl"
    return {
        "case_id": case_id,
        "case_output_dir": case_output_dir,
        "advice_path": advice_path,
        "summary_path": summary_path,
        "log_path": log_path,
        "run_dir": run_dir,
        "metadata_path": metadata_path,
        "events_path": events_path,
        "metadata": read_json_file(metadata_path) if metadata_path.exists() else {},
        "advice": read_text(advice_path) if advice_path.exists() else "",
        "summary": read_text(summary_path) if summary_path.exists() else "",
        "payload": payload,
    }


def parse_run_datetime(run_id: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(run_id, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def result_log_files(run_dir: Path) -> list[Path]:
    results_dir = run_dir / "results"
    if not results_dir.exists():
        return []
    return sorted(results_dir.glob("*/*模拟记录.json"))


def list_history_entries() -> list[dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []

    entries: list[dict[str, Any]] = []
    for run_dir in sorted((path for path in RUNS_DIR.iterdir() if path.is_dir()), key=lambda path: path.name, reverse=True):
        metadata_path = run_dir / "run_metadata.json"
        metadata = read_json_file(metadata_path) if metadata_path.exists() else {}
        logs = result_log_files(run_dir)
        if logs:
            for log_path in logs:
                payload = read_json_file(log_path)
                run_time = parse_run_datetime(run_dir.name)
                case_id = payload.get("case_id", log_path.parent.name)
                selected_party = payload.get("selected_party", "-")
                mode = "离线" if metadata.get("dry_run") else "模型"
                rounds = payload.get("rounds", "-")
                qa_pairs = payload.get("qa_pairs_per_segment", "-")
                created = run_time.strftime("%Y-%m-%d %H:%M:%S") if run_time else run_dir.name
                entries.append(
                    {
                        "run_id": run_dir.name,
                        "run_dir": run_dir,
                        "log_path": log_path,
                        "metadata": metadata,
                        "status": "完成",
                        "label": f"{created} · {case_id} · {selected_party} · {rounds}轮/{qa_pairs}问答 · {mode}",
                        "sort_time": log_path.stat().st_mtime,
                    }
                )
        else:
            run_time = parse_run_datetime(run_dir.name)
            created = run_time.strftime("%Y-%m-%d %H:%M:%S") if run_time else run_dir.name
            entries.append(
                {
                    "run_id": run_dir.name,
                    "run_dir": run_dir,
                    "log_path": None,
                    "metadata": metadata,
                    "status": "未完成",
                    "label": f"{created} · 未完成/无结果",
                    "sort_time": run_dir.stat().st_mtime,
                }
            )

    entries.sort(key=lambda item: item["sort_time"], reverse=True)
    return entries


def load_history_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    log_path = entry.get("log_path")
    if not isinstance(log_path, Path) or not log_path.exists():
        return None
    return load_result_from_log(log_path, entry.get("run_dir"))


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value)
    if isinstance(value, dict):
        return "\n".join(f"{key}：{as_text(item)}" for key, item in value.items())
    return str(value)


# ---------------------------------------------------------------------------
# RAG citation helpers
# ---------------------------------------------------------------------------

_RAG_CITATION_PATTERN = re.compile(
    r">\s*\[([^\]]+),\s*((?:case|law|result):[^\]]+)\]\s*原文：(.+)",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# PDF / DOCX text extraction (used by load_rag_source for law documents)
# ---------------------------------------------------------------------------

def _extract_docx_text(path: Path) -> str:
    import zipfile
    from xml.etree import ElementTree as ET

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


def _extract_pdf_text(path: Path) -> str:
    try:
        import fitz
    except ImportError:
        return "[无法解析PDF：缺少 pymupdf 依赖。请运行 pip install pymupdf]"

    with fitz.open(path) as document:
        return "\n".join(page.get_text("text") for page in document)


def _read_knowledge_text(path: Path) -> str:
    """Read a knowledge document, handling PDF/DOCX/MD/TXT."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(payload, ensure_ascii=False, indent=2)
    if suffix == ".docx":
        return _extract_docx_text(path)
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _split_text_into_chunks(text: str, *, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Re-split document text using the same algorithm as the RAG builder.

    Must match ``split_text_into_chunks`` in ``scripts/simulate_case.py`` so that
    a chunk_id like ``law:xxx:0015`` maps to the exact same text block.
    """
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
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


def parse_rag_citations(text: str) -> list[dict[str, str]]:
    """Extract RAG citations from text content.

    Each citation has the format:
        > [来源, chunk_id] 原文：excerpt text

    Returns a list of dicts with keys: full, source_label, chunk_id, excerpt.
    """
    citations: list[dict[str, str]] = []
    for match in _RAG_CITATION_PATTERN.finditer(text):
        citations.append(
            {
                "full": match.group(0),
                "source_label": match.group(1).strip(),
                "chunk_id": match.group(2).strip(),
                "excerpt": match.group(3).strip(),
            }
        )
    return citations


def load_rag_source(chunk_id: str) -> str | None:
    """Load the specific chunk (with context) referenced by *chunk_id*.

    chunk_id format:
        case:case005:0011  →  data/processed/rag_corpus/case005/text.md
        law:doc_name:0015  →  ref_rules_doc/<matching file>

    The document is re-split with the same parameters the RAG builder used
    (max_chars=1200, overlap=150), so chunk 0015 maps to the identical text
    block the LLM saw.  Surrounding chunks are included for context.
    """
    parts = chunk_id.split(":")
    if len(parts) < 3:
        return None
    id_prefix, stem, chunk_num_str = parts[0], parts[1], parts[2]

    if id_prefix == "result":
        result_sources = st.session_state.get("qa_result_chunk_sources", {})
        if isinstance(result_sources, dict):
            source_text = result_sources.get(chunk_id)
            if isinstance(source_text, str):
                return source_text
        return None

    source_path: Path | None = None

    if id_prefix == "case":
        candidate = CASE_RAG_DIR / stem / "text.md"
        if candidate.exists():
            source_path = candidate
    elif id_prefix == "law":
        if RULES_DIR.exists():
            for rule_path in RULES_DIR.iterdir():
                if not rule_path.is_file():
                    continue
                rule_stem = rule_path.stem
                if rule_stem == stem or rule_stem.replace("-", "").replace("_", "") == stem.replace("-", "").replace("_", ""):
                    source_path = rule_path
                    break
            if source_path is None:
                for rule_path in RULES_DIR.iterdir():
                    if rule_path.is_file() and stem[:6] in rule_path.stem:
                        source_path = rule_path
                        break

    if source_path is None or not source_path.exists():
        return None

    try:
        chunk_num = int(chunk_num_str)
    except ValueError:
        return None

    try:
        full_text = _read_knowledge_text(source_path)
        chunks = _split_text_into_chunks(full_text)
        total = len(chunks)

        if chunk_num < 1 or chunk_num > total:
            return (
                f"⚠️ chunk_id 超出范围：文档共有 {total} 个文本块，"
                f"请求的是第 {chunk_num} 个。\n\n"
                f"文档路径：{source_path}\n\n"
                f"--- 文档开头预览 ---\n{full_text[:3000]}"
            )

        # Show the specific chunk with one chunk of context on each side
        parts_out: list[str] = []
        parts_out.append(f"📄 源文件：{source_path.name}（共 {total} 个文本块，当前为第 {chunk_num} 块）\n")

        if chunk_num > 1:
            parts_out.append(f"━━━ 上一块 (chunk {chunk_num - 1}) ━━━")
            parts_out.append(chunks[chunk_num - 2])
            parts_out.append("")

        parts_out.append(f"━━━ ★ 引用的文本块 (chunk {chunk_num}) ★ ━━━")
        parts_out.append(chunks[chunk_num - 1])
        parts_out.append("")

        if chunk_num < total:
            parts_out.append(f"━━━ 下一块 (chunk {chunk_num + 1}) ━━━")
            parts_out.append(chunks[chunk_num])

        return "\n".join(parts_out)
    except Exception:
        return None


def render_rag_card_html(source_label: str, chunk_id: str, excerpt: str) -> str:
    """Build an HTML card for a single RAG citation."""
    if "案例" in source_label:
        card_class = "rag-card-case"
        icon = "📋"
    elif "法律" in source_label:
        card_class = "rag-card-law"
        icon = "📜"
    elif "本案" in source_label or chunk_id.startswith("result:"):
        card_class = "rag-card-result"
        icon = "案"
    else:
        card_class = "rag-card-default"
        icon = "📎"

    detail_html = ""
    source_text = load_rag_source(chunk_id)
    if source_text:
        escaped_source = html.escape(source_text)
        detail_html = (
            "<details class='rag-detail'>"
            "<summary>📂 查看原始文档内容</summary>"
            f"<div class='rag-source-content'>{escaped_source}</div>"
            "</details>"
        )

    return (
        f"<div class='rag-card {card_class}'>"
        f"<div class='rag-card-header'>"
        f"<span class='rag-card-icon'>{icon}</span>"
        f"<span class='rag-card-source'>{html.escape(source_label)}</span>"
        f"<span class='rag-card-id'>{html.escape(chunk_id)}</span>"
        f"</div>"
        f"<div class='rag-card-body'>"
        f"<div class='rag-card-excerpt'>{html.escape(excerpt)}</div>"
        f"</div>"
        f"{detail_html}"
        f"</div>"
    )


def render_content_with_rag_cards(content: str) -> str:
    """Render text content, converting RAG citations into HTML cards."""
    if not content:
        return ""

    matches = list(_RAG_CITATION_PATTERN.finditer(content))
    if not matches:
        return html.escape(content).replace("\n", "<br>")

    parts: list[str] = []
    cursor = 0
    for match in matches:
        # Text before this citation
        before = content[cursor : match.start()]
        if before:
            parts.append(html.escape(before).replace("\n", "<br>"))

        # RAG citation card
        source_label = match.group(1).strip()
        chunk_id = match.group(2).strip()
        excerpt = match.group(3).strip()
        parts.append(render_rag_card_html(source_label, chunk_id, excerpt))

        cursor = match.end()

    # Remaining text after the last citation
    tail = content[cursor:]
    if tail:
        parts.append(html.escape(tail).replace("\n", "<br>"))

    return "".join(parts)


def render_markdown_with_rag_cards(md_text: str) -> str:
    """Process markdown text, replacing RAG citation blockquotes with HTML cards.

    Unlike ``render_content_with_rag_cards`` which escapes everything outside
    citations, this function keeps the surrounding markdown intact so it can be
    fed to ``st.markdown(..., unsafe_allow_html=True)``.
    """
    if not md_text:
        return ""

    # Multi-line aware: capture the citation header and any continuation lines
    # that are part of the same blockquote paragraph.
    pattern = re.compile(
        r"(?:^|\n)> *\[([^\]]+), *((?:case|law|result):[^\]]+)\] *原文：(.+?)(?=\n(?:[^>]|\n|$)|$)",
        re.MULTILINE,
    )

    matches = list(pattern.finditer(md_text))
    if not matches:
        return md_text

    parts: list[str] = []
    cursor = 0
    for match in matches:
        before = md_text[cursor : match.start()]
        parts.append(before)

        source_label = match.group(1).strip()
        chunk_id = match.group(2).strip()
        excerpt = match.group(3).strip()

        # Also consume any immediately following blockquote continuation lines
        end = match.end()
        remaining = md_text[end:]
        continuation_pattern = re.compile(r"^\n(> .*)", re.MULTILINE)
        while True:
            cont_match = continuation_pattern.match(remaining)
            if not cont_match:
                break
            cont_line = cont_match.group(1)
            # Check that this continuation line is NOT a new citation
            if re.match(r"> *\[[^\]]+, *(?:case|law|result):[^\]]+\] *原文：", cont_line):
                break
            excerpt += "\n" + cont_line[2:].strip()
            end += len(cont_match.group(0))
            remaining = md_text[end:]

        parts.append(render_rag_card_html(source_label, chunk_id, excerpt))
        cursor = end

    parts.append(md_text[cursor:])
    return "".join(parts)


def qa_result_widget_key(result: dict[str, Any], suffix: str) -> str:
    raw_key = str(result.get("log_path") or result.get("case_output_dir") or result.get("case_id") or "result")
    digest = hashlib.sha1(raw_key.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"qa-{suffix}-{digest}"


def qa_int_to_chinese_number(value: int) -> str:
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


def expand_qa_query(text: str) -> str:
    additions: list[str] = []
    for match in re.finditer(r"第\s*(\d{1,4})\s*条", text):
        additions.append(f"第{qa_int_to_chinese_number(int(match.group(1)))}条")
    if "通谋虚伪表示" in text or "虚伪表示" in text:
        additions.append("虚假的意思表示 民事法律行为无效 意思表示隐藏")
    if any(keyword in text for keyword in ("怎么回答", "如何回答", "应当回答", "对方问", "盘问")):
        additions.append("证人回答 应答策略 安全回答 盘问目的 风险 防守口径")
    if any(keyword in text for keyword in ("总结", "概括", "结果", "重点", "建议", "复盘")):
        additions.append("训练过程总结 最终庭前建议 迭代记录 风险点 行动清单")
    return f"{text}\n{' '.join(additions)}" if additions else text


def tokenize_qa_text(text: str) -> list[str]:
    expanded = expand_qa_query(text).lower()
    terms: list[str] = []
    terms.extend(re.findall(r"[a-z0-9_]{2,}", expanded))
    terms.extend(re.findall(r"\d+(?:\.\d+)?", expanded))
    for segment in re.findall(r"[\u4e00-\u9fff]{2,}", expanded):
        terms.extend(segment[index : index + 2] for index in range(len(segment) - 1))
        terms.extend(segment[index : index + 3] for index in range(len(segment) - 2))
    return terms


def normalize_context_text(text: str, *, limit: int | None = None) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if limit is not None and len(normalized) > limit:
        return normalized[:limit].rstrip() + "..."
    return normalized


def qa_doc_id(value: str) -> str:
    cleaned = str(value).replace(":", "_").replace("]", "_").strip()
    return cleaned or "doc"


def make_qa_chunk_record(
    *,
    chunk_id: str,
    source_label: str,
    source_title: str,
    source_path: str,
    text: str,
) -> dict[str, Any]:
    searchable_text = f"{source_label}\n{source_title}\n{text}"
    return {
        "chunk_id": chunk_id,
        "source_label": source_label,
        "source_title": source_title,
        "source_path": source_path,
        "text": text.strip(),
        "terms": dict(Counter(tokenize_qa_text(searchable_text))),
    }


def append_qa_document_chunks(
    records: list[dict[str, Any]],
    *,
    id_prefix: str,
    doc_id: str,
    source_label: str,
    source_title: str,
    source_path: str,
    text: str,
    max_chars: int = 1200,
    overlap: int = 150,
) -> None:
    if not text.strip():
        return
    safe_doc_id = qa_doc_id(doc_id)
    for index, chunk_text in enumerate(_split_text_into_chunks(text, max_chars=max_chars, overlap=overlap), start=1):
        if len(re.sub(r"\s+", "", chunk_text)) < 20:
            continue
        records.append(
            make_qa_chunk_record(
                chunk_id=f"{id_prefix}:{safe_doc_id}:{index:04d}",
                source_label=source_label,
                source_title=source_title,
                source_path=source_path,
                text=chunk_text,
            )
        )


def json_text_for_qa(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return as_text(value)


def format_public_rounds_for_qa(payload: dict[str, Any]) -> str:
    sections: list[str] = []
    for round_record in payload.get("public_rounds", []):
        round_no = round_record.get("round", "-")
        sections.append(
            f"第 {round_no} 轮：{round_record.get('learning_side', '')}训练，"
            f"{round_record.get('frozen_side', '')}策略冻结。"
        )
        for segment in round_record.get("segments", []):
            sections.append(
                f"第 {round_no} 轮环节 {segment.get('segment', '-')}: "
                f"{segment.get('questioner', '')}盘问{segment.get('witness', '')}"
            )
            for turn in segment.get("turns", []):
                question = turn.get("question", {})
                answer = turn.get("answer", {})
                sections.append(
                    "\n".join(
                        part
                        for part in (
                            f"问答 {turn.get('turn', '-')}",
                            f"律师问题：{as_text(question.get('question'))}",
                            f"盘问目的：{as_text(question.get('purpose'))}",
                            f"预期压力：{as_text(question.get('expected_pressure'))}",
                            f"证人回答：{as_text(answer.get('answer'))}",
                            f"防守动作：{as_text(answer.get('defense_move'))}",
                            f"让步：{as_text(answer.get('concessions'))}",
                            f"暴露风险：{as_text(answer.get('risks_created'))}",
                        )
                        if part.strip()
                    )
                )
            tribunal_review = segment.get("tribunal_review")
            if isinstance(tribunal_review, dict):
                sections.append(f"仲裁庭点评：\n{as_text(tribunal_review)}")
        closing_statements = round_record.get("closing_statements", [])
        if closing_statements:
            sections.append(f"最后陈述：\n{as_text(closing_statements)}")
        critic_review = round_record.get("critic_review")
        if isinstance(critic_review, dict):
            sections.append(f"点评 Agent 总评：\n{as_text(critic_review)}")
    return "\n\n".join(sections)


def format_training_updates_for_qa(payload: dict[str, Any]) -> str:
    sections: list[str] = []
    for update in payload.get("training_updates", []):
        sections.append(
            "\n".join(
                part
                for part in (
                    f"第 {update.get('round', '-')} 轮训练方：{update.get('learning_side', '')}",
                    f"点评 Agent：\n{as_text(update.get('critic_review'))}",
                    f"训练方复盘：\n{as_text(update.get('reflection'))}",
                    f"更新后策略：\n{as_text(update.get('updated_strategy'))}",
                )
                if part.strip()
            )
        )
    return "\n\n".join(sections)


def format_selected_side_materials_for_qa(payload: dict[str, Any]) -> str:
    sections: list[str] = []
    if payload.get("selected_initial_strategy"):
        sections.append(f"初始策略：\n{json_text_for_qa(payload.get('selected_initial_strategy'))}")
    if payload.get("selected_strategy_versions"):
        sections.append(f"策略版本：\n{json_text_for_qa(payload.get('selected_strategy_versions'))}")
    if payload.get("selected_reflections"):
        sections.append(f"选定阵营复盘：\n{json_text_for_qa(payload.get('selected_reflections'))}")
    return "\n\n".join(sections)


def build_result_qa_chunk_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    payload = result.get("payload", {})
    metadata = result.get("metadata", {})
    result_path = str(result.get("log_path") or result.get("case_output_dir") or result.get("case_id") or "")
    records: list[dict[str, Any]] = []

    overview = "\n".join(
        part
        for part in (
            f"案件：{result.get('case_id', payload.get('case_id', '-'))}",
            f"训练阵营：{payload.get('selected_party', '-')}",
            f"轮数：{payload.get('rounds', '-')}",
            f"每环节问答组数：{payload.get('qa_pairs_per_segment', '-')}",
            f"策略交替块大小：{payload.get('strategy_block_size', '-')}",
            f"案件文件：{payload.get('case_doc') or metadata.get('case_doc') or '-'}",
            f"规则文档：{as_text(metadata.get('effective_rules'))}",
        )
        if part.strip()
    )
    append_qa_document_chunks(
        records,
        id_prefix="result",
        doc_id="overview",
        source_label=RESULT_RAG_SOURCE_LABEL,
        source_title="结果概览",
        source_path=result_path,
        text=overview,
    )

    case_doc_value = payload.get("case_doc") or metadata.get("case_doc")
    if isinstance(case_doc_value, str) and case_doc_value.strip():
        case_doc_path = Path(case_doc_value)
        try:
            if case_doc_path.exists():
                append_qa_document_chunks(
                    records,
                    id_prefix="result",
                    doc_id="case_file",
                    source_label=RESULT_RAG_SOURCE_LABEL,
                    source_title="原始案件材料",
                    source_path=compact_filename(case_doc_path),
                    text=read_text(case_doc_path),
                )
        except OSError:
            pass

    if result.get("advice"):
        append_qa_document_chunks(
            records,
            id_prefix="result",
            doc_id="advice",
            source_label=RESULT_RAG_SOURCE_LABEL,
            source_title="最终庭前建议",
            source_path=compact_filename(result["advice_path"]),
            text=result["advice"],
        )
    if result.get("summary"):
        append_qa_document_chunks(
            records,
            id_prefix="result",
            doc_id="summary",
            source_label=RESULT_RAG_SOURCE_LABEL,
            source_title="训练过程总结",
            source_path=compact_filename(result["summary_path"]),
            text=result["summary"],
        )

    transcript = format_public_rounds_for_qa(payload)
    append_qa_document_chunks(
        records,
        id_prefix="result",
        doc_id="public_rounds",
        source_label=RESULT_RAG_SOURCE_LABEL,
        source_title="模拟问答记录",
        source_path=result_path,
        text=transcript,
    )

    training_updates = format_training_updates_for_qa(payload)
    append_qa_document_chunks(
        records,
        id_prefix="result",
        doc_id="training_updates",
        source_label=RESULT_RAG_SOURCE_LABEL,
        source_title="训练复盘与策略迭代",
        source_path=result_path,
        text=training_updates,
    )

    selected_side_materials = format_selected_side_materials_for_qa(payload)
    append_qa_document_chunks(
        records,
        id_prefix="result",
        doc_id="selected_side_materials",
        source_label=RESULT_RAG_SOURCE_LABEL,
        source_title="选定阵营私有策略材料",
        source_path=result_path,
        text=selected_side_materials,
    )
    return records


def update_result_chunk_sources(result_chunks: list[dict[str, Any]]) -> None:
    st.session_state["qa_result_chunk_sources"] = {
        chunk["chunk_id"]: (
            f"来源：{chunk.get('source_title', '')}\n"
            f"路径：{chunk.get('source_path', '')}\n\n"
            f"{chunk.get('text', '')}"
        )
        for chunk in result_chunks
    }


def selected_rule_paths_for_result(result: dict[str, Any]) -> list[Path]:
    rule_docs = list_rule_docs()
    metadata = result.get("metadata", {})
    effective_rules = metadata.get("effective_rules")
    if isinstance(effective_rules, list) and effective_rules:
        selected_names = {str(name) for name in effective_rules}
        selected = [path for path in rule_docs if path.name in selected_names]
        if selected:
            selected_set = {path.name for path in selected}
            return selected + [path for path in rule_docs if path.name not in selected_set]
    return rule_docs


def external_case_paths() -> list[Path]:
    if not CASE_RAG_DIR.exists():
        return []
    return sorted(path for path in CASE_RAG_DIR.glob("case*/text.md") if path.is_file())


def file_signatures(paths: list[Path]) -> tuple[tuple[str, float, int], ...]:
    signatures: list[tuple[str, float, int]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        signatures.append((str(path), stat.st_mtime, stat.st_size))
    return tuple(signatures)


@st.cache_data(show_spinner=False)
def build_external_qa_chunk_records_cached(
    rule_path_values: tuple[str, ...],
    case_path_values: tuple[str, ...],
    signatures: tuple[tuple[str, float, int], ...],
) -> list[dict[str, Any]]:
    del signatures
    records: list[dict[str, Any]] = []
    for path_value in rule_path_values:
        path = Path(path_value)
        try:
            text = _read_knowledge_text(path)
        except Exception:
            continue
        append_qa_document_chunks(
            records,
            id_prefix="law",
            doc_id=path.stem,
            source_label=LAW_RAG_SOURCE_LABEL,
            source_title=path.stem,
            source_path=compact_filename(path),
            text=text,
        )
    for path_value in case_path_values:
        path = Path(path_value)
        try:
            text = _read_knowledge_text(path)
        except Exception:
            continue
        case_id = path.parent.name if path.name == "text.md" else path.stem
        append_qa_document_chunks(
            records,
            id_prefix="case",
            doc_id=case_id,
            source_label=CASE_RAG_SOURCE_LABEL,
            source_title=f"公开案例 {case_id}",
            source_path=compact_filename(path),
            text=text,
        )
    return records


def build_external_qa_chunk_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    rule_paths = selected_rule_paths_for_result(result)
    case_paths = external_case_paths()
    paths = rule_paths + case_paths
    return build_external_qa_chunk_records_cached(
        tuple(str(path) for path in rule_paths),
        tuple(str(path) for path in case_paths),
        file_signatures(paths),
    )


def search_qa_chunks(chunks: list[dict[str, Any]], query: str, *, top_k: int) -> list[tuple[float, dict[str, Any]]]:
    if not chunks:
        return []
    query_terms = Counter(tokenize_qa_text(query))
    if not query_terms:
        return []

    doc_freq: Counter[str] = Counter()
    for chunk in chunks:
        terms = chunk.get("terms", {})
        if isinstance(terms, dict):
            doc_freq.update(terms.keys())

    total = max(len(chunks), 1)
    idf = {term: math.log((1 + total) / (1 + freq)) + 1 for term, freq in doc_freq.items()}
    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk in chunks:
        terms = chunk.get("terms", {})
        if not isinstance(terms, dict):
            continue
        score = 0.0
        for term, query_tf in query_terms.items():
            chunk_tf = terms.get(term, 0)
            if chunk_tf:
                score += (1 + math.log(chunk_tf)) * (1 + math.log(query_tf)) * idf.get(term, 1.0)
        if score:
            scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def is_summary_question(question: str) -> bool:
    return any(keyword in question for keyword in ("总结", "概括", "结果", "重点", "建议", "复盘"))


def with_priority_result_chunks(
    hits: list[tuple[float, dict[str, Any]]],
    chunks: list[dict[str, Any]],
    question: str,
) -> list[tuple[float, dict[str, Any]]]:
    if not is_summary_question(question):
        return hits
    priority_titles = ("结果概览", "最终庭前建议", "训练过程总结")
    seen = {chunk["chunk_id"] for _, chunk in hits}
    priority_hits: list[tuple[float, dict[str, Any]]] = []
    for title in priority_titles:
        for chunk in chunks:
            if chunk.get("source_title") == title and chunk.get("chunk_id") not in seen:
                priority_hits.append((math.inf, chunk))
                seen.add(chunk["chunk_id"])
                break
    return priority_hits + hits


def format_qa_context(hits: list[tuple[float, dict[str, Any]]], *, max_chars: int) -> str:
    if not hits:
        return "（未检索到高相关片段）"
    blocks: list[str] = []
    used_chars = 0
    for score, chunk in hits:
        available = max_chars - used_chars
        if available <= 0:
            break
        excerpt = normalize_context_text(str(chunk.get("text", "")), limit=min(1200, max(300, available)))
        score_text = "priority" if math.isinf(score) else f"{score:.2f}"
        block = (
            f"- [{chunk.get('source_label')}, {chunk.get('chunk_id')}] score={score_text} "
            f"source={chunk.get('source_title')} ({chunk.get('source_path')})\n"
            f"  原文：{excerpt}"
        )
        used_chars += len(block)
        blocks.append(block)
    return "\n".join(blocks)


def qa_source_entries(
    result_hits: list[tuple[float, dict[str, Any]]],
    external_hits: list[tuple[float, dict[str, Any]]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for score, chunk in result_hits + external_hits:
        entries.append(
            {
                "score": None if math.isinf(score) else round(score, 2),
                "source_label": chunk.get("source_label", ""),
                "source_title": chunk.get("source_title", ""),
                "source_path": chunk.get("source_path", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "excerpt": normalize_context_text(str(chunk.get("text", "")), limit=500),
            }
        )
    return entries


def build_qa_context_for_question(
    *,
    question: str,
    result: dict[str, Any],
    result_chunks: list[dict[str, Any]],
    include_external: bool,
    top_k: int,
) -> tuple[str, str, list[dict[str, Any]]]:
    result_hits = search_qa_chunks(result_chunks, question, top_k=top_k)
    result_hits = with_priority_result_chunks(result_hits, result_chunks, question)

    external_hits: list[tuple[float, dict[str, Any]]] = []
    if include_external:
        external_chunks = build_external_qa_chunk_records(result)
        external_hits = search_qa_chunks(external_chunks, question, top_k=top_k)

    result_context = format_qa_context(result_hits, max_chars=7500)
    external_context = format_qa_context(external_hits, max_chars=5500) if include_external else "（未启用相关案例/法条检索）"
    return result_context, external_context, qa_source_entries(result_hits, external_hits)


def format_qa_history(messages: list[dict[str, Any]], *, limit: int = 4000) -> str:
    if not messages:
        return "（无）"
    lines: list[str] = []
    for message in messages[-6:]:
        role = "用户" if message.get("role") == "user" else "Agent"
        content = normalize_context_text(str(message.get("content", "")), limit=900)
        lines.append(f"{role}：{content}")
    history = "\n".join(lines)
    return history if len(history) <= limit else history[-limit:]


def offline_qa_response(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "未检测到 `OPENAI_API_KEY`，且本次没有检索到可用片段。请先设置模型密钥，或换一个更具体的问题。"
    lines = [
        "未检测到 `OPENAI_API_KEY`，暂时只展示本次检索到的高相关片段。设置密钥后，Agent 会基于这些片段生成完整回答。",
    ]
    for source in sources[:6]:
        excerpt = source["excerpt"]
        lines.append(
            "\n".join(
                (
                    f"### {source['source_title']}",
                    excerpt,
                    f"> [{source['source_label']}, {source['chunk_id']}] 原文：{excerpt}",
                )
            )
        )
    return "\n\n".join(lines)


def answer_qa_question(
    *,
    question: str,
    result: dict[str, Any],
    result_chunks: list[dict[str, Any]],
    include_external: bool,
    top_k: int,
    model: str,
    base_url: str,
    temperature: float,
    history: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    result_context, external_context, sources = build_qa_context_for_question(
        question=question,
        result=result,
        result_chunks=result_chunks,
        include_external=include_external,
        top_k=top_k,
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return offline_qa_response(sources), sources

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai 依赖，请先安装 requirements.txt。") from exc

    client_kwargs: dict[str, str] = {"api_key": api_key}
    if base_url.strip():
        client_kwargs["base_url"] = base_url.strip()
    client = OpenAI(**client_kwargs)

    payload = result.get("payload", {})
    system_prompt = (
        "你是庭前仲裁模拟结果的智能问答 Agent。你必须优先依据本案模拟结果回答，"
        "必要时结合相关案例和法条 RAG 片段。不要编造材料中不存在的事实、证据或法条。"
        "如果材料不足，要明确说明还缺什么。回答应当直接、可执行，尤其是用户询问如何回答对方盘问时，"
        "要给出可直接使用的中文回答口径、应避免承认的内容、可反问或转回的证据点。"
        "如果使用了检索片段，必须在相关段落后用 Markdown 引用格式标明："
        "> [来源, chunk_id] 原文：对应原文片段。chunk_id 必须完全来自检索片段，不得改写。"
    )
    user_prompt = f"""用户问题：
{question}

当前案件信息：
- 案件：{result.get('case_id', payload.get('case_id', '-'))}
- 训练阵营：{payload.get('selected_party', '-')}
- 模拟轮数：{payload.get('rounds', '-')}

此前问答历史：
{format_qa_history(history)}

本案模拟结果检索片段：
{result_context}

相关案例/法条检索片段：
{external_context}

请基于上述材料回答。若用户问题是“怎么回答/如何应对”，请按“建议回答口径 / 回答理由 / 风险提醒”组织。若用户问题是总结类，请按要点归纳。"""

    completion = client.chat.completions.create(
        model=model.strip() or os.getenv("OPENAI_MODEL", "qwen3.6-flash"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("模型返回了空内容。")
    return content.strip(), sources


def render_qa_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.caption("本次没有检索到片段。")
        return
    for source in sources:
        score = "优先片段" if source.get("score") is None else f"score {source['score']}"
        st.markdown(f"**{source['source_label']} · {source['source_title']}**  `{source['chunk_id']}`")
        st.caption(f"{score} · {source['source_path']}")
        st.write(source["excerpt"])


def render_result_qa_agent(
    result: dict[str, Any],
    *,
    model: str,
    base_url: str,
    temperature: float,
) -> None:
    result_chunks = build_result_qa_chunk_records(result)
    update_result_chunk_sources(result_chunks)

    messages_key = qa_result_widget_key(result, "messages")
    include_external_key = qa_result_widget_key(result, "include-external")
    top_k_key = qa_result_widget_key(result, "top-k")
    show_sources_key = qa_result_widget_key(result, "show-sources")

    if messages_key not in st.session_state:
        st.session_state[messages_key] = []

    controls_col1, controls_col2, controls_col3 = st.columns([1.2, 1, 1])
    with controls_col1:
        include_external = st.toggle("检索相关案例/法条", value=True, key=include_external_key)
    with controls_col2:
        top_k = st.slider("每类检索片段", min_value=2, max_value=8, value=5, step=1, key=top_k_key)
    with controls_col3:
        show_sources = st.toggle("显示检索片段", value=False, key=show_sources_key)

    if not os.getenv("OPENAI_API_KEY"):
        st.caption("当前未检测到 OPENAI_API_KEY，Agent 会先以离线方式展示检索片段。")

    if st.button("清空问答", key=qa_result_widget_key(result, "clear")):
        st.session_state[messages_key] = []
        st.rerun()

    messages = st.session_state[messages_key]
    if not messages:
        st.caption("可以询问：如果对方追问某个事实应如何回答，或让 Agent 总结本次模拟结果。")

    for message in messages:
        with st.chat_message(message.get("role", "assistant")):
            if message.get("role") == "assistant":
                processed = render_markdown_with_rag_cards(str(message.get("content", "")))
                st.markdown(processed, unsafe_allow_html=True)
                if show_sources:
                    with st.expander("本次检索片段"):
                        render_qa_sources(message.get("sources", []))
            else:
                st.markdown(str(message.get("content", "")))

    prompt = st.chat_input(
        "询问本案结果，例如：如果对方问我某个事实，我应该怎么回答？",
        key=qa_result_widget_key(result, "input"),
    )
    if not prompt:
        return

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Agent 正在检索并生成回答..."):
            try:
                answer, sources = answer_qa_question(
                    question=prompt,
                    result=result,
                    result_chunks=result_chunks,
                    include_external=include_external,
                    top_k=int(top_k),
                    model=model,
                    base_url=base_url,
                    temperature=temperature,
                    history=messages[:-1],
                )
            except Exception as exc:
                answer = f"问答 Agent 调用失败：{exc}"
                sources = []
        processed = render_markdown_with_rag_cards(answer)
        st.markdown(processed, unsafe_allow_html=True)
        if show_sources:
            with st.expander("本次检索片段"):
                render_qa_sources(sources)
    messages.append({"role": "assistant", "content": answer, "sources": sources})


def render_bubble(speaker: str, content: str, meta: str = "", tone: str = "neutral") -> None:
    initials = speaker[:2] if speaker else "AI"
    escaped_speaker = html.escape(speaker)
    escaped_content = render_content_with_rag_cards(content)
    escaped_meta = render_content_with_rag_cards(meta) if meta else ""
    meta_html = f'<div class="bubble-meta">{escaped_meta}</div>' if escaped_meta else ""
    st.markdown(
        f"""
        <div class="chat-row {tone}">
          <div class="avatar">{html.escape(initials)}</div>
          <div class="bubble">
            <div class="speaker">{escaped_speaker}</div>
            <div class="bubble-content">{escaped_content}</div>
            {meta_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tribunal_review(review: dict[str, Any]) -> None:
    lines: list[str] = []
    for key, label in (
        ("procedural_comments", "程序观察"),
        ("effective_questions", "有效提问"),
        ("weak_answers", "薄弱回答"),
        ("neutral_observations", "中立观察"),
    ):
        text = as_text(review.get(key))
        if text:
            lines.append(f"{label}\n{text}")
    if lines:
        render_bubble("仲裁庭", "\n\n".join(lines), tone="tribunal")


def render_critic_review(review: dict[str, Any]) -> None:
    lines: list[str] = []
    summary = review.get("round_summary")
    if summary:
        lines.append(f"总评\n{summary}")
    for key, label in (("claimant", "申请方"), ("respondent", "被申请方")):
        side_review = review.get(key, {})
        if not isinstance(side_review, dict):
            continue
        score = side_review.get("score", "-")
        side_lines = [
            f"评分：{score}/10",
            as_text(side_review.get("comment")),
        ]
        strengths = as_text(side_review.get("strengths"))
        weaknesses = as_text(side_review.get("weaknesses"))
        risks = as_text(side_review.get("risk_points"))
        if strengths:
            side_lines.append(f"优势：\n{strengths}")
        if weaknesses:
            side_lines.append(f"问题：\n{weaknesses}")
        if risks:
            side_lines.append(f"风险：\n{risks}")
        lines.append(f"{label}\n" + "\n".join(part for part in side_lines if part))
    overall = as_text(review.get("overall_comments"))
    if overall:
        lines.append(f"其他观察\n{overall}")
    if lines:
        render_bubble("点评 Agent", "\n\n".join(lines), tone="critic")


def render_closing_statement(statement_record: dict[str, Any]) -> None:
    statement = statement_record.get("statement", {})
    if not isinstance(statement, dict):
        render_bubble(statement_record.get("speaker", "最后陈述"), as_text(statement), tone="closing")
        return
    meta = "\n\n".join(
        part
        for part in (
            f"核心主张：\n{as_text(statement.get('key_points'))}",
            f"利用的问答/证据：\n{as_text(statement.get('evidence_hooks'))}",
            f"风险控制：\n{as_text(statement.get('risk_control'))}",
            f"请求方向：\n{as_text(statement.get('requested_outcome'))}",
        )
        if part.strip() and not part.endswith("：\n")
    )
    render_bubble(
        statement_record.get("speaker", "最后陈述"),
        as_text(statement.get("statement")),
        meta=meta,
        tone="closing",
    )


def render_live_event_note(text: str, tone: str = "system") -> None:
    st.markdown(
        f"<div class='live-note {tone}'>{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


def render_reflection_event(event: dict[str, Any]) -> None:
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        render_bubble(event.get("speaker", "复盘 Agent"), event.get("content", ""), tone="reflection")
        return
    content_parts = [
        f"评分：{payload.get('round_score', '-')}",
        f"做得好的地方：\n{as_text(payload.get('what_worked'))}",
        f"需要修正的地方：\n{as_text(payload.get('what_failed'))}",
    ]
    meta = as_text(payload.get("new_lessons"))
    render_bubble(
        event.get("speaker", "复盘 Agent"),
        "\n\n".join(part for part in content_parts if part.strip()),
        meta=meta,
        tone="reflection",
    )


def render_strategy_event(event: dict[str, Any]) -> None:
    payload = event.get("payload", {})
    meta = ""
    if isinstance(payload, dict):
        meta = "\n\n".join(
            part
            for part in (
                f"盘问策略：\n{as_text(payload.get('lawyer_strategy'))}",
                f"应答策略：\n{as_text(payload.get('witness_strategy'))}",
            )
            if part.strip()
        )
    render_bubble(
        event.get("speaker", "策略 Agent"),
        event.get("content", "策略已更新。"),
        meta=meta,
        tone="strategy",
    )


def render_live_events(events: list[dict[str, Any]]) -> None:
    if not events:
        st.info("Agent 完成发言后，会在这里实时出现聊天气泡。")
        return

    for event in events:
        event_type = event.get("type")
        if event_type == "round_started":
            render_live_event_note(event.get("content", "新一轮模拟开始。"), tone="round")
        elif event_type in {"run_started", "initial_strategies"}:
            render_live_event_note(event.get("content", "模拟准备中。"))
        elif event_type == "question":
            render_bubble(
                event.get("speaker", "律师"),
                event.get("content", ""),
                meta=event.get("meta", ""),
                tone="question",
            )
        elif event_type == "answer":
            render_bubble(
                event.get("speaker", "证人"),
                event.get("content", ""),
                meta=event.get("meta", ""),
                tone="answer",
            )
        elif event_type == "closing_statement":
            payload = event.get("payload")
            if isinstance(payload, dict):
                render_closing_statement(payload)
            else:
                render_bubble(
                    event.get("speaker", "最后陈述"),
                    event.get("content", ""),
                    meta=event.get("meta", ""),
                    tone="closing",
                )
        elif event_type == "tribunal":
            payload = event.get("payload")
            if isinstance(payload, dict):
                render_tribunal_review(payload)
            else:
                render_bubble(event.get("speaker", "仲裁庭"), event.get("content", ""), tone="tribunal")
        elif event_type == "critic_review":
            payload = event.get("payload")
            if isinstance(payload, dict):
                render_critic_review(payload)
            else:
                render_bubble(event.get("speaker", "点评 Agent"), event.get("content", ""), tone="critic")
        elif event_type == "reflection":
            render_reflection_event(event)
        elif event_type == "strategy_update":
            render_strategy_event(event)
        elif event_type in {"final_advice", "run_completed"}:
            render_live_event_note(event.get("content", "模拟完成。"), tone="done")


def render_chat(payload: dict[str, Any]) -> None:
    public_rounds = payload.get("public_rounds", [])
    if not public_rounds:
        st.info("暂无模拟记录。")
        return

    for round_record in public_rounds:
        round_label = f"第 {round_record.get('round')} 轮"
        learning_side = round_record.get("learning_side", "")
        frozen_side = round_record.get("frozen_side", "")
        with st.expander(f"{round_label} · {learning_side}训练 · {frozen_side}冻结", expanded=round_record.get("round") == 1):
            for segment in round_record.get("segments", []):
                st.markdown(
                    f"<div class='segment-title'>环节 {segment.get('segment')}："
                    f"{html.escape(segment.get('questioner', ''))} 盘问 "
                    f"{html.escape(segment.get('witness', ''))}</div>",
                    unsafe_allow_html=True,
                )
                for turn in segment.get("turns", []):
                    question = turn.get("question", {})
                    answer = turn.get("answer", {})
                    render_bubble(
                        segment.get("questioner", "律师"),
                        as_text(question.get("question")),
                        meta=as_text(question.get("purpose")),
                        tone="question",
                    )
                    render_bubble(
                        segment.get("witness", "证人"),
                        as_text(answer.get("answer")),
                        meta=as_text(answer.get("defense_move")),
                        tone="answer",
                    )
                review = segment.get("tribunal_review")
                if isinstance(review, dict):
                    render_tribunal_review(review)
            closing_statements = round_record.get("closing_statements", [])
            if closing_statements:
                st.markdown("<div class='segment-title'>双方最后陈述</div>", unsafe_allow_html=True)
                for statement_record in closing_statements:
                    if isinstance(statement_record, dict):
                        render_closing_statement(statement_record)
            critic_review = round_record.get("critic_review")
            if isinstance(critic_review, dict):
                st.markdown("<div class='segment-title'>本轮点评 Agent 评价</div>", unsafe_allow_html=True)
                render_critic_review(critic_review)


def strategy_column(strategy: dict[str, Any], key: str) -> str:
    value = strategy.get(key, {})
    return as_text(value)


def critic_side_text(review: dict[str, Any], side: str) -> tuple[Any, str]:
    side_review = review.get(side, {})
    if not isinstance(side_review, dict):
        return "", ""
    detail = "\n\n".join(
        part
        for part in (
            as_text(side_review.get("comment")),
            f"优势：\n{as_text(side_review.get('strengths'))}",
            f"问题：\n{as_text(side_review.get('weaknesses'))}",
            f"风险：\n{as_text(side_review.get('risk_points'))}",
        )
        if part.strip() and not part.endswith("：\n")
    )
    return side_review.get("score", ""), detail


def build_training_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for update in payload.get("training_updates", []):
        reflection = update.get("reflection", {})
        strategy = update.get("updated_strategy", {})
        critic_review = update.get("critic_review", {})
        claimant_score, claimant_detail = critic_side_text(critic_review, "claimant")
        respondent_score, respondent_detail = critic_side_text(critic_review, "respondent")
        rows.append(
            {
                "轮次": update.get("round"),
                "训练方": update.get("learning_side"),
                "申请方点评分": claimant_score,
                "申请方点评意见": claimant_detail,
                "被申请方点评分": respondent_score,
                "被申请方点评意见": respondent_detail,
                "点评Agent总评": as_text(critic_review.get("round_summary")) if isinstance(critic_review, dict) else "",
                "训练方复盘评分": reflection.get("round_score"),
                "总结复盘": "\n\n".join(
                    part
                    for part in (
                        f"有效：\n{as_text(reflection.get('what_worked'))}",
                        f"不足：\n{as_text(reflection.get('what_failed'))}",
                        f"对方施压点：\n{as_text(reflection.get('opponent_pressure_points_seen'))}",
                    )
                    if part.strip()
                ),
                "更新后盘问策略": strategy_column(strategy, "lawyer_strategy"),
                "更新后应答策略": strategy_column(strategy, "witness_strategy"),
            }
        )
    return pd.DataFrame(rows)


def render_result(result: dict[str, Any], *, model: str, base_url: str, temperature: float) -> None:
    payload = result["payload"]
    result_key = str(result.get("run_dir", result["case_output_dir"]))
    metadata = result.get("metadata", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("案件", result["case_id"])
    col2.metric("阵营", payload.get("selected_party", "-"))
    col3.metric("轮数", payload.get("rounds", "-"))
    col4.metric("每环节问答", payload.get("qa_pairs_per_segment", "-"))
    if result.get("run_dir"):
        mode = "离线演示" if metadata.get("dry_run") else "模型运行"
        st.caption(f"记录目录：{compact_filename(result['run_dir'])} · {mode}")

    tab_chat, tab_advice, tab_training, tab_qa, tab_files = st.tabs(["模拟对话", "最终建议", "迭代记录", "智能问答", "输出文件"])

    with tab_chat:
        render_chat(payload)

    with tab_advice:
        processed_advice = render_markdown_with_rag_cards(result["advice"])
        st.markdown(processed_advice, unsafe_allow_html=True)

    with tab_training:
        frame = build_training_dataframe(payload)
        if frame.empty:
            st.info("暂无迭代记录。")
        else:
            st.dataframe(frame, use_container_width=True, height=520, hide_index=True)
            st.download_button(
                "下载迭代表格 CSV",
                frame.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{result['case_id']}_training_updates.csv",
                mime="text/csv",
                key=f"download-training-{result_key}",
            )
        with st.expander("Markdown 总结"):
            processed_summary = render_markdown_with_rag_cards(result["summary"])
            st.markdown(processed_summary, unsafe_allow_html=True)

    with tab_qa:
        render_result_qa_agent(result, model=model, base_url=base_url, temperature=temperature)

    with tab_files:
        st.write(compact_filename(result["case_output_dir"]))
        frame = build_training_dataframe(payload)
        st.download_button(
            "下载庭前建议",
            result["advice"].encode("utf-8"),
            file_name=result["advice_path"].name,
            mime="text/markdown",
            key=f"download-advice-{result_key}",
        )
        if result.get("summary"):
            st.download_button(
                "下载训练过程总结",
                result["summary"].encode("utf-8"),
                file_name=result["summary_path"].name,
                mime="text/markdown",
                key=f"download-summary-{result_key}",
            )
        st.download_button(
            "下载模拟记录 JSON",
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=result["log_path"].name,
            mime="application/json",
            key=f"download-log-{result_key}",
        )
        if not frame.empty:
            st.download_button(
                "下载迭代表格 CSV",
                frame.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{result['case_id']}_training_updates.csv",
                mime="text/csv",
                key=f"download-training-files-{result_key}",
            )
        events_path = result.get("events_path")
        if isinstance(events_path, Path) and events_path.exists():
            st.download_button(
                "下载实时事件流 JSONL",
                events_path.read_bytes(),
                file_name=events_path.name,
                mime="application/jsonl",
                key=f"download-events-{result_key}",
            )
        metadata_path = result.get("metadata_path")
        if isinstance(metadata_path, Path) and metadata_path.exists():
            st.download_button(
                "下载运行参数 JSON",
                metadata_path.read_bytes(),
                file_name=metadata_path.name,
                mime="application/json",
                key=f"download-metadata-{result_key}",
            )
            with st.expander("运行参数"):
                st.json(result.get("metadata", {}))
        files = sorted(path for path in result["case_output_dir"].glob("*") if path.is_file())
        if files:
            with st.expander("产出文件清单", expanded=True):
                for path in files:
                    st.write(f"- {compact_filename(path)}")
        run_dir = result.get("run_dir")
        if isinstance(run_dir, Path) and run_dir.exists():
            run_files = sorted(path for path in run_dir.rglob("*") if path.is_file())
            with st.expander("运行目录文件清单"):
                for path in run_files:
                    st.write(f"- {compact_filename(path)}")


def add_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; max-width: 1320px; }
        .segment-title {
            margin: 1.1rem 0 .65rem;
            padding: .45rem .65rem;
            border-left: 4px solid #b45309;
            background: #fff7ed;
            color: #3f3f46;
            font-weight: 650;
        }
        .chat-row {
            display: grid;
            grid-template-columns: 42px minmax(0, 1fr);
            gap: .7rem;
            margin: .55rem 0;
            align-items: start;
        }
        .avatar {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: .82rem;
            font-weight: 700;
            color: #ffffff;
            background: #334155;
        }
        .bubble {
            border: 1px solid #e4e4e7;
            border-radius: 8px;
            padding: .7rem .85rem;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, .05);
        }
        .question .avatar { background: #7c2d12; }
        .answer .avatar { background: #155e75; }
        .closing .avatar { background: #4d7c0f; }
        .closing .bubble { background: #f7fee7; }
        .tribunal .avatar { background: #52525b; }
        .tribunal .bubble { background: #f4f4f5; }
        .critic .avatar { background: #854d0e; }
        .critic .bubble { background: #fffbeb; }
        .reflection .avatar { background: #6b21a8; }
        .reflection .bubble { background: #faf5ff; }
        .strategy .avatar { background: #166534; }
        .strategy .bubble { background: #f0fdf4; }
        .speaker {
            margin-bottom: .3rem;
            color: #18181b;
            font-size: .88rem;
            font-weight: 700;
        }
        .bubble-content {
            color: #27272a;
            line-height: 1.72;
        }
        .bubble-meta {
            margin-top: .55rem;
            padding-top: .45rem;
            border-top: 1px dashed #d4d4d8;
            color: #52525b;
            font-size: .86rem;
            line-height: 1.55;
        }
        .live-note {
            margin: .65rem 0;
            padding: .5rem .7rem;
            border: 1px solid #e4e4e7;
            border-radius: 8px;
            background: #fafafa;
            color: #3f3f46;
            font-size: .9rem;
        }
        .live-note.round {
            border-left: 4px solid #b45309;
            background: #fff7ed;
            font-weight: 650;
        }
        .live-note.done {
            border-left: 4px solid #15803d;
            background: #f0fdf4;
        }
        /* ── RAG citation cards ── */
        .rag-card {
            border: 1.5px solid #e2e8f0;
            border-radius: 10px;
            margin: 0.65rem 0;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
            font-size: 0.88rem;
        }
        .rag-card-case {
            border-left: 4px solid #0891b2;
            background: #f0fdff;
        }
        .rag-card-law {
            border-left: 4px solid #059669;
            background: #ecfdf5;
        }
        .rag-card-result {
            border-left: 4px solid #b45309;
            background: #fff7ed;
        }
        .rag-card-default {
            border-left: 4px solid #6366f1;
            background: #fafafe;
        }
        .rag-card-header {
            padding: 0.45rem 0.7rem;
            background: rgba(0, 0, 0, 0.025);
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }
        .rag-card-icon {
            font-size: 1rem;
            flex-shrink: 0;
        }
        .rag-card-source {
            font-weight: 700;
            color: #334155;
        }
        .rag-card-id {
            color: #64748b;
            font-family: "SF Mono", "Fira Code", "Consolas", monospace;
            font-size: 0.75rem;
            margin-left: auto;
            background: rgba(0, 0, 0, 0.05);
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
        }
        .rag-card-body {
            padding: 0.6rem 0.7rem;
        }
        .rag-card-excerpt {
            color: #334155;
            line-height: 1.68;
        }
        .rag-detail {
            margin: 0;
            border-top: 1px solid #e2e8f0;
        }
        .rag-detail summary {
            padding: 0.45rem 0.7rem;
            cursor: pointer;
            color: #2563eb;
            font-size: 0.84rem;
            font-weight: 500;
            user-select: none;
        }
        .rag-detail summary:hover {
            background: rgba(37, 99, 235, 0.06);
        }
        .rag-source-content {
            padding: 0.7rem;
            max-height: 320px;
            overflow-y: auto;
            background: #f8fafc;
            border-top: 1px solid #e2e8f0;
            font-size: 0.84rem;
            line-height: 1.7;
            white-space: pre-wrap;
            color: #475569;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="庭前仲裁模拟工作台", layout="wide")
    add_styles()

    st.title("庭前仲裁模拟工作台")

    sample_cases = list_case_docs()
    rule_docs = list_rule_docs()
    history_entries = list_history_entries()
    selected_case_text = ""

    if "last_result" not in st.session_state:
        latest_completed = next((entry for entry in history_entries if entry.get("status") == "完成"), None)
        if latest_completed is not None:
            loaded = load_history_entry(latest_completed)
            if loaded is not None:
                st.session_state["last_result"] = loaded
                st.session_state["active_result_label"] = latest_completed["label"]

    with st.sidebar:
        st.subheader("历史记录")
        if history_entries:
            selected_history = st.selectbox(
                "已保存分析",
                history_entries,
                format_func=lambda entry: entry["label"],
            )
            if selected_history.get("status") != "完成":
                st.caption("这条记录没有完整输出，可能是运行中断或失败。")
            if st.button("打开历史记录", use_container_width=True):
                loaded = load_history_entry(selected_history)
                if loaded is None:
                    st.warning("这条历史记录没有可加载的模拟结果。")
                else:
                    st.session_state["last_result"] = loaded
                    st.session_state["active_result_label"] = selected_history["label"]
            if st.button("刷新历史列表", use_container_width=True):
                st.rerun()
        else:
            st.caption("暂无历史记录。完成一次模拟后会自动保存在这里。")

        st.divider()
        st.subheader("案件")
        source_mode = st.radio("来源", ["示例案件", "上传文件"], horizontal=True)
        uploaded_file = None
        selected_sample = None
        if source_mode == "示例案件":
            if not sample_cases:
                st.warning("未找到示例案件。")
            else:
                selected_sample = st.selectbox(
                    "案件文件",
                    sample_cases,
                    format_func=lambda path: compact_filename(path),
                )
                selected_case_text = read_text(selected_sample)
        else:
            uploaded_file = st.file_uploader("上传 Markdown/TXT 案件", type=["md", "txt"])
            if uploaded_file is not None:
                selected_case_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

        st.subheader("规则")
        law = st.selectbox("适用法律", ["中国大陆法律"])
        inferred_docs = infer_rule_docs(selected_case_text, rule_docs) if selected_case_text else []
        rule_options = [AUTO_RULE_OPTION, ALL_RULES_OPTION] + [path.name for path in rule_docs]
        default_rule_index = 0 if inferred_docs else min(1, len(rule_options) - 1)
        rule_choice = st.selectbox("仲裁规则", rule_options, index=default_rule_index)
        if rule_choice == AUTO_RULE_OPTION and inferred_docs:
            st.caption("识别到：" + "、".join(path.stem for path in inferred_docs))

        st.subheader("模拟")
        position_label = st.selectbox("训练阵营", list(POSITION_OPTIONS.keys()))
        rounds = st.number_input("轮数", min_value=1, max_value=10, value=3, step=1)
        qa_pairs = st.number_input("每环节问答组数", min_value=1, max_value=5, value=2, step=1)
        strategy_block_size = st.number_input("策略交替块大小", min_value=1, max_value=10, value=1, step=1)
        dry_run_default = not bool(os.getenv("OPENAI_API_KEY"))
        dry_run = st.toggle("离线演示模式", value=dry_run_default)
        skip_tribunal = st.toggle("跳过仲裁庭点评", value=False)
        disable_rag = st.toggle("关闭本地 RAG", value=False)

        with st.expander("模型参数"):
            model = st.text_input("模型", value=os.getenv("OPENAI_MODEL", "qwen3.6-flash"))
            base_url = st.text_input("Base URL", value=os.getenv("OPENAI_BASE_URL", ""))
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        run_clicked = st.button("开始模拟", type="primary", use_container_width=True)

    if selected_case_text:
        with st.expander("案件材料预览", expanded=False):
            st.markdown(selected_case_text[:12000])

    if run_clicked:
        if source_mode == "上传文件" and uploaded_file is None:
            st.error("请先上传案件文件。")
            st.stop()
        if source_mode == "示例案件" and selected_sample is None:
            st.error("请先选择示例案件。")
            st.stop()

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / timestamp
        output_root = run_dir / "results"
        events_path = run_dir / "events.jsonl"
        run_dir.mkdir(parents=True, exist_ok=True)

        case_doc = selected_sample if selected_sample is not None else save_uploaded_case(run_dir, uploaded_file)
        if rule_choice == AUTO_RULE_OPTION:
            selected_rule_docs = inferred_docs or rule_docs
        elif rule_choice == ALL_RULES_OPTION:
            selected_rule_docs = rule_docs
        else:
            selected_rule_docs = [path for path in rule_docs if path.name == rule_choice]

        effective_rules_dir = RULES_DIR
        if selected_rule_docs and len(selected_rule_docs) != len(rule_docs):
            effective_rules_dir = make_selected_rules_dir(run_dir, selected_rule_docs)

        metadata = {
            "case_doc": str(case_doc),
            "law": law,
            "rule_choice": rule_choice,
            "effective_rules": [path.name for path in selected_rule_docs],
            "position": POSITION_OPTIONS[position_label],
            "position_label": position_label,
            "rounds": int(rounds),
            "qa_pairs": int(qa_pairs),
            "strategy_block_size": int(strategy_block_size),
            "dry_run": dry_run,
            "skip_tribunal": skip_tribunal,
            "disable_rag": disable_rag,
            "model": model,
            "base_url": base_url,
            "temperature": temperature,
            "events_path": str(events_path),
        }
        (run_dir / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        command, env = build_command(
            case_doc=case_doc,
            output_root=output_root,
            position=POSITION_OPTIONS[position_label],
            rounds=int(rounds),
            qa_pairs=int(qa_pairs),
            strategy_block_size=int(strategy_block_size),
            dry_run=dry_run,
            skip_tribunal=skip_tribunal,
            disable_rag=disable_rag,
            rules_dir=effective_rules_dir,
            model=model,
            base_url=base_url,
            temperature=temperature,
            events_path=events_path,
        )

        with st.status("模拟运行中", expanded=True) as status:
            return_code, logs = run_simulation(command, env, events_path)
            if return_code == 0:
                status.update(label="模拟完成", state="complete")
            else:
                status.update(label="模拟失败", state="error")
                st.error("模拟脚本返回错误。")
                st.session_state["last_logs"] = logs
                st.stop()

        st.session_state["last_result"] = load_result(case_doc, output_root, POSITION_OPTIONS[position_label])
        st.session_state["active_result_label"] = f"刚完成 · {st.session_state['last_result']['case_id']} · {position_label}"
        st.session_state["last_logs"] = logs

    if "last_result" in st.session_state:
        if st.session_state.get("active_result_label"):
            st.info(f"当前展示：{st.session_state['active_result_label']}")
        render_result(st.session_state["last_result"], model=model, base_url=base_url, temperature=temperature)


if __name__ == "__main__":
    main()
