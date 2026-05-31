from __future__ import annotations

import datetime as dt
import html
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
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


def render_bubble(speaker: str, content: str, meta: str = "", tone: str = "neutral") -> None:
    initials = speaker[:2] if speaker else "AI"
    escaped_speaker = html.escape(speaker)
    escaped_content = html.escape(content).replace("\n", "<br>")
    escaped_meta = html.escape(meta).replace("\n", "<br>")
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


def render_result(result: dict[str, Any]) -> None:
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

    tab_chat, tab_advice, tab_training, tab_files = st.tabs(["模拟对话", "最终建议", "迭代记录", "输出文件"])

    with tab_chat:
        render_chat(payload)

    with tab_advice:
        st.markdown(result["advice"])

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
            st.markdown(result["summary"])

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
        render_result(st.session_state["last_result"])


if __name__ == "__main__":
    main()
