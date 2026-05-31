"""Preprocess Jus Mundi case data for the arbitration simulator.

The pipeline has two separable stages:

1. Build a leakage-safe split between the public RAG corpus and test cases.
2. Optionally call an OpenAI-compatible LLM to turn test PDFs into Chinese
   pre-hearing simulation case files under inputs/test_case/caseXXX/main.md.

Default execution is offline and deterministic. It extracts PDF text, writes a
manifest, writes RAG corpus files, and writes source packs/prompts for test
cases without calling an API.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from tqdm import tqdm

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Missing dependency: PyMuPDF. Install it with `pip install pymupdf`.") from exc

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Missing dependency: pandas. Install it with `pip install pandas openpyxl`.") from exc


ROOT = Path(__file__).resolve().parents[1]

TITLE_COL = "案件标题 (Case Title)"
CASE_NO_COL = "案号 (Case Number)"
PDF_COL = "PDF 保存路径"
URL_COL = "案例 URL"


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    title: str
    case_number: str
    pdf_path: Path
    pdf_sha256: str
    text_sha256: str
    page_count: int
    char_count: int
    metadata: dict[str, Any]
    source_rows: list[int]
    urls: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess case PDFs into leakage-safe RAG/test datasets.",
    )
    parser.add_argument("--metadata", default="original_case_raw/jusmundi_cases.xlsx")
    parser.add_argument("--pdf-dir", default="original_case_raw/pdfs")
    parser.add_argument("--cleaning-prompt", default="scripts/原始数据清洗prompt.md")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--inputs-dir", default="inputs/test_case")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-size",
        type=int,
        default=2,
        help="Number of usable cases to reserve as test cases when --test-cases is not provided.",
    )
    parser.add_argument(
        "--test-cases",
        nargs="*",
        default=None,
        help="Explicit case ids to use as test cases, e.g. case001 case003.",
    )
    parser.add_argument(
        "--rag-cases",
        nargs="*",
        default=None,
        help="Explicit case ids to use as RAG cases. Remaining cases become tests.",
    )
    parser.add_argument(
        "--generate-test-cases",
        action="store_true",
        help="Call an OpenAI-compatible model and write inputs/test_case/caseXXX/main.md.",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "qwen3.6-flash"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--max-source-chars",
        type=int,
        default=99999999999999,
        help="Maximum PDF text characters sent to the model per case.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated inputs/test_case/caseXXX/main.md files.",
    )
    return parser.parse_args()


def resolve_root_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.lower()).strip("-")
    return slug or "case"


def normalize_key(row: dict[str, Any]) -> tuple[str, str, str]:
    title = clean_cell(row.get(TITLE_COL)).lower()
    case_no = clean_cell(row.get(CASE_NO_COL)).lower()
    pdf_name = Path(clean_cell(row.get(PDF_COL))).name.lower()
    return title, case_no, pdf_name


def sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_pdf_text(path: Path) -> tuple[str, int]:
    with fitz.open(path) as doc:
        pages = [page.get_text("text") for page in doc]
        return "\n\n".join(pages).strip(), doc.page_count


def unique_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = clean_cell(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def row_to_metadata(row: dict[str, Any]) -> dict[str, str]:
    return {
        str(key): clean_cell(value)
        for key, value in row.items()
        if clean_cell(value)
    }


def find_pdf_path(raw_pdf_value: str, pdf_dir: Path) -> Path | None:
    raw_path = Path(raw_pdf_value) if raw_pdf_value else Path()
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    elif raw_pdf_value:
        candidates.append(ROOT / raw_path)

    if raw_path.name:
        candidates.append(pdf_dir / raw_path.name)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def load_cases(metadata_path: Path, pdf_dir: Path) -> tuple[list[CaseRecord], list[dict[str, Any]]]:
    df = pd.read_excel(metadata_path)
    raw_rows = df.to_dict(orient="records")

    grouped: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = {}
    skipped: list[dict[str, Any]] = []

    for index, row in enumerate(raw_rows):
        title = clean_cell(row.get(TITLE_COL))
        raw_pdf = clean_cell(row.get(PDF_COL))
        if not title and not raw_pdf:
            skipped.append({"row": index + 2, "reason": "empty row"})
            continue
        if not raw_pdf:
            skipped.append({"row": index + 2, "title": title, "reason": "missing PDF path"})
            continue

        pdf_path = find_pdf_path(raw_pdf, pdf_dir)
        if not pdf_path:
            skipped.append(
                {
                    "row": index + 2,
                    "title": title,
                    "raw_pdf_path": raw_pdf,
                    "reason": "PDF file not found under --pdf-dir",
                }
            )
            continue

        row = dict(row)
        row[PDF_COL] = str(pdf_path)
        key = normalize_key(row)
        grouped.setdefault(key, []).append((index, row))

    records: list[CaseRecord] = []
    for serial, (_, rows) in enumerate(grouped.items(), start=1):
        first_index, first_row = rows[0]
        pdf_path = Path(clean_cell(first_row[PDF_COL]))
        text, extracted_pages = extract_pdf_text(pdf_path)
        title = clean_cell(first_row.get(TITLE_COL)) or pdf_path.stem
        case_number = clean_cell(first_row.get(CASE_NO_COL))
        metadata = row_to_metadata(first_row)
        metadata["local_pdf_path"] = str(pdf_path)
        metadata["source_pdf_name"] = pdf_path.name
        metadata["source_rows"] = ", ".join(str(index + 2) for index, _ in rows)
        metadata["deduplicated_row_count"] = str(len(rows))

        urls = unique_strings(row.get(URL_COL) for _, row in rows)
        documents = unique_strings(row.get("案件文档 (Documents)") for _, row in rows)
        if urls:
            metadata["case_urls"] = "\n".join(urls)
        if documents:
            metadata["case_documents"] = "\n".join(documents)

        records.append(
            CaseRecord(
                case_id=f"case{serial:03d}",
                title=title,
                case_number=case_number,
                pdf_path=pdf_path,
                pdf_sha256=sha256_bytes(pdf_path),
                text_sha256=sha256_text(text),
                page_count=extracted_pages,
                char_count=len(text),
                metadata=metadata,
                source_rows=[first_index + 2 for first_index, _ in rows],
                urls=urls,
            )
        )

    return records, skipped


def choose_split(
    cases: list[CaseRecord],
    seed: int,
    test_size: int,
    test_cases: list[str] | None,
    rag_cases: list[str] | None,
) -> tuple[list[str], list[str]]:
    all_ids = [case.case_id for case in cases]
    all_id_set = set(all_ids)

    if test_cases and rag_cases:
        overlap = set(test_cases) & set(rag_cases)
        if overlap:
            raise ValueError(f"Cases cannot be both test and RAG: {sorted(overlap)}")
        unknown = (set(test_cases) | set(rag_cases)) - all_id_set
        if unknown:
            raise ValueError(f"Unknown case ids: {sorted(unknown)}")
        test_ids = list(test_cases)
        rag_ids = list(rag_cases)
        leftover = [case_id for case_id in all_ids if case_id not in set(test_ids) | set(rag_ids)]
        rag_ids.extend(leftover)
    elif test_cases:
        unknown = set(test_cases) - all_id_set
        if unknown:
            raise ValueError(f"Unknown test case ids: {sorted(unknown)}")
        test_ids = list(test_cases)
        rag_ids = [case_id for case_id in all_ids if case_id not in set(test_ids)]
    elif rag_cases:
        unknown = set(rag_cases) - all_id_set
        if unknown:
            raise ValueError(f"Unknown RAG case ids: {sorted(unknown)}")
        rag_ids = list(rag_cases)
        test_ids = [case_id for case_id in all_ids if case_id not in set(rag_ids)]
    else:
        if test_size <= 0:
            raise ValueError("--test-size must be positive.")
        if test_size >= len(cases):
            raise ValueError("--test-size must be smaller than the number of usable cases.")
        rng = random.Random(seed)
        test_ids = sorted(rng.sample(all_ids, test_size))
        rag_ids = [case_id for case_id in all_ids if case_id not in set(test_ids)]

    if not test_ids:
        raise ValueError("At least one test case is required.")
    if not rag_ids:
        raise ValueError("At least one RAG case is required.")
    if set(test_ids) & set(rag_ids):
        raise ValueError("Leakage check failed: RAG/test split overlaps.")
    return rag_ids, test_ids


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def markdown_metadata(metadata: dict[str, Any]) -> str:
    lines = ["## Metadata"]
    for key, value in metadata.items():
        if value:
            value_text = str(value).replace("\r\n", "\n")
            if "\n" in value_text:
                lines.append(f"- {key}:")
                for sub_line in value_text.splitlines():
                    lines.append(f"  {sub_line}")
            else:
                lines.append(f"- {key}: {value_text}")
    return "\n".join(lines)


def write_case_source(path: Path, case: CaseRecord, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n\n".join(
        [
            f"# {case.case_id}: {case.title}",
            markdown_metadata(case.metadata),
            "## Extracted PDF Text",
            text,
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def load_cleaning_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_llm_messages(
    cleaning_prompt: str,
    case: CaseRecord,
    source_text: str,
    max_source_chars: int,
) -> list[dict[str, str]]:
    truncated = source_text[:max_source_chars]
    truncation_note = ""
    if len(source_text) > max_source_chars:
        truncation_note = (
            f"\n\n[Note: source text was truncated from {len(source_text)} to "
            f"{max_source_chars} characters for this request.]"
        )

    user_content = "\n\n".join(
        [
            cleaning_prompt,
            "下面是真实案例的元数据和 PDF 提取文本。请基于这些材料生成中文模拟案例 main.md。",
            markdown_metadata(case.metadata),
            "## PDF 提取文本",
            truncated + truncation_note,
        ]
    )
    return [
        {
            "role": "system",
            "content": (
                "You transform real arbitration/court case materials into Chinese "
                "pre-hearing simulation case files. Do not invent facts. Exclude final "
                "rulings, post-hearing reasoning, and outcome-only information unless "
                "it is needed only as a factual procedural background."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def generate_with_openai(
    messages: list[dict[str, str]],
    model: str,
    base_url: str | None,
    api_key_env: str,
    temperature: float,
) -> str:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {api_key_env} before using --generate-test-cases.")

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Missing dependency: openai. Install it with `pip install openai`.") from exc

    client_kwargs: dict[str, str] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    
    
    
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    content = completion.choices[0].message.content
    
    print("---------------- llm's response ----------------")
    truncate_len = min(len(content),1000)//2
    print(content[:truncate_len],"...",content[-truncate_len:])
    print("-----------------------------------------------")
    
    
    if not content:
        raise RuntimeError("Model returned empty content.")
    return content.strip()


def assert_no_hash_leakage(cases: list[CaseRecord], rag_ids: list[str], test_ids: list[str]) -> None:
    by_id = {case.case_id: case for case in cases}
    rag_pdf_hashes = {by_id[case_id].pdf_sha256 for case_id in rag_ids}
    test_pdf_hashes = {by_id[case_id].pdf_sha256 for case_id in test_ids}
    rag_text_hashes = {by_id[case_id].text_sha256 for case_id in rag_ids}
    test_text_hashes = {by_id[case_id].text_sha256 for case_id in test_ids}
    if rag_pdf_hashes & test_pdf_hashes or rag_text_hashes & test_text_hashes:
        raise ValueError("Leakage check failed: identical PDF/text hash appears in both splits.")


def case_summary(case: CaseRecord) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "title": case.title,
        "case_number": case.case_number,
        "pdf_name": case.pdf_path.name,
        "page_count": case.page_count,
        "char_count": case.char_count,
        "pdf_sha256": case.pdf_sha256,
        "text_sha256": case.text_sha256,
        "source_rows": case.source_rows,
        "urls": case.urls,
        "metadata": case.metadata,
    }


def write_cleaning_prompt(path: Path, cleaning_prompt: str, case: CaseRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n\n".join(
        [
            "# Cleaning Prompt",
            cleaning_prompt,
            "## Case Metadata",
            markdown_metadata(case.metadata),
            "## Source Text",
            "See source_text.md in this directory. The generation CLI reads the same extracted text directly.",
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def run() -> int:
    args = parse_args()
    metadata_path = resolve_root_path(args.metadata)
    pdf_dir = resolve_root_path(args.pdf_dir)
    processed_dir = resolve_root_path(args.processed_dir)
    inputs_dir = resolve_root_path(args.inputs_dir)
    cleaning_prompt_path = resolve_root_path(args.cleaning_prompt)

    cases, skipped = load_cases(metadata_path, pdf_dir)
    if len(cases) < 2:
        raise RuntimeError("Need at least two usable PDF-backed cases for a split.")

    rag_ids, test_ids = choose_split(
        cases=cases,
        seed=args.seed,
        test_size=args.test_size,
        test_cases=args.test_cases,
        rag_cases=args.rag_cases,
    )
    assert_no_hash_leakage(cases, rag_ids, test_ids)

    cleaning_prompt = load_cleaning_prompt(cleaning_prompt_path)
    by_id = {case.case_id: case for case in cases}

    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metadata_file": str(metadata_path),
        "pdf_dir": str(pdf_dir),
        "cleaning_prompt": str(cleaning_prompt_path),
        "seed": args.seed,
        "rag_case_ids": rag_ids,
        "test_case_ids": test_ids,
        "usable_case_count": len(cases),
        "skipped_rows": skipped,
        "cases": [case_summary(case) for case in cases],
    }

    write_json(processed_dir / "manifest.json", manifest)
    write_json(
        processed_dir / "splits.json",
        {
            "rag_case_ids": rag_ids,
            "test_case_ids": test_ids,
            "leakage_policy": "No case_id, PDF hash, or extracted-text hash may appear in both splits.",
        },
    )

    for case_id in tqdm(rag_ids,desc='prepross rag data'):
        case = by_id[case_id]
        text, _ = extract_pdf_text(case.pdf_path)
        case_dir = processed_dir / "rag_corpus" / case.case_id
        write_json(case_dir / "metadata.json", case_summary(case))
        write_case_source(case_dir / "text.md", case, text)

    for case_id in tqdm(test_ids,desc='prepross test data'):
        case = by_id[case_id]
        text, _ = extract_pdf_text(case.pdf_path)
        source_dir = processed_dir / "test_case_sources" / case.case_id
        write_json(source_dir / "metadata.json", case_summary(case))
        write_case_source(source_dir / "source_text.md", case, text)
        write_cleaning_prompt(source_dir / "cleaning_prompt.md", cleaning_prompt, case)

        if args.generate_test_cases:
            output_path = inputs_dir / case.case_id / "main.md"
            if output_path.exists() and not args.overwrite:
                print(f"Skip existing generated case: {output_path}", file=sys.stderr)
                continue
            messages = build_llm_messages(
                cleaning_prompt=cleaning_prompt,
                case=case,
                source_text=text,
                max_source_chars=args.max_source_chars,
            )
            generated = generate_with_openai(
                messages=messages,
                model=args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                temperature=args.temperature,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(generated.rstrip() + "\n", encoding="utf-8")

    print(f"Usable cases: {len(cases)}")
    print(f"RAG cases: {len(rag_ids)} -> {', '.join(rag_ids)}")
    print(f"Test cases: {len(test_ids)} -> {', '.join(test_ids)}")
    print(f"Wrote: {processed_dir}")
    if args.generate_test_cases:
        print(f"Generated test cases under: {inputs_dir}")
    else:
        print("LLM generation skipped. Use --generate-test-cases to write inputs/test_case/caseXXX/main.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
