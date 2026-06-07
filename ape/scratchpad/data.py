"""Dataset adapters for scratchpad/gist-token QA experiments.

The adapters intentionally mirror the small set-oriented conversion style used
in the neighboring sparse-docs project, but keep this package self-contained so
the APE repo can be run on its own.
"""

from __future__ import annotations

import gzip
import json
import logging
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = "Use the provided contexts to answer the question."

DEFAULT_SOURCES: list[dict[str, Any]] = [
    {
        "name": "hotpotqa",
        "kind": "context_qa",
        "hf_path": "hotpotqa/hotpot_qa",
        "hf_config": "fullwiki",
        "train_split": "train",
        "eval_split": "validation",
    },
    {
        "name": "2wikimultihopqa",
        "kind": "context_qa",
        "hf_path": "voidful/2WikiMultihopQA",
        "hf_config": "default",
        "train_split": "train",
        "eval_split": "validation",
    },
    {
        "name": "musique",
        "kind": "musique",
        "hf_path": "dgslibisey/MuSiQue",
        "train_split": "train",
        "eval_split": "validation",
    },
    {
        "name": "wikihop",
        "kind": "qangaroo",
        "hf_path": "community-datasets/qangaroo",
        "hf_config": "wikihop",
        "train_split": "train",
        "eval_split": "validation",
        "streaming": False,
    },
    {
        "name": "msmarco",
        "kind": "msmarco",
        "hf_path": "microsoft/ms_marco",
        "hf_config": "v2.1",
        "train_split": "train",
        "eval_split": "validation",
    },
    {
        "name": "triviaqa",
        "kind": "triviaqa",
        "hf_path": "mandarjoshi/trivia_qa",
        "hf_config": "rc",
        "train_split": "train",
        "eval_split": "validation",
    },
]


def clean(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def clean_text_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(clean(part) for part in value if clean(part)).strip()
    return clean(value)


def sequence_to_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], dict):
            return [dict(item) for item in value]
        return [{"value": item} for item in value]
    if isinstance(value, dict):
        keys = list(value.keys())
        lengths = [len(value[key]) for key in keys if isinstance(value.get(key), (list, tuple))]
        if not lengths:
            return [dict(value)]
        count = min(lengths)
        records = []
        for idx in range(count):
            record = {}
            for key in keys:
                item = value[key]
                record[key] = item[idx] if isinstance(item, (list, tuple)) else item
            records.append(record)
        return records
    return []


def answer_candidates(value: Any) -> list[str]:
    if isinstance(value, list):
        answers = []
        for answer in value:
            if isinstance(answer, dict):
                answer = answer.get("text", answer.get("answer", ""))
            answers.append(clean(answer))
        return [answer for answer in answers if answer]
    answer = clean(value)
    return [answer] if answer else []


def answers_from_row(row: dict[str, Any]) -> list[str]:
    raw = row.get("golden_answers", row.get("answers", row.get("answer", [])))
    return answer_candidates(raw)


def answer_aliases_from_value(value: Any) -> list[str]:
    if isinstance(value, dict):
        answers = []
        for key in ("value", "normalized_value", "text", "answer"):
            text = clean(value.get(key, ""))
            if text:
                answers.append(text)
        for key in ("aliases", "normalized_aliases"):
            aliases = value.get(key, [])
            if isinstance(aliases, (list, tuple)):
                answers.extend(clean(alias) for alias in aliases if clean(alias))
        return list(dict.fromkeys(answer for answer in answers if answer))
    if isinstance(value, (list, tuple)):
        return [clean(answer) for answer in value if clean(answer)]
    text = clean(value)
    return [text] if text else []


def support_titles(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {})
    facts = row.get("supporting_facts")
    if facts is None and isinstance(metadata, dict):
        facts = metadata.get("supporting_facts", {})
    titles = facts.get("title", []) if isinstance(facts, dict) else []
    return {clean(title) for title in titles if clean(title)}


def cap_docs(docs: list[dict[str, Any]], max_docs: int | None) -> list[dict[str, Any]]:
    if max_docs is None or int(max_docs) <= 0:
        return docs
    return docs[: int(max_docs)]


def context_items(
    context: Any,
    source: str,
    row_idx: int,
    gold_titles: set[str] | None = None,
    max_docs: int | None = None,
) -> list[dict[str, Any]]:
    docs = []
    gold_titles = gold_titles or set()
    for item_idx, record in enumerate(sequence_to_records(context)):
        title = clean(record.get("title", ""))
        body = record.get(
            "sentences",
            record.get(
                "content",
                record.get(
                    "paragraphs",
                    record.get(
                        "text",
                        record.get(
                            "passage",
                            record.get(
                                "paragraph",
                                record.get(
                                    "wiki_context",
                                    record.get("search_context", record.get("value", "")),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        if isinstance(body, (list, tuple)):
            body = " ".join(clean(sentence) for sentence in body if clean(sentence))
        else:
            body = clean(body)
        if not body:
            continue
        docs.append(
            {
                "id": f"{source}-{row_idx}-doc-{item_idx}",
                "title": title,
                "text": body,
                "is_gold": bool(title and title in gold_titles),
                "metadata": {"source_item_index": item_idx},
            }
        )
    return cap_docs(docs, max_docs)


def context_from_row(row: dict[str, Any]) -> Any:
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("context", "ctxs", "passages", "supports"):
            if key in metadata:
                return metadata[key]
        if "summary" in metadata:
            summary = metadata["summary"]
            if isinstance(summary, dict):
                return [{"title": "summary", "content": summary.get("text", "")}]
            return [{"title": "summary", "content": summary}]
        if "text" in metadata:
            return [{"title": metadata.get("title", ""), "content": metadata.get("text", "")}]
    for key in ("context", "ctxs", "passages", "supports"):
        if key in row:
            return row[key]
    return None


def convert_context_qa_row(
    row: dict[str, Any],
    row_idx: int,
    source: str,
    instruction: str,
    max_docs: int | None,
) -> dict[str, Any] | None:
    question = row.get("question", row.get("query", ""))
    if isinstance(question, dict):
        question = question.get("text", question.get("question", ""))
    question = clean(question)
    answers = answers_from_row(row)
    context = context_from_row(row)
    if not question or not answers or context is None:
        return None
    docs = context_items(context, source, row_idx, support_titles(row), max_docs)
    if not docs:
        return None
    return {
        "id": f"{source}-{row.get('id', row_idx)}",
        "source": source,
        "task": "qa",
        "instruction": instruction,
        "question": question,
        "documents": docs,
        "answers": answers,
        "answer": answers[0],
        "metadata": {"raw_id": row.get("id")},
    }


def convert_msmarco_row(
    row: dict[str, Any],
    row_idx: int,
    source: str,
    instruction: str,
    max_docs: int | None,
) -> dict[str, Any] | None:
    question = clean(row.get("query", row.get("question", "")))
    phrase_answers = answers_from_row(row)
    well_formed_answers = [
        answer
        for answer in answer_candidates(row.get("wellFormedAnswers"))
        if answer.lower() != "no answer present"
    ]
    answers = list(dict.fromkeys([*well_formed_answers, *phrase_answers]))
    passages = row.get("passages", {})
    if not question or not answers or not isinstance(passages, dict):
        return None
    texts = passages.get("passage_text", passages.get("text", []))
    selected = passages.get("is_selected", [])
    urls = passages.get("url", [])
    docs = []
    for item_idx, text in enumerate(texts):
        body = clean(text)
        if not body:
            continue
        docs.append(
            {
                "id": f"{source}-{row_idx}-passage-{item_idx}",
                "title": "",
                "text": body,
                "is_gold": bool(selected[item_idx]) if item_idx < len(selected) else False,
                "metadata": {
                    "source_item_index": item_idx,
                    "url": urls[item_idx] if item_idx < len(urls) else None,
                },
            }
        )
    docs = cap_docs(docs, max_docs)
    if not docs:
        return None
    return {
        "id": f"{source}-{row.get('query_id', row.get('id', row_idx))}",
        "source": source,
        "task": "qa",
        "instruction": instruction,
        "question": question,
        "documents": docs,
        "answers": answers,
        "answer": answers[0],
        "metadata": {
            "raw_id": row.get("query_id", row.get("id")),
            "phrase_answers": phrase_answers,
            "well_formed_answers": well_formed_answers,
        },
    }


def dict_sequence_docs(
    value: Any,
    source: str,
    row_idx: int,
    item_prefix: str,
    body_keys: tuple[str, ...],
    answers: list[str],
) -> list[dict[str, Any]]:
    docs = []
    answer_lc = [answer.lower() for answer in answers if answer]
    for item_idx, record in enumerate(sequence_to_records(value)):
        body = ""
        for key in body_keys:
            body = clean(record.get(key, ""))
            if body:
                break
        title = clean(record.get("title", ""))
        if not body:
            continue
        joined = f"{title}\n{body}" if title else body
        docs.append(
            {
                "id": f"{source}-{row_idx}-{item_prefix}-{item_idx}",
                "title": title,
                "text": body,
                "is_gold": any(answer in joined.lower() for answer in answer_lc),
                "metadata": {
                    "source_item_index": item_idx,
                    "filename": record.get("filename"),
                    "url": record.get("url"),
                    "rank": record.get("rank"),
                },
            }
        )
    return docs


def convert_triviaqa_row(
    row: dict[str, Any],
    row_idx: int,
    source: str,
    instruction: str,
    max_docs: int | None,
) -> dict[str, Any] | None:
    question = clean(row.get("question", ""))
    answers = answer_aliases_from_value(row.get("answer", {}))
    if not question or not answers:
        return None
    docs = []
    docs.extend(dict_sequence_docs(row.get("entity_pages", []), source, row_idx, "entity", ("wiki_context", "description"), answers))
    docs.extend(dict_sequence_docs(row.get("search_results", []), source, row_idx, "search", ("search_context", "description"), answers))
    seen = set()
    deduped = []
    for doc in docs:
        key = (doc.get("title", ""), doc.get("text", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    docs = cap_docs(deduped, max_docs)
    if not docs:
        return None
    return {
        "id": f"{source}-{row.get('question_id', row_idx)}",
        "source": source,
        "task": "qa",
        "instruction": instruction,
        "question": question,
        "documents": docs,
        "answers": answers,
        "answer": answers[0],
        "metadata": {"raw_id": row.get("question_id")},
    }


def convert_qangaroo_row(
    row: dict[str, Any],
    row_idx: int,
    source: str,
    instruction: str,
    max_docs: int | None,
) -> dict[str, Any] | None:
    query = clean_text_value(row.get("query", row.get("question", "")))
    answer = clean(row.get("answer", ""))
    if not query or not answer:
        return None
    candidates = [clean(candidate) for candidate in row.get("candidates", []) if clean(candidate)]
    question = f"{query}\nCandidates: {', '.join(candidates)}" if candidates else query
    docs = context_items(row.get("supports", row.get("context", [])), source, row_idx, set(), max_docs)
    if not docs:
        return None
    answer_lc = answer.lower()
    for doc in docs:
        joined = f"{doc.get('title', '')}\n{doc.get('text', '')}".lower()
        doc["is_gold"] = bool(answer_lc and answer_lc in joined)
        doc.setdefault("metadata", {})["candidates"] = candidates
    return {
        "id": f"{source}-{row.get('id', row_idx)}",
        "source": source,
        "task": "qa",
        "instruction": instruction,
        "question": question,
        "documents": docs,
        "answers": [answer],
        "answer": answer,
        "metadata": {"raw_id": row.get("id"), "candidates": candidates},
    }


def musique_support_indices(row: dict[str, Any]) -> set[int]:
    indices = set()
    for step in sequence_to_records(row.get("question_decomposition", [])):
        idx = step.get("paragraph_support_idx")
        if isinstance(idx, int):
            indices.add(idx)
    return indices


def convert_musique_row(
    row: dict[str, Any],
    row_idx: int,
    source: str,
    instruction: str,
    max_docs: int | None,
) -> dict[str, Any] | None:
    question = clean(row.get("question", ""))
    answer = clean(row.get("answer", ""))
    if not question or not answer:
        return None
    support_indices = musique_support_indices(row)
    docs = []
    for item_idx, paragraph in enumerate(sequence_to_records(row.get("paragraphs", []))):
        body = clean(paragraph.get("paragraph_text", paragraph.get("text", "")))
        if not body:
            continue
        title = clean(paragraph.get("title", ""))
        paragraph_idx = paragraph.get("idx", item_idx)
        is_gold = bool(paragraph.get("is_supporting", False)) or (
            isinstance(paragraph_idx, int) and paragraph_idx in support_indices
        )
        docs.append(
            {
                "id": f"{source}-{row_idx}-paragraph-{item_idx}",
                "title": title,
                "text": body,
                "is_gold": is_gold,
                "metadata": {"source_item_index": paragraph_idx},
            }
        )
    docs = cap_docs(docs, max_docs)
    if not docs:
        return None
    aliases = [clean(alias) for alias in row.get("answer_aliases", []) if clean(alias)]
    answers = [answer, *aliases]
    return {
        "id": f"{source}-{row.get('id', row_idx)}",
        "source": source,
        "task": "qa",
        "instruction": instruction,
        "question": question,
        "documents": docs,
        "answers": answers,
        "answer": answer,
        "metadata": {"raw_id": row.get("id")},
    }


def convert_row(
    row: dict[str, Any],
    row_idx: int,
    source_cfg: dict[str, Any],
    instruction: str,
    max_docs: int | None,
) -> dict[str, Any] | None:
    source = str(source_cfg.get("name", source_cfg.get("hf_path", "source")))
    kind = str(source_cfg.get("kind", "context_qa"))
    if kind == "msmarco":
        return convert_msmarco_row(row, row_idx, source, instruction, max_docs)
    if kind == "triviaqa":
        return convert_triviaqa_row(row, row_idx, source, instruction, max_docs)
    if kind in {"qangaroo", "wikihop", "wiki_hop"}:
        return convert_qangaroo_row(row, row_idx, source, instruction, max_docs)
    if kind == "musique":
        return convert_musique_row(row, row_idx, source, instruction, max_docs)
    return convert_context_qa_row(row, row_idx, source, instruction, max_docs)


def open_hf_rows(source_cfg: dict[str, Any], split: str) -> Iterable[dict[str, Any]]:
    split_key = "train_split" if split == "train" else "eval_split"
    split_name = str(source_cfg.get(split_key, split))
    hf_path = source_cfg["hf_path"]
    hf_config = source_cfg.get(f"{split}_hf_config", source_cfg.get("hf_config"))
    streaming = bool(source_cfg.get("streaming", True))
    trust_remote_code = bool(source_cfg.get("trust_remote_code", False))
    LOGGER.info(
        "Loading HF dataset path=%s config=%s split=%s streaming=%s",
        hf_path,
        hf_config,
        split_name,
        streaming,
    )
    kwargs = {
        "split": split_name,
        "streaming": streaming,
        "trust_remote_code": trust_remote_code,
    }
    if hf_config:
        return load_dataset(hf_path, hf_config, **kwargs)
    return load_dataset(hf_path, **kwargs)


def load_source_examples(
    source_cfg: dict[str, Any],
    split: str,
    count: int | None,
    instruction: str = DEFAULT_INSTRUCTION,
    max_docs: int | None = None,
    min_docs: int = 2,
    seed: int = 42,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    rows = open_hf_rows(source_cfg, split)
    if bool(source_cfg.get("shuffle", split == "train")):
        if hasattr(rows, "shuffle"):
            try:
                rows = rows.shuffle(seed=seed, buffer_size=int(source_cfg.get("shuffle_buffer_size", 10_000)))  # type: ignore[attr-defined]
            except TypeError:
                rows = rows.shuffle(seed=seed)  # type: ignore[attr-defined]
    examples = []
    skipped = 0
    row_iter = tqdm(
        iter(rows),
        desc=f"{source_cfg.get('name', source_cfg.get('hf_path'))}:{split}",
        disable=not show_progress,
        dynamic_ncols=True,
        unit="row",
    )
    for row_idx, row in enumerate(row_iter):
        example = convert_row(dict(row), row_idx, source_cfg, instruction, max_docs)
        if example is None:
            skipped += 1
            if skipped > 100_000 and not examples:
                LOGGER.warning("Stopping after %d skipped rows for %s", skipped, source_cfg)
                break
            continue
        if len([doc for doc in example["documents"] if clean(doc.get("text"))]) < int(min_docs):
            skipped += 1
            continue
        examples.append(example)
        if count is not None and len(examples) >= int(count):
            break
    return examples


def load_examples_for_split(
    sources: list[dict[str, Any]],
    split: str,
    examples_per_source: int | None,
    instruction: str = DEFAULT_INSTRUCTION,
    max_docs: int | None = None,
    min_docs: int = 2,
    seed: int = 42,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    all_examples = []
    for source_idx, source_cfg in enumerate(sources):
        count = source_cfg.get(f"{split}_examples", source_cfg.get("examples", examples_per_source))
        source_examples = load_source_examples(
            source_cfg=source_cfg,
            split=split,
            count=None if count is None else int(count),
            instruction=instruction,
            max_docs=max_docs,
            min_docs=min_docs,
            seed=seed + source_idx,
            show_progress=show_progress,
        )
        all_examples.extend(source_examples)
    if split == "train":
        random.Random(seed).shuffle(all_examples)
    return all_examples


def open_text(path: str | Path, mode: str = "rt"):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding=None if "b" in mode else "utf-8")
    return path.open(mode, encoding=None if "b" in mode else "utf-8")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open_text(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_examples_from_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    examples = []
    for example in iter_jsonl(path):
        examples.append(example)
        if limit is not None and len(examples) >= int(limit):
            break
    return examples


def write_examples_jsonl(examples: Iterable[dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open_text(path, "wt") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    return count


def truncate_documents_with_tokenizer(
    example: dict[str, Any],
    tokenizer: Any,
    max_doc_tokens: int | None,
) -> dict[str, Any]:
    if max_doc_tokens is None or int(max_doc_tokens) <= 0:
        return example
    truncated = dict(example)
    docs = []
    for doc in example.get("documents", []):
        doc = dict(doc)
        text = clean(doc.get("text", ""))
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > int(max_doc_tokens):
            doc["text"] = tokenizer.decode(ids[: int(max_doc_tokens)], skip_special_tokens=True)
            metadata = dict(doc.get("metadata", {}))
            metadata["truncated_from_tokens"] = len(ids)
            doc["metadata"] = metadata
        docs.append(doc)
    truncated["documents"] = docs
    return truncated
