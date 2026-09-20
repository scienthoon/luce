"""
luce.convert — 학습용 JSONL 만들기 (`luce convert ...`).

1) 다운로드 없는 합성 스모크 데이터 (규칙 기반, LLM 없음):
    luce convert synthetic --out data --n 3000 --label-noise 0.05

2) HuggingFace 공개 데이터셋 프리셋:
    luce convert hf --preset klue_ynat --out data          # 한국어 뉴스 주제 (7지선다)
    luce convert hf --preset nsmc      --out data          # 한국어 영화평 감성 (noul)
    luce convert hf --preset boolq     --out data          # 영어 예/아니오 독해 (noul)
    luce convert hf --preset arc_easy  --out data          # 영어 과학 객관식 (4지선다)
    luce convert hf --preset klue_ynat --preset nsmc --preset arc_easy --out data   # 섞기

3) 내 CSV/JSONL (과거 결정 로그):
    luce convert csv --in tickets.csv --out data \
        --state-cols subject,body --type choice --question "Which queue should handle this?" \
        --label-col queue --options "billing=Payments and refunds,shipping=Delivery,technical=Bugs,general=Other"

출력: <out>/train.jsonl, <out>/val.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence

from .data import split_records, write_jsonl


# ---------------------------------------------------------------------------
# 1) 합성 스모크 데이터
# ---------------------------------------------------------------------------

_SYNTH_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "billing": [
        {"subject": "Charged twice for order #{order}", "body": "My card shows two charges of ${amount} for one order. {tail}", "urgency": 2},
        {"subject": "Refund not received", "body": "I returned the item {days} days ago and the refund for ${amount} is still missing. {tail}", "urgency": 1},
        {"subject": "Wrong amount on invoice", "body": "Invoice #{order} says ${amount} but the quote was lower. {tail}", "urgency": 1},
        {"subject": "구독 요금이 두 번 결제됐어요", "body": "이번 달 카드 명세서에 {amount}달러가 두 번 찍혀 있습니다. {tail}", "urgency": 2},
        {"subject": "환불이 아직 안 들어왔습니다", "body": "{days}일 전에 반품했는데 환불이 안 됐어요. {tail}", "urgency": 1},
    ],
    "shipping": [
        {"subject": "Package marked delivered but not here", "body": "Tracking for order #{order} says delivered {days} days ago. Nothing arrived. {tail}", "urgency": 2},
        {"subject": "Where is my order?", "body": "Order #{order} was placed {days} days ago and tracking has not updated. {tail}", "urgency": 1},
        {"subject": "Damaged on arrival", "body": "The box for order #{order} was crushed and the item inside is broken. {tail}", "urgency": 2},
        {"subject": "배송이 너무 늦어요", "body": "주문 #{order} 한 지 {days}일이 지났는데 아직 배송 준비중입니다. {tail}", "urgency": 1},
        {"subject": "배송 완료라는데 못 받았습니다", "body": "송장은 배송 완료인데 집에 아무것도 없어요. {tail}", "urgency": 2},
    ],
    "technical": [
        {"subject": "App crashes on login", "body": "Since the last update the app closes immediately after I enter my password. {tail}", "urgency": 2},
        {"subject": "Cannot reset password", "body": "The reset email for my account never arrives, checked spam too. {tail}", "urgency": 1},
        {"subject": "Export button does nothing", "body": "Clicking export on the reports page shows a spinner forever. {tail}", "urgency": 1},
        {"subject": "로그인이 안 됩니다", "body": "비밀번호를 맞게 입력해도 오류 코드 {order}가 뜹니다. {tail}", "urgency": 2},
        {"subject": "앱이 계속 꺼져요", "body": "업데이트 이후 사진을 열면 앱이 종료됩니다. {tail}", "urgency": 1},
    ],
    "general": [
        {"subject": "Question about your return policy", "body": "How many days do I have to return an unopened item? {tail}", "urgency": 0},
        {"subject": "Do you ship to Canada?", "body": "Planning to order as a gift, want to confirm shipping options first. {tail}", "urgency": 0},
        {"subject": "Feature suggestion", "body": "It would be nice to sort the dashboard by date. Not urgent. {tail}", "urgency": 0},
        {"subject": "영업시간 문의", "body": "고객센터 전화 상담은 몇 시까지인가요? {tail}", "urgency": 0},
        {"subject": "제품 사양 질문", "body": "이 모델이 해외 전압에서도 쓸 수 있는지 궁금합니다. {tail}", "urgency": 0},
    ],
}

_TAILS_CALM: List[str] = [
    "Thanks in advance.",
    "Let me know what you need from me.",
    "Appreciate any help.",
    "확인 부탁드립니다.",
    "답변 기다리겠습니다.",
    "",
]

_TAILS_ANGRY: List[str] = [
    "This is unacceptable and I want it fixed today.",
    "I have contacted you three times already. Ridiculous.",
    "If this is not resolved I am disputing the charge and leaving a review.",
    "정말 화가 납니다. 당장 처리해 주세요.",
    "이게 몇 번째인지 모르겠네요. 진짜 실망입니다.",
]

_QUEUE_OPTIONS: Dict[str, str] = {
    "billing": "Payments, refunds, duplicate charges",
    "shipping": "Delivery status, lost or damaged packages",
    "technical": "App or website bugs, login problems",
    "general": "Questions, feedback, anything else",
}

_PRIORITY_LEVELS: List[str] = ["Low", "Normal", "High", "Critical"]


def build_synthetic(n_states: int, label_noise: float, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    queues = list(_SYNTH_TEMPLATES.keys())
    records: List[Dict[str, Any]] = []
    for _ in range(n_states):
        queue = rng.choice(queues)
        template = rng.choice(_SYNTH_TEMPLATES[queue])
        angry = rng.random() < 0.3
        tail = rng.choice(_TAILS_ANGRY if angry else _TAILS_CALM)
        tier = rng.choice(["free", "standard", "gold", "enterprise"])
        channel = rng.choice(["email", "chat", "phone", "app"])
        fill = {
            "order": rng.randint(1000, 9999),
            "amount": rng.choice([19.99, 49.00, 89.99, 120.50, 300.00]),
            "days": rng.randint(1, 21),
            "tail": tail,
        }
        state = {
            "channel": channel,
            "customer_tier": tier,
            "subject": template["subject"].format(**fill),
            "body": template["body"].format(**fill).strip(),
        }

        # priority: 템플릿 기본 긴급도 + 화남 + 티어
        priority = int(template["urgency"])
        if angry:
            priority += 1
        if tier in ("gold", "enterprise") and priority >= 1:
            priority += 1
        priority = max(0, min(len(_PRIORITY_LEVELS) - 1, priority))

        queue_label = queue
        angry_label = angry
        if label_noise > 0.0:
            if rng.random() < label_noise:
                queue_label = rng.choice([q for q in queues if q != queue])
            if rng.random() < label_noise:
                priority = rng.randint(0, len(_PRIORITY_LEVELS) - 1)
            if rng.random() < label_noise:
                angry_label = not angry

        records.append({
            "state": state,
            "type": "choice",
            "question": "Which support queue should handle this ticket?",
            "options": dict(_QUEUE_OPTIONS),
            "label": queue_label,
        })
        records.append({
            "state": state,
            "type": "score",
            "question": "How should this ticket be prioritized?",
            "levels": list(_PRIORITY_LEVELS),
            "label": priority,
        })
        records.append({
            "state": state,
            "type": "noul",
            "question": "The customer sounds angry.",
            "label": angry_label,
        })
    return records


# ---------------------------------------------------------------------------
# 2) HuggingFace 프리셋
# ---------------------------------------------------------------------------

def _load_hf(name: str, config: Optional[str], split: str):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets 가 필요합니다")
    if config:
        return load_dataset(name, config, split=split)
    return load_dataset(name, split=split)


def _label_names(dataset, column: str, fallback: Sequence[str]) -> List[str]:
    try:
        return list(dataset.features[column].names)
    except Exception:
        return list(fallback)


def preset_klue_ynat(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    # 구 "klue" 는 스크립트 기반이라 datasets>=3 에서 로드가 거부된다. parquet 로 옮겨진 "klue/klue" 를 쓴다.
    dataset = _load_hf("klue/klue", "ynat", split)
    names = _label_names(dataset, "label", ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"])
    options = {name: name for name in names}
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        records.append({
            "state": row["title"],
            "type": "choice",
            "question": "이 뉴스 제목의 주제 분류는 무엇인가?",
            "options": options,
            "label": names[int(row["label"])],
            "source": "klue_ynat",
        })
    return records


_NSMC_URLS = {
    "train": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
    "test": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
}


def _nsmc_rows(split: str):
    """
    NSMC 는 HF 허브에서 스크립트 기반이라 datasets>=3 에서 로드되지 않는다.
    원본 GitHub 의 TSV (id, document, label) 를 직접 받아 ~/.cache/luce/nsmc/ 에 캐시한다.
    """
    import urllib.request
    url = _NSMC_URLS[split]
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "luce", "nsmc")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(path):
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            yield dict(zip(header, parts))


def preset_nsmc(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(_nsmc_rows(split)):
        if limit is not None and i >= limit:
            break
        document = row["document"].strip()
        if not document:
            continue
        records.append({
            "state": document,
            "type": "noul",
            "question": "이 영화 리뷰는 긍정적이다.",
            "label": bool(int(row["label"]) == 1),
            "source": "nsmc",
        })
    return records


def preset_boolq(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    dataset = _load_hf("google/boolq", None, split)
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        records.append({
            "state": row["passage"],
            "type": "noul",
            "question": row["question"].rstrip("?") + "?",
            "label": bool(row["answer"]),
            "source": "boolq",
        })
    return records


def preset_arc_easy(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    dataset = _load_hf("allenai/ai2_arc", "ARC-Easy", split)
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        options = {label: text for label, text in zip(labels, texts)}
        answer = str(row["answerKey"])
        if answer not in options:
            continue
        records.append({
            "state": row["question"],
            "type": "choice",
            "question": "Which option is the correct answer?",
            "options": options,
            "label": answer,
            "source": "arc_easy",
        })
    return records


_BANKING77_URLS = {
    "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}


def _banking77_rows(split: str) -> List[Dict[str, str]]:
    """
    HF 의 PolyAI/banking77 은 스크립트 기반이라 datasets>=3 에서 로드되지 않는다.
    원본 GitHub CSV (text, category) 를 받아 ~/.cache/luce/banking77/ 에 캐시한다. CC BY 4.0.
    """
    import urllib.request
    url = _BANKING77_URLS[split]
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "luce", "banking77")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{split}.csv")
    if not os.path.exists(path):
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def preset_banking77(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    """온라인 뱅킹 문의 77개 의도 (CC BY 4.0). 닫힌 라벨 집합 분류, 선택지 77개."""
    rows = _banking77_rows(split)
    # 라벨 집합은 train 기준으로 고정 (test 도 같은 77개). 정렬해서 순서를 결정적으로.
    names = sorted({row["category"] for row in _banking77_rows("train")})
    if len(names) != 77:
        raise SystemExit(f"banking77: expected 77 categories, found {len(names)}")
    options = {name: name.replace("_", " ") for name in names}
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        if row["category"] not in options:
            continue
        records.append({
            "state": row["text"],
            "type": "choice",
            "question": "Which banking intent does this customer message express?",
            "options": options,
            "label": row["category"],
            "source": "banking77",
        })
    return records


def preset_commonsense_qa(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    """5지선다 상식 (MIT). held-out 용."""
    dataset = _load_hf("tau/commonsense_qa", None, split)
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        options = {label: text for label, text in zip(labels, texts)}
        answer = str(row["answerKey"])
        if answer not in options:
            continue
        records.append({
            "state": row["question"],
            "type": "choice",
            "question": "Which option is the correct answer?",
            "options": options,
            "label": answer,
            "source": "commonsense_qa",
        })
    return records


def preset_hellaswag(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    """4지선다, 선택지가 긴 문장 (MIT). held-out 용."""
    dataset = _load_hf("Rowan/hellaswag", None, split)
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        label = str(row["label"]).strip()
        if not label.isdigit():
            continue
        endings = list(row["endings"])
        if not (0 <= int(label) < len(endings)):
            continue
        options = {str(k): ending for k, ending in enumerate(endings)}
        records.append({
            "state": row["ctx"],
            "type": "choice",
            "question": "Which ending most plausibly continues the text?",
            "options": options,
            "label": label,
            "source": "hellaswag",
        })
    return records


def preset_openbookqa(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    """4지선다 과학 (Apache 2.0). ARC 와 가까운 held-out 대조군."""
    dataset = _load_hf("allenai/openbookqa", "main", split)
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        options = {label: text for label, text in zip(labels, texts)}
        answer = str(row["answerKey"])
        if answer not in options:
            continue
        records.append({
            "state": row["question_stem"],
            "type": "choice",
            "question": "Which option is the correct answer?",
            "options": options,
            "label": answer,
            "source": "openbookqa",
        })
    return records


_PRESETS = {
    "klue_ynat": {"fn": preset_klue_ynat, "train": "train", "val": "validation"},
    "nsmc": {"fn": preset_nsmc, "train": "train", "val": "test"},
    "boolq": {"fn": preset_boolq, "train": "train", "val": "validation"},
    "arc_easy": {"fn": preset_arc_easy, "train": "train", "val": "validation"},
    "banking77": {"fn": preset_banking77, "train": "train", "val": "test"},
    "commonsense_qa": {"fn": preset_commonsense_qa, "train": "train", "val": "validation"},
    "hellaswag": {"fn": preset_hellaswag, "train": "train", "val": "validation"},
    "openbookqa": {"fn": preset_openbookqa, "train": "train", "val": "validation"},
}


# ---------------------------------------------------------------------------
# 3) CSV / JSONL 매핑
# ---------------------------------------------------------------------------

def _parse_options(spec: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, desc = part.split("=", 1)
            options[key.strip()] = desc.strip()
        else:
            options[part] = part
    if len(options) < 2:
        raise SystemExit("--options 는 최소 2개 필요 (예: a=desc a,b=desc b)")
    return options


def _read_table(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_from_table(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _read_table(args.input)
    state_cols = [c.strip() for c in args.state_cols.split(",") if c.strip()]
    records: List[Dict[str, Any]] = []
    options = _parse_options(args.options) if args.type == "choice" else None
    levels = [x.strip() for x in args.levels.split(",")] if args.type == "score" else None

    for row in rows:
        if len(state_cols) == 1:
            state: Any = row[state_cols[0]]
        else:
            state = {col: row[col] for col in state_cols}
        raw_label = row[args.label_col]
        record: Dict[str, Any] = {"state": state, "type": args.type, "question": args.question}
        if args.type == "choice":
            if str(raw_label) not in options:  # type: ignore[operator]
                continue
            record["options"] = options
            record["label"] = str(raw_label)
        elif args.type == "score":
            if str(raw_label).isdigit():
                index = int(raw_label)
            else:
                if str(raw_label) not in levels:  # type: ignore[operator]
                    continue
                index = levels.index(str(raw_label))  # type: ignore[union-attr]
            if not (0 <= index < len(levels)):  # type: ignore[arg-type]
                continue
            record["levels"] = levels
            record["label"] = index
        elif args.type == "noul":
            value = str(raw_label).strip().lower()
            if value in ("1", "true", "yes", "y", "t"):
                record["label"] = True
            elif value in ("0", "false", "no", "n", "f"):
                record["label"] = False
            else:
                continue
        records.append(record)
    if not records:
        raise SystemExit("변환된 레코드가 없습니다. 컬럼 이름과 라벨 값을 확인하세요.")
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _subsample(rows: List[Dict[str, Any]], limit: Optional[int], seed: int, name: str) -> List[Dict[str, Any]]:
    """limit 개를 seed 로 균등 무작위 추출 (limit 이 None 이거나 전체보다 크면 그대로)."""
    if limit is None or limit >= len(rows):
        return rows
    picked = random.Random(seed).sample(rows, limit)
    labels = {str(r.get("label")) for r in rows}
    picked_labels = {str(r.get("label")) for r in picked}
    print(f"  {name}: {len(rows)} -> {limit} random (seed {seed}); labels {len(picked_labels)}/{len(labels)}")
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description="build luce JSONL datasets")
    sub = parser.add_subparsers(dest="command", required=True)

    p_syn = sub.add_parser("synthetic", help="규칙 기반 합성 스모크 데이터")
    p_syn.add_argument("--out", required=True)
    p_syn.add_argument("--n", type=int, default=3000, help="state 개수 (레코드는 x3)")
    p_syn.add_argument("--label-noise", type=float, default=0.05)
    p_syn.add_argument("--val-fraction", type=float, default=0.1)
    p_syn.add_argument("--seed", type=int, default=0)

    p_hf = sub.add_parser("hf", help="HuggingFace 프리셋")
    p_hf.add_argument("--preset", action="append", required=True, choices=sorted(_PRESETS.keys()))
    p_hf.add_argument("--out", required=True)
    p_hf.add_argument("--limit", type=int, default=None, help="프리셋당 train 최대 개수")
    p_hf.add_argument("--val-limit", type=int, default=2000, help="프리셋당 val 최대 개수")
    p_hf.add_argument("--seed", type=int, default=0)

    p_csv = sub.add_parser("csv", help="CSV/JSONL 매핑")
    p_csv.add_argument("--in", dest="input", required=True)
    p_csv.add_argument("--out", required=True)
    p_csv.add_argument("--state-cols", required=True, help="쉼표 구분. 1개면 문자열 state, 여러 개면 객체 state")
    p_csv.add_argument("--type", required=True, choices=["choice", "score", "noul"])
    p_csv.add_argument("--question", required=True)
    p_csv.add_argument("--label-col", required=True)
    p_csv.add_argument("--options", default="", help="choice: key=desc,key=desc")
    p_csv.add_argument("--levels", default="", help="score: 낮은순,높은순")
    p_csv.add_argument("--val-fraction", type=float, default=0.1)
    p_csv.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.command == "synthetic":
        records = build_synthetic(args.n, args.label_noise, args.seed)
        # 같은 state 의 3개 레코드가 train/val 에 갈리지 않도록 state 단위로 나눈다.
        groups = [records[i:i + 3] for i in range(0, len(records), 3)]
        train_groups, val_groups = split_records(groups, args.val_fraction, seed=args.seed)  # type: ignore[arg-type]
        train_records = [r for g in train_groups for r in g]
        val_records = [r for g in val_groups for r in g]
    elif args.command == "hf":
        train_records = []
        val_records = []
        for preset in args.preset:
            spec = _PRESETS[preset]
            print(f"loading {preset} ...")
            # 전체를 읽은 뒤 seed 로 무작위 표본. (예전엔 앞 N 행을 잘랐는데 banking77 CSV 처럼 라벨별로
            # 정렬된 소스에서는 2,000 행이 77개 의도 중 17개만 담는 문제가 있었다.)
            train_records.extend(_subsample(spec["fn"](spec["train"], None), args.limit, args.seed, f"{preset}/train"))
            val_records.extend(_subsample(spec["fn"](spec["val"], None), args.val_limit, args.seed + 1, f"{preset}/val"))
        random.Random(args.seed).shuffle(train_records)
        random.Random(args.seed + 1).shuffle(val_records)
    else:
        records = build_from_table(args)
        train_records, val_records = split_records(records, args.val_fraction, seed=args.seed)

    train_path = os.path.join(args.out, "train.jsonl")
    val_path = os.path.join(args.out, "val.jsonl")
    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)
    print(f"wrote {len(train_records)} -> {train_path}")
    print(f"wrote {len(val_records)} -> {val_path}")


if __name__ == "__main__":
    main()
