import os, json, time
from datetime import datetime
import numpy as np

def _to_jsonable(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.ndarray,)): return x.tolist()
    return x

def append_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_to_jsonable, ensure_ascii=False) + "\n")

def log_cv_run(jsonl_path: str, params: dict, cv_results: dict, extra: dict | None = None):
    """
    params: 你這次 run 的超參數/設定
    cv_results: cross_validation_mil 回傳的 dict
    """
    overall = cv_results.get("overall_metrics", {}) or {}
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "run_id": f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{int(time.time()*1000)%100000}',
        **{f"p_{k}": v for k, v in params.items()},
        **{f"m_{k}": v for k, v in overall.items()},
        "cv": cv_results.get("cv"),
        "k": cv_results.get("k"),
    }
    # 讓 label_map / class_counts 這些也可追溯（重要！）
    if "label_map" in cv_results:
        record["label_map"] = cv_results["label_map"]
    if extra:
        record.update(extra)

    append_jsonl(jsonl_path, record)
    return record