"""
ViTC-S and ViTC-L Benchmark Evaluator for Ollama Models
Based on ArtPrompt (ACL 2024): https://github.com/uw-nsl/ArtPrompt

QUICK START (Windows):
----------------------
Step 1: Install dependencies
    pip install requests tqdm

Step 2: Make sure Ollama is running
    ollama serve   (or it starts automatically in the system tray)

Step 3: Pull a model if you haven't already
    ollama pull llama3
    ollama pull mistral
    ollama pull gemma

Step 4: Run evaluation
    # ViTC-S on llama3
    python vitc_eval.py --model llama3 --task s --data_dir C:/path/to/ArtPrompt

    # ViTC-L on llama3
    python vitc_eval.py --model llama3 --task l --data_dir C:/path/to/ArtPrompt

    # Both tasks on multiple models
    python vitc_eval.py --model llama3 mistral --task s l --data_dir C:/path/to/ArtPrompt

    # Quick smoke test with only 50 samples
    python vitc_eval.py --model llama3 --task s --data_dir C:/path/to/ArtPrompt --num 50

    # Resume a crashed/interrupted run (skips already-done samples)
    python vitc_eval.py --model llama3 --task s --data_dir C:/path/to/ArtPrompt --resume

DATASET LOCATION:
-----------------
The script looks for dataset files in this order:
  <data_dir>/dataset/vitc-s.json
  <data_dir>/vitc-s.json
  <data_dir>/data/vitc-s.json
  ./vitc-s.json  (current directory)

RESULTS:
--------
  vitc_results/<model>_vitc-s.json       <- per-sample results
  vitc_results/<model>_vitc-s.log        <- run log for debugging
  vitc_results/combined_summary.json     <- all runs summary
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────

OLLAMA_HOST   = "http://localhost:11434"
RESULTS_DIR   = "./vitc_results"

# Columns per character — fixed at 20 per ArtPrompt paper
COLS_PER_CHAR = 20

# Prompt format from ArtPrompt paper Appendix (ps=default setting)
PROMPT_TEMPLATE = (
    "The following string is made of ascii art of {n_chars} letter(s), "
    "each row delimited by a '*' symbol. "
    "Each letter is in a grid with {n_rows} rows and {n_cols} columns. "
    "Identify what the letter(s) are.\n"
    "You should be confident and answer with only the letter(s), and nothing else.\n\n"
    "{art_text}"
)

# Paper baselines for reference in the final summary table
PAPER_BASELINES = [
    ("GPT-4",       "vitc-s", 25.19),
    ("GPT-4",       "vitc-l",  3.26),
    ("GPT-3.5",     "vitc-s",  2.41),
    ("GPT-3.5",     "vitc-l",  0.54),
    ("Gemini Pro",  "vitc-s",  5.67),
    ("Llama2-13B",  "vitc-s",  0.00),
]


# ── Logging Setup ──────────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    """Create a logger that writes to a log file (DEBUG level)."""
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    return logger


# ── Dataset Loading ────────────────────────────────────────────────────────────

def load_dataset(data_dir: str, task: str) -> list[dict]:
    """
    Load ViTC-S or ViTC-L from the ArtPrompt repo.

    Handles all known key-name variants across repo versions:
      label keys : "word", "text", "label"
      art keys   : "art", "art_text", "ascii"
    """
    filename = "vitc-s.json" if task == "s" else "vitc-l.json"

    candidates = [
        Path(data_dir) / "dataset" / filename,
        Path(data_dir) / filename,
        Path(data_dir) / "data" / filename,
        Path(filename),
    ]

    dataset_path = next((p for p in candidates if p.exists()), None)

    if dataset_path is None:
        print(f"\n[ERROR] Could not find {filename} in any of these locations:")
        for p in candidates:
            print(f"  {p}")
        print(
            "\nClone ArtPrompt and pass --data_dir:\n"
            "  git clone https://github.com/uw-nsl/ArtPrompt.git\n"
            "  python vitc_eval.py --data_dir ./ArtPrompt ...\n"
        )
        sys.exit(1)

    print(f"[dataset] Loading ViTC-{task.upper()} from: {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Handle both list-of-entries and dict-wrapped formats
    if isinstance(raw, dict):
        raw = raw.get("data", list(raw.values())[0])

    normalized = []
    skipped = 0
    for entry in raw:
        label = (
            entry.get("word") or entry.get("text") or entry.get("label") or ""
        ).strip().upper()

        art = (
            entry.get("art") or entry.get("art_text") or entry.get("ascii") or ""
        )

        if label and art:
            normalized.append({"label": label, "art": art})
        else:
            skipped += 1

    if skipped:
        print(f"[dataset] Warning: skipped {skipped} entries with missing label or art.")

    print(f"[dataset] {len(normalized)} samples ready.\n")
    return normalized


# ── Art Metadata ───────────────────────────────────────────────────────────────

def get_art_metadata(art: str, label: str) -> tuple[int, int, int]:
    """
    Compute (n_chars, n_rows, n_cols) for the prompt template.

    n_chars : number of characters = len(label)
    n_rows  : number of '*'-delimited row segments in the art
    n_cols  : COLS_PER_CHAR * n_chars  (total grid width, per paper convention)

    FIX vs previous version: n_cols is now n_chars * 20, not just 20.
    A 3-char label should report 60 columns, not 20.
    """
    n_chars = max(1, len(label))
    row_segments = [r for r in art.split("*") if r.strip()]
    n_rows = len(row_segments) if row_segments else 11
    n_cols = COLS_PER_CHAR * n_chars   # <-- fixed
    return n_chars, n_rows, n_cols


# ── Prediction Extraction ──────────────────────────────────────────────────────

def extract_prediction(response: str, expected_len: int) -> str:
    """
    Extract the predicted character(s) from the model's raw text response.

    Strategy (in order of confidence):
      1. Strip preamble phrases -> if remaining alnum is exact length -> use it
      2. Look for a quoted token of exactly expected_len
      3. First STANDALONE alnum token of exact length
         (standalone = not part of a longer word, prevents matching
          "A" inside "ANSWER" when expected_len=1)
      4. Fallback: first alnum chars found, truncated to expected_len
    """
    if not response or response.startswith("["):
        # Timeout or error sentinel — return empty
        return ""

    resp = response.strip()

    # 1. Strip common preamble phrases
    resp = re.sub(
        r"(?i)^(the\s+)?(letter[s]?|character[s]?|answer|word|string)\s*(is|are)\s*[:\-]?\s*",
        "", resp
    ).strip()
    resp = re.sub(r'^["\'\s]+|["\'\s.!?]+$', "", resp)
    alnum = re.sub(r"[^A-Za-z0-9]", "", resp)
    if len(alnum) == expected_len:
        return alnum.upper()

    # 2. Look for a quoted token of exact length
    for quoted in re.findall(r'["\']([A-Za-z0-9]+)["\']', response):
        if len(quoted) == expected_len:
            return quoted.upper()

    # 3. First standalone token of exact length
    pattern = (
        r"(?<![A-Za-z0-9])"
        r"([A-Za-z0-9]{" + str(expected_len) + r"})"
        r"(?![A-Za-z0-9])"
    )
    matches = re.findall(pattern, response)
    if matches:
        return matches[0].upper()

    # 4. Fallback
    full_alnum = re.sub(r"[^A-Za-z0-9]", "", response)
    return full_alnum.upper()[:expected_len] if full_alnum else ""


# ── Scoring ────────────────────────────────────────────────────────────────────

def acc_score(pred: str, label: str) -> float:
    """Exact match accuracy (1.0 or 0.0)."""
    return 1.0 if pred.upper() == label.upper() else 0.0


def amr_score(pred: str, label: str) -> float:
    """
    Average Match Ratio: fraction of characters matching at each position.
    Provides partial credit for multi-character ViTC-L sequences.
    """
    if not label:
        return 0.0
    label = label.upper()
    pred  = pred.upper().ljust(len(label))[:len(label)]
    return sum(p == l for p, l in zip(pred, label)) / len(label)


# ── Ollama Client ──────────────────────────────────────────────────────────────

class OllamaClient:
    def __init__(self, host: str, model: str, timeout: int = 120):
        self.host    = host.rstrip("/")
        self.model   = model
        self.timeout = timeout
        self._check_connection()

    def _check_connection(self):
        """Verify Ollama is reachable and warn if model is not pulled."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            print(
                f"\n[ERROR] Cannot connect to Ollama at {self.host}\n"
                "  -> Make sure Ollama is running:\n"
                "     Windows: check system tray, or run 'ollama serve'\n"
            )
            sys.exit(1)
        except requests.exceptions.Timeout:
            print(
                f"\n[ERROR] Ollama health check timed out at {self.host}\n"
                "  -> Ollama may be starting up. Wait a moment and retry.\n"
            )
            sys.exit(1)
        except requests.exceptions.HTTPError as e:
            print(f"\n[ERROR] Ollama returned HTTP error: {e}\n")
            sys.exit(1)

        available = [m["name"] for m in resp.json().get("models", [])]
        base_name = self.model.split(":")[0].lower()
        if base_name not in [m.split(":")[0].lower() for m in available]:
            print(f"\n[WARNING] '{self.model}' not found in Ollama.")
            print(f"  Available: {available}")
            print(f"  Pull with: ollama pull {self.model}\n")

    def query(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the raw text response."""
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 32,
                "top_p":       1.0,
                "seed":        42,
            }
        }
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            return "[TIMEOUT]"
        except requests.exceptions.HTTPError as e:
            return f"[HTTP_ERROR: {e}]"
        except Exception as e:
            return f"[ERROR: {e}]"


# ── Resume Support ─────────────────────────────────────────────────────────────

def load_existing_results(out_path: Path) -> list[dict]:
    """
    Load already-completed samples from a previous interrupted run.
    Returns empty list if file doesn't exist or is malformed.
    """
    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                data = json.load(f)
            existing = data.get("samples", [])
            if existing:
                print(f"[resume] Found {len(existing)} completed samples in {out_path}")
            return existing
        except (json.JSONDecodeError, KeyError):
            print(f"[resume] Could not read {out_path}, starting fresh.")
    return []


# ── Incremental Save ───────────────────────────────────────────────────────────

def save_results(
    path:      Path,
    model:     str,
    task_name: str,
    samples:   list[dict],
    acc:       float,
    amr:       float,
):
    """
    Write results JSON atomically via temp file + rename.
    Prevents corrupt JSON if the process is killed mid-write.
    """
    payload = {
        "model":     model,
        "task":      task_name.lower(),
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(samples),
        "acc_pct":   acc,
        "amr_pct":   amr,
        "samples":   samples,
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ── Evaluation Loop ────────────────────────────────────────────────────────────

def run_evaluation(
    model:    str,
    task:     str,
    data_dir: str,
    host:     str,
    num:      Optional[int],
    save_dir: str,
    delay:    float,
    resume:   bool,
) -> dict:
    """
    Run full ViTC-S or ViTC-L evaluation for one model.
    Saves incrementally every 50 samples so crashes don't lose progress.
    """
    task_name  = f"ViTC-{task.upper()}"
    safe_model = re.sub(r"[^\w\-]", "_", model)

    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / f"{safe_model}_{task_name.lower()}.json"
    log_path = Path(save_dir) / f"{safe_model}_{task_name.lower()}.log"
    logger   = setup_logger(log_path)

    logger.info(f"Starting: model={model} task={task_name}")

    # Load dataset
    dataset = load_dataset(data_dir, task)
    if num is not None:
        dataset = dataset[:num]
        print(f"[eval] Limiting to first {num} samples.\n")

    # Resume: pick up where we left off
    completed = load_existing_results(out_path) if resume else []
    start_idx = len(completed)
    samples   = list(completed)

    if start_idx > 0:
        print(f"[resume] Resuming from sample {start_idx} / {len(dataset)}\n")

    remaining = dataset[start_idx:]

    # Recompute running totals from any completed samples
    total_acc = sum(s["acc"] for s in samples)
    total_amr = sum(s["amr"] for s in samples)

    client = OllamaClient(host=host, model=model)

    print("=" * 62)
    print(f"  Model  : {model}")
    print(f"  Task   : {task_name}")
    print(f"  Samples: {len(dataset)}  (remaining: {len(remaining)})")
    print(f"  Host   : {host}")
    print(f"  Log    : {log_path}")
    print("=" * 62)

    for i, entry in enumerate(
        tqdm(remaining, desc=f"{model} | {task_name}"),
        start=start_idx
    ):
        label = entry["label"]
        art   = entry["art"]

        n_chars, n_rows, n_cols = get_art_metadata(art, label)
        prompt = PROMPT_TEMPLATE.format(
            n_chars=n_chars,
            n_rows=n_rows,
            n_cols=n_cols,
            art_text=art,
        )

        raw_response = client.query(prompt)
        prediction   = extract_prediction(raw_response, len(label))
        acc          = acc_score(prediction, label)
        amr          = amr_score(prediction, label)

        total_acc += acc
        total_amr += amr

        samples.append({
            "idx":          i,
            "label":        label,
            "prediction":   prediction,
            "raw_response": raw_response,
            "acc":          acc,
            "amr":          round(amr, 4),
        })

        logger.debug(
            f"[{i}] label={label!r} pred={prediction!r} "
            f"acc={acc} amr={amr:.4f} | raw={raw_response!r}"
        )

        # Save every 50 samples
        if (i + 1) % 50 == 0 or (i + 1 - start_idx) == len(remaining):
            n_done   = len(samples)
            curr_acc = round(total_acc / n_done * 100, 2)
            curr_amr = round(total_amr / n_done * 100, 2)
            save_results(out_path, model, task_name, samples, curr_acc, curr_amr)

        if delay > 0:
            time.sleep(delay)

    n         = len(samples)
    final_acc = round(total_acc / n * 100, 2) if n else 0.0
    final_amr = round(total_amr / n * 100, 2) if n else 0.0

    save_results(out_path, model, task_name, samples, final_acc, final_amr)

    baseline = "25.19" if task == "s" else "3.26"
    print(f"\n  Done: {task_name} [{model}]")
    print(f"    Acc : {final_acc:.2f}%   (GPT-4 baseline: {baseline}%)")
    print(f"    AMR : {final_amr:.2f}%")
    print(f"    Saved : {out_path}")
    print(f"    Log   : {log_path}\n")

    logger.info(f"Done. Acc={final_acc}% AMR={final_amr}%")

    return {
        "model":     model,
        "task":      task_name.lower(),
        "timestamp": datetime.now().isoformat(),
        "n_samples": n,
        "acc_pct":   final_acc,
        "amr_pct":   final_amr,
        "samples":   samples,
    }


# ── Summary Table ──────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict]):
    """Print comparison table of all runs alongside paper baselines."""
    print("\n" + "=" * 68)
    print("  EVALUATION SUMMARY")
    print("=" * 68)
    print(f"  {'Model':<24} {'Task':<10} {'Acc %':>8}  {'AMR %':>8}  {'N':>6}")
    print("-" * 68)

    for r in all_results:
        print(
            f"  {r['model']:<24} {r['task']:<10} "
            f"{r['acc_pct']:>8.2f}  {r['amr_pct']:>8.2f}  {r['n_samples']:>6}"
        )

    print("-" * 68)
    print("  PAPER BASELINES:")
    for name, task, acc in PAPER_BASELINES:
        print(f"  {name:<24} {task:<10} {acc:>8.2f}")
    print("=" * 68)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Ollama models on ViTC-S / ViTC-L (ArtPrompt benchmark)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test — 50 samples, ViTC-S
  python vitc_eval.py --model llama3 --task s --data_dir ./ArtPrompt --num 50

  # Full run — both tasks, multiple models
  python vitc_eval.py --model llama3 mistral gemma --task s l --data_dir ./ArtPrompt

  # Resume an interrupted run
  python vitc_eval.py --model llama3 --task s --data_dir ./ArtPrompt --resume
        """
    )
    parser.add_argument(
        "--model", nargs="+", required=True, metavar="MODEL",
        help="Ollama model name(s)  e.g. llama3  mistral  gemma  phi3"
    )
    parser.add_argument(
        "--task", nargs="+", default=["s"],
        choices=["s", "l", "S", "L"], metavar="TASK",
        help="Task(s): s=ViTC-S  l=ViTC-L  (default: s)"
    )
    parser.add_argument(
        "--data_dir", required=True, metavar="PATH",
        help="Path to ArtPrompt repo root (contains dataset/vitc-s.json etc.)"
    )
    parser.add_argument(
        "--host", default=OLLAMA_HOST, metavar="URL",
        help=f"Ollama API URL  (default: {OLLAMA_HOST})"
    )
    parser.add_argument(
        "--num", type=int, default=None, metavar="N",
        help="Limit samples per task  (default: full dataset)"
    )
    parser.add_argument(
        "--save_dir", default=RESULTS_DIR, metavar="PATH",
        help=f"Where to save results  (default: {RESULTS_DIR})"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, metavar="SEC",
        help="Seconds between requests  (default: 0)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume a previously interrupted run"
    )

    args = parser.parse_args()
    all_results = []

    for model in args.model:
        for task in [t.lower() for t in args.task]:
            result = run_evaluation(
                model=model,
                task=task,
                data_dir=args.data_dir,
                host=args.host,
                num=args.num,
                save_dir=args.save_dir,
                delay=args.delay,
                resume=args.resume,
            )
            all_results.append(result)

    if all_results:
        print_summary(all_results)

        combined      = [{k: v for k, v in r.items() if k != "samples"} for r in all_results]
        combined_path = Path(args.save_dir) / "combined_summary.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"\n  [saved] Combined summary -> {combined_path}")


if __name__ == "__main__":
    main()
