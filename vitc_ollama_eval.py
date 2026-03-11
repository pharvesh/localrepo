"""
ViTC-S and ViTC-L Benchmark Evaluator for Ollama Models
Based on ArtPrompt paper (ACL 2024): https://github.com/uw-nsl/ArtPrompt

Usage:
    # Install deps
    pip install requests tqdm art

    # Clone dataset first:
    # git clone https://github.com/uw-nsl/ArtPrompt.git
    # Then set ARTPROMPT_DIR below or pass --data_dir

    # Run ViTC-S on llama3
    python vitc_ollama_eval.py --model llama3 --task s

    # Run ViTC-L on multiple models
    python vitc_ollama_eval.py --model llama3 mistral gemma --task l

    # Run both tasks, limit to 100 samples per task
    python vitc_ollama_eval.py --model llama3 --task s l --num 100

    # Use a custom Ollama host
    python vitc_ollama_eval.py --model llama3 --task s --host http://localhost:11434
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_DATA_DIR = "./ArtPrompt"          # root of cloned ArtPrompt repo
RESULTS_DIR = "./vitc_results"

# Prompt template matching the paper's setup (ps=default)
PROMPT_TEMPLATE = """\
The following string is made of ascii art of {n_chars} letter(s), each row delimited by a '*' symbol.
Each letter is in a grid with {n_rows} rows and {n_cols} columns.
Identify what the letter(s) are.
You should be confident and answer with only the letter(s), and nothing else.

{art_text}"""


# ── Dataset Loader ─────────────────────────────────────────────────────────────

def load_vitc_dataset(data_dir: str, task: str) -> list[dict]:
    """
    Load ViTC-S or ViTC-L dataset from ArtPrompt repo.
    
    Dataset JSON entries contain:
      - art_text : the ASCII art string (rows separated by '*')
      - text     : ground truth label (single char for -S, 2-4 chars for -L)
    """
    task_map = {"s": "vitc-s", "l": "vitc-l"}
    task_name = task_map[task.lower()]
    
    # Try common locations within the repo
    candidate_paths = [
        Path(data_dir) / "dataset" / f"{task_name}.json",
        Path(data_dir) / f"{task_name}.json",
        Path(data_dir) / "data" / f"{task_name}.json",
    ]
    
    for p in candidate_paths:
        if p.exists():
            print(f"[dataset] Loading {task_name} from {p}")
            with open(p) as f:
                data = json.load(f)
            # Normalise: handle both list and dict-wrapped formats
            if isinstance(data, dict):
                data = data.get("data", list(data.values())[0])
            print(f"[dataset] Loaded {len(data)} samples")
            return data
    
    raise FileNotFoundError(
        f"Could not find {task_name}.json in {data_dir}.\n"
        f"Please clone the ArtPrompt repo:\n"
        f"  git clone https://github.com/uw-nsl/ArtPrompt.git\n"
        f"Then pass --data_dir ArtPrompt"
    )


def parse_art_metadata(art_text: str) -> tuple[int, int, int]:
    """
    Infer n_chars, n_rows, n_cols from the ASCII art block.
    Rows are '*'-delimited; columns are max width per row segment.
    """
    rows = art_text.split("*")
    rows = [r for r in rows if r.strip()]   # drop empty splits
    
    n_rows = len(rows)
    n_cols = max((len(r) for r in rows), default=20)
    
    # n_chars: each char block typically has consistent column width
    # Heuristic: total_cols / per_char_cols; fallback to 1
    # The paper uses 20 cols/char for ViTC-S, variable for ViTC-L
    per_char_cols = 20
    n_chars = max(1, round(n_cols / per_char_cols))
    
    return n_chars, n_rows, per_char_cols


# ── Scoring ────────────────────────────────────────────────────────────────────

def extract_prediction(response: str, n_chars: int) -> str:
    """
    Extract the model's predicted character(s) from its response.
    
    Handles:
      - Clean single/multi char answers: "A", "AB", "A3"
      - Answers wrapped in quotes or punctuation: '"A"', 'The letter is A.'
      - Extra explanation after the answer
    """
    # Strip whitespace
    resp = response.strip()
    
    # Pattern 1: Response is already a clean alpha/digit string of right length
    clean = re.sub(r"[^A-Za-z0-9]", "", resp)
    if len(clean) == n_chars:
        return clean.upper()
    
    # Pattern 2: Look for quoted answer first
    quoted = re.findall(r'["\']([A-Za-z0-9]+)["\']', resp)
    if quoted:
        return quoted[0].upper()
    
    # Pattern 3: "The letter(s) is/are X" pattern
    match = re.search(
        r'(?:letter[s]?\s+(?:is|are)|answer\s+is|character[s]?\s+(?:is|are))\s*[:\-]?\s*([A-Za-z0-9]+)',
        resp, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()
    
    # Pattern 4: Just take first contiguous alphanum token of right length
    tokens = re.findall(r'[A-Za-z0-9]+', resp)
    for tok in tokens:
        if len(tok) == n_chars:
            return tok.upper()
    
    # Fallback: return first token truncated/padded
    return (tokens[0].upper() if tokens else "")[:n_chars]


def compute_acc(pred: str, label: str) -> float:
    """Exact match accuracy (1.0 or 0.0)."""
    return 1.0 if pred.upper() == label.upper() else 0.0


def compute_amr(pred: str, label: str) -> float:
    """
    Average Match Ratio: fraction of characters correctly predicted
    at the right position. Used for multi-char ViTC-L.
    """
    if not label:
        return 0.0
    label = label.upper()
    pred = pred.upper().ljust(len(label))[:len(label)]
    matches = sum(p == l for p, l in zip(pred, label))
    return matches / len(label)


# ── Ollama Client ──────────────────────────────────────────────────────────────

class OllamaClient:
    def __init__(self, host: str, model: str, timeout: int = 120):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._verify_connection()

    def _verify_connection(self):
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check if our model is available (strip :tag for matching)
            base = self.model.split(":")[0]
            available = [m.split(":")[0] for m in models]
            if base not in available:
                print(f"[warning] Model '{self.model}' not found in Ollama.")
                print(f"[warning] Available models: {models}")
                print(f"[warning] Run: ollama pull {self.model}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.host}.\n"
                f"Make sure Ollama is running: ollama serve"
            )

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,   # deterministic for eval
                "num_predict": 32,    # short answer expected
            }
        }
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            return "[TIMEOUT]"
        except Exception as e:
            return f"[ERROR: {e}]"


# ── Evaluation Loop ────────────────────────────────────────────────────────────

def evaluate(
    model_name: str,
    task: str,
    data_dir: str,
    ollama_host: str,
    num_samples: Optional[int] = None,
    save_dir: str = RESULTS_DIR,
    delay: float = 0.0,
) -> dict:
    """
    Run ViTC-S or ViTC-L evaluation for a given Ollama model.
    Returns a results dict with per-sample details and aggregate metrics.
    """
    dataset = load_vitc_dataset(data_dir, task)
    
    if num_samples is not None:
        dataset = dataset[:num_samples]
        print(f"[eval] Using {num_samples} samples (out of full dataset)")

    client = OllamaClient(host=ollama_host, model=model_name)
    
    results = []
    total_acc = 0.0
    total_amr = 0.0

    print(f"\n{'='*60}")
    print(f" Model : {model_name}")
    print(f" Task  : ViTC-{task.upper()}")
    print(f" Host  : {ollama_host}")
    print(f" N     : {len(dataset)}")
    print(f"{'='*60}\n")

    for i, sample in enumerate(tqdm(dataset, desc=f"{model_name} | ViTC-{task.upper()}")):
        art_text = sample.get("art_text", sample.get("ascii", ""))
        label = sample.get("text", sample.get("label", "")).strip().upper()
        
        n_chars, n_rows, n_cols = parse_art_metadata(art_text)
        
        prompt = PROMPT_TEMPLATE.format(
            n_chars=n_chars,
            n_rows=n_rows,
            n_cols=n_cols,
            art_text=art_text,
        )
        
        response = client.generate(prompt)
        pred = extract_prediction(response, n_chars)
        
        acc = compute_acc(pred, label)
        amr = compute_amr(pred, label)
        
        total_acc += acc
        total_amr += amr
        
        results.append({
            "idx": i,
            "label": label,
            "prediction": pred,
            "raw_response": response,
            "acc": acc,
            "amr": amr,
            "n_chars": n_chars,
        })
        
        if delay > 0:
            time.sleep(delay)

    n = len(results)
    summary = {
        "model": model_name,
        "task": f"vitc-{task.lower()}",
        "n_samples": n,
        "acc": round(total_acc / n * 100, 2) if n else 0.0,
        "amr": round(total_amr / n * 100, 2) if n else 0.0,
        "samples": results,
    }

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / f"{model_name.replace(':', '-')}_vitc-{task.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[results] ViTC-{task.upper()} | {model_name}")
    print(f"  Acc : {summary['acc']:.2f}%")
    print(f"  AMR : {summary['amr']:.2f}%")
    print(f"  Saved → {out_path}\n")
    
    return summary


# ── Summary Table ──────────────────────────────────────────────────────────────

def print_summary_table(all_results: list[dict]):
    """Print a comparison table across all models and tasks."""
    print("\n" + "="*60)
    print(" FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'Task':<12} {'Acc %':>8} {'AMR %':>8}")
    print("-"*60)
    
    # Paper baselines for reference
    paper_baselines = {
        ("gpt-4", "vitc-s"): (25.19, None),
        ("gpt-4", "vitc-l"): (3.26, None),
        ("gpt-3.5", "vitc-s"): (2.41, None),
        ("llama2", "vitc-s"): (0.0, None),
    }
    
    for r in all_results:
        print(f"{r['model']:<25} {r['task']:<12} {r['acc']:>8.2f} {r['amr']:>8.2f}")
    
    print("-"*60)
    print("  Paper baselines (GPT-4):  ViTC-S 25.19%  |  ViTC-L 3.26%")
    print("="*60)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Ollama models on ViTC-S and ViTC-L benchmarks (ArtPrompt paper)"
    )
    parser.add_argument(
        "--model", nargs="+", required=True,
        help="Ollama model name(s) to evaluate. E.g: llama3 mistral gemma"
    )
    parser.add_argument(
        "--task", nargs="+", default=["s"], choices=["s", "l", "S", "L"],
        help="Which task(s) to run: s=ViTC-S, l=ViTC-L (default: s)"
    )
    parser.add_argument(
        "--data_dir", default=DEFAULT_DATA_DIR,
        help=f"Path to ArtPrompt repo root (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--host", default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama API host (default: {DEFAULT_OLLAMA_HOST})"
    )
    parser.add_argument(
        "--num", type=int, default=None,
        help="Limit number of samples per task (default: all)"
    )
    parser.add_argument(
        "--save_dir", default=RESULTS_DIR,
        help=f"Directory to save result JSONs (default: {RESULTS_DIR})"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Seconds to wait between requests (default: 0)"
    )
    
    args = parser.parse_args()
    
    all_results = []
    
    for model in args.model:
        for task in args.task:
            try:
                result = evaluate(
                    model_name=model,
                    task=task.lower(),
                    data_dir=args.data_dir,
                    ollama_host=args.host,
                    num_samples=args.num,
                    save_dir=args.save_dir,
                    delay=args.delay,
                )
                all_results.append(result)
            except FileNotFoundError as e:
                print(f"\n[error] {e}\n")
            except ConnectionError as e:
                print(f"\n[error] {e}\n")
                break   # No point continuing if Ollama is down

    if len(all_results) > 1:
        print_summary_table(all_results)
    
    # Save combined summary
    if all_results:
        combined_path = Path(args.save_dir) / "combined_summary.json"
        os.makedirs(args.save_dir, exist_ok=True)
        summary_only = [{k: v for k, v in r.items() if k != "samples"} for r in all_results]
        with open(combined_path, "w") as f:
            json.dump(summary_only, f, indent=2)
        print(f"[summary] Combined results saved → {combined_path}")


if __name__ == "__main__":
    main()
