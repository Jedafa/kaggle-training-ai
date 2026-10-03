#!/usr/bin/env python3
"""Training pipeline: dataset → QLoRA fine-tune → merge → GGUF → Ollama → HuggingFace.

Runs as a detached subprocess; reports progress to:
  runs/<name>/state.json      — phase/status/metadata (atomic writes)
  runs/<name>/progress.jsonl  — one {"step","loss","lr","epoch"} per log point
  runs/<name>/train.log       — full stdout/stderr
TRAINER_FAKE=1 simulates the whole pipeline without a GPU (for testing the panel).
"""
import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = Path.home() / ".kaggle-panel"
TOOLS = BASE / "tools"
FAKE = os.environ.get("TRAINER_FAKE") == "1"


def log(m):
    print(f"[trainer {time.strftime('%H:%M:%S')}] {m}", flush=True)


class Run:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = Path(cfg["_run_dir"])
        self.state_file = self.dir / "state.json"
        self.progress_file = self.dir / "progress.jsonl"
        self.stopping = False
        self.state = {
            "run": cfg["run_name"], "status": "preparing", "phase": "prepare",
            "step": 0, "total_steps": 0, "loss": None, "lr": None, "epoch": 0,
            "message": "", "error": None,
            "started_at": time.time(), "updated_at": time.time(), "pid": os.getpid(),
            "base_model": cfg.get("base_model"), "adapter_dir": None,
            "merged_dir": None, "gguf_path": None,
            "ollama_model": None, "ollama_ready": False, "hf_url": None,
        }
        self.set_state()
        (self.dir / "train.pid").write_text(str(os.getpid()))
        signal.signal(signal.SIGTERM, self._sigterm)

    def _sigterm(self, *a):
        self.stopping = True
        self.set_state(status="stopped", message="stopped by user")
        log("SIGTERM — stopping")
        sys.exit(0)

    def set_state(self, **kw):
        self.state.update(kw)
        self.state["updated_at"] = time.time()
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False))
        tmp.replace(self.state_file)

    def progress_point(self, step, loss, lr, epoch):
        with self.progress_file.open("a") as f:
            f.write(json.dumps({"step": step, "loss": loss, "lr": lr,
                                "epoch": epoch, "ts": time.time()}) + "\n")


# ---------------------------------------------------------------- dataset
def fetch_dataset(cfg):
    v = cfg["dataset_value"]
    if cfg["dataset_type"] == "file":
        p = Path(v)
        if not p.exists():
            raise RuntimeError(f"dataset file missing: {v}")
        return p
    name = re.sub(r"[^A-Za-z0-9._-]", "_", v.split("/")[-1].split("?")[0]) or "dataset.bin"
    p = BASE / "uploads" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading dataset: {v}")
    urllib.request.urlretrieve(v, p)
    log(f"downloaded → {p.name} ({p.stat().st_size} bytes)")
    return p


def item_to_text(it, tokenizer):
    if not isinstance(it, dict):
        return str(it)
    low = {(k.lower() if isinstance(k, str) else k): v for k, v in it.items()}
    msgs = low.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False)
        except Exception:
            pass
    ins = low.get("instruction") or low.get("prompt") or low.get("question") or low.get("input") or ""
    out = low.get("output") or low.get("response") or low.get("answer") or low.get("completion") or ""
    if ins and out:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": str(ins)},
                 {"role": "assistant", "content": str(out)}], tokenize=False)
        except Exception:
            return f"### Instruction:\n{ins}\n\n### Response:\n{out}"
    return str(low.get("text") or low.get("content") or low.get("body") or "")


def parse_dataset(path, tokenizer, max_seq):
    suf = path.suffix.lower()
    samples = []
    if suf in (".jsonl", ".json"):
        text = path.read_text(errors="ignore")
        items = []
        if suf == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        else:
            items = json.loads(text)
            if isinstance(items, dict):
                items = [items]
        for it in items:
            samples.append(item_to_text(it, tokenizer))
    elif suf == ".csv":
        import csv
        with path.open(errors="ignore") as f:
            for row in csv.DictReader(f):
                samples.append(item_to_text(row, tokenizer))
    elif suf == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            samples.append(item_to_text(row.to_dict(), tokenizer))
    else:  # raw text
        t = path.read_text(errors="ignore")
        chunk = max_seq * 4
        samples = [t[i:i + chunk] for i in range(0, len(t), chunk)]
    samples = [s for s in samples if s and s.strip()]
    if not samples:
        raise RuntimeError("dataset parsed to 0 samples — check the format")
    return samples


def build_blocks(samples, tokenizer, max_seq):
    ids = []
    eos = tokenizer.eos_token_id or 0
    for s in samples:
        ids.extend(tokenizer(s, add_special_tokens=True,
                             truncation=True, max_length=max_seq)["input_ids"])
        ids.append(eos)
    n = len(ids) // max_seq
    return [ids[i * max_seq:(i + 1) * max_seq] for i in range(n)]


class BlockDataset:
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        import torch
        ids = torch.tensor(self.blocks[i], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone(),
                "attention_mask": torch.ones_like(ids)}


# ---------------------------------------------------------------- real pipeline
def train_real(cfg, run):
    import gc
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainerCallback, TrainingArguments,
                              BitsAndBytesConfig, DataCollatorForLanguageModeling)
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    run.set_state(status="training", phase="train", message="loading model…")
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    samples = parse_dataset(fetch_dataset(cfg), tok, cfg["max_seq"])
    log(f"dataset samples: {len(samples)}")
    blocks = build_blocks(samples, tok, cfg["max_seq"])
    log(f"packed blocks (len={cfg['max_seq']}): {len(blocks)}")
    if not blocks:
        raise RuntimeError("dataset too small for one block — lower max_seq or add data")
    total_steps = max(1, (len(blocks) * cfg["epochs"]) //
                      max(1, cfg["batch_size"] * cfg["grad_accum"]))
    run.set_state(message=f"{len(samples)} samples → {len(blocks)} blocks",
                  total_steps=total_steps)

    qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=qconf, device_map="auto",
        torch_dtype=torch.float16, attn_implementation="sdpa")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    resume = cfg.get("resume_from")
    if resume and Path(resume).exists():
        model = PeftModel.from_pretrained(model, resume, is_trainable=True)
        log(f"resumed adapter: {resume}")
    else:
        lconf = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_r"] * 2,
                           lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                           "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lconf)
    model.print_trainable_parameters()

    class CB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs and not run.stopping:
                step = state.global_step or 0
                run.set_state(status="training", phase="train", step=step,
                              loss=round(float(logs["loss"]), 4),
                              lr=float(logs.get("learning_rate", 0) or 0),
                              epoch=round(float(logs.get("epoch", 0) or 0), 3))
                run.progress_point(step, round(float(logs["loss"]), 4),
                                   float(logs.get("learning_rate", 0) or 0),
                                   round(float(logs.get("epoch", 0) or 0), 3))

    targs = TrainingArguments(
        output_dir=str(run.dir / "ckpt"),
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        logging_steps=5, save_steps=cfg["save_steps"], save_total_limit=2,
        fp16=True, optim="paged_adamw_8bit", gradient_checkpointing=True,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        report_to=[], remove_unused_columns=False, dataloader_drop_last=True)

    trainer = Trainer(model=model, args=targs, train_dataset=BlockDataset(blocks),
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
                      callbacks=[CB()])
    trainer.train()

    adapter_dir = run.dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    run.set_state(status="merging", phase="merge", adapter_dir=str(adapter_dir),
                  message="training done — merging adapter")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    merge_real(cfg, run, adapter_dir, tok)


def merge_real(cfg, run, adapter_dir, tok):
    import gc
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    log("merging adapter into base (fp16 on cpu)…")
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True)
    m = PeftModel.from_pretrained(base, adapter_dir)
    m = m.merge_and_unload()
    merged = run.dir / "merged"
    m.save_pretrained(str(merged), safe_serialization=True)
    tok.save_pretrained(str(merged))
    run.set_state(merged_dir=str(merged))
    del m, base
    gc.collect()


def to_gguf(cfg, run):
    run.set_state(status="gguf", phase="gguf", message="converting to GGUF…")
    merged = Path(run.state["merged_dir"])
    TOOLS.mkdir(parents=True, exist_ok=True)
    script = TOOLS / "convert_hf_to_gguf.py"
    if not script.exists():
        log("downloading llama.cpp convert script…")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py",
            script)
    try:
        import gguf  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf"], check=False)
    gguf_path = run.dir / f"{cfg['run_name']}.gguf"
    cmd = [sys.executable, str(script), str(merged), "--outfile", str(gguf_path),
           "--outtype", cfg.get("gguf_outtype", "q8_0")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not gguf_path.exists():
        log(f"convert failed → fallback f16\n{r.stdout[-500:]}\n{r.stderr[-500:]}")
        gguf_path = run.dir / f"{cfg['run_name']}-f16.gguf"
        subprocess.run([sys.executable, str(script), str(merged), "--outfile",
                        str(gguf_path), "--outtype", "f16"], check=True)
    run.set_state(gguf_path=str(gguf_path))
    log(f"GGUF ready: {gguf_path.name} ({gguf_path.stat().st_size >> 20} MB)")


def ollama_import(cfg, run):
    if not cfg.get("ollama_import"):
        return
    run.set_state(status="ollama", phase="ollama", message="importing into Ollama…")
    if subprocess.run(["pgrep", "-x", "ollama"], capture_output=True).returncode != 0:
        subprocess.Popen("nohup ollama serve >> $HOME/.kaggle-panel/ollama.log 2>&1 &",
                         shell=True, start_new_session=True)
        for _ in range(60):
            if subprocess.run(["pgrep", "-x", "ollama"],
                              capture_output=True).returncode == 0:
                break
            time.sleep(1)
    time.sleep(2)
    name = cfg.get("ollama_name") or cfg["run_name"]
    (run.dir / "Modelfile").write_text(f"FROM {run.state['gguf_path']}\n")
    r = subprocess.run(["ollama", "create", name, "-f", str(run.dir / "Modelfile")],
                       capture_output=True, text=True)
    log(f"ollama create: {(r.stdout or r.stderr)[-400:]}")
    ok = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    ready = name in ok
    run.set_state(ollama_model=name, ollama_ready=ready)
    log(f"ollama model '{name}' ready" if ready else "ollama import FAILED")


def hf_upload(cfg, run):
    if not cfg.get("hf_upload"):
        return
    run.set_state(status="hf_upload", phase="hf", message="uploading to HuggingFace…")
    tok = cfg.get("hf_token") or os.environ.get("HF_TOKEN", "")
    if not tok:
        run.set_state(message="HF: no token — upload skipped")
        return
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    who = api.whoami()["name"]
    repo = cfg.get("hf_repo") or f"{who}/{cfg['run_name']}"
    api.create_repo(repo, exist_ok=True, private=True)
    gg = run.state.get("gguf_path")
    if gg and Path(gg).exists():
        api.upload_file(path_or_fileobj=gg, path_in_repo=Path(gg).name, repo_id=repo)
    ad = Path(run.state.get("adapter_dir") or "")
    if ad.exists():
        api.upload_folder(folder_path=str(ad), repo_id=repo, path_in_repo="adapter")
    run.set_state(hf_url=f"https://huggingface.co/{repo}", message="uploaded to HuggingFace")
    log(f"uploaded → https://huggingface.co/{repo}")


# ---------------------------------------------------------------- fake pipeline
def fake_run(cfg, run):
    log("FAKE MODE — simulating the pipeline")
    run.set_state(message="fake training (no GPU)")
    time.sleep(1.5)
    run.set_state(status="training", phase="train", total_steps=120)
    loss = 2.6
    for step in range(1, 121):
        if run.stopping:
            return
        loss = max(0.15, 2.6 * (2.718 ** (-step / 40)) + 0.1 + random.uniform(0, 0.06))
        if step % 5 == 0:
            run.set_state(status="training", phase="train", step=step,
                          loss=round(loss, 4), lr=cfg["lr"] * (1 - step / 160),
                          epoch=round(step / 40, 3))
            run.progress_point(step, round(loss, 4), cfg["lr"], step / 40)
        time.sleep(0.12)
    adapter = run.dir / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text('{"fake": true}')
    run.set_state(status="merging", phase="merge", adapter_dir=str(adapter))
    merged = run.dir / "merged"
    merged.mkdir(exist_ok=True)
    (merged / "config.json").write_text('{"fake": true}')
    run.set_state(status="gguf", phase="gguf", merged_dir=str(merged))
    time.sleep(1)
    gguf_path = run.dir / f"{cfg['run_name']}.gguf"
    gguf_path.write_bytes(b"GGUF-fake-model" + os.urandom(4096))
    run.set_state(status="ollama", phase="ollama", gguf_path=str(gguf_path))
    time.sleep(1)
    run.set_state(ollama_model=cfg.get("ollama_name"), ollama_ready=True)
    if cfg.get("hf_token"):
        run.set_state(status="hf_upload", phase="hf")
        time.sleep(1)
        run.set_state(hf_url=f"https://huggingface.co/fake/{cfg['run_name']}")
    run.set_state(status="done", phase="done", message="fake model ready")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    cfg["_run_dir"] = str(Path(a.config).parent)
    run = Run(cfg)
    try:
        if cfg.get("hf_token"):
            os.environ.setdefault("HF_TOKEN", cfg["hf_token"])
        if FAKE:
            fake_run(cfg, run)
        else:
            train_real(cfg, run)
            to_gguf(cfg, run)
            ollama_import(cfg, run)
            hf_upload(cfg, run)
            run.set_state(status="done", phase="done",
                          message=f"model ready → ollama run {cfg.get('ollama_name')}")
        log(f"finished: {run.state['status']}")
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        run.set_state(status="error", error=str(e)[:600], message="see train.log")


if __name__ == "__main__":
    main()
