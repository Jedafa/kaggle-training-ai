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
    main_rank = True

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
            "last_ckpt_local": None, "last_ckpt_step": 0, "last_ckpt_uploaded": None,
            "hf_uploading": False, "hf_ckpt_error": None, "command": None,
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
        if not self.main_rank:
            return  # in DDP only rank 0 owns the state file
        self.state.update(kw)
        self.state["updated_at"] = time.time()
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False))
        tmp.replace(self.state_file)

    def progress_point(self, step, loss, lr, epoch):
        if not self.main_rank:
            return
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


HF_BASE = "https://huggingface.co"
DATA_EXT = (".jsonl", ".json", ".csv", ".parquet", ".txt")


def hf_list_dataset_files(repo, token):
    import urllib.request
    hdr = {"User-Agent": "kaggle-training-ai"}
    if token:
        hdr["Authorization"] = "Bearer " + token
    req = urllib.request.Request(f"{HF_BASE}/api/datasets/{repo}/tree/main?recursive=true", headers=hdr)
    with urllib.request.urlopen(req, timeout=30) as r:
        tree = json.loads(r.read().decode())
    files = [f["path"] for f in tree
             if isinstance(f, dict) and f.get("type") == "file"
             and f["path"].lower().endswith(DATA_EXT)]
    if not files:
        raise RuntimeError(f"no data files in {repo}")
    return files


def hf_download_file(repo, path, out_path, token):
    import urllib.request
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = {"User-Agent": "kaggle-training-ai"}
    if token:
        hdr["Authorization"] = "Bearer " + token
    url = f"{HF_BASE}/datasets/{repo}/resolve/main/{path}"
    log(f"hf: downloading {repo}/{path}")
    req = urllib.request.Request(url, headers=hdr)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with urllib.request.urlopen(req, timeout=600) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(out_path)
    return out_path


def hf_download_dataset(repo, token, dest_dir):
    """Download all data files of an HF dataset repo via direct resolve URLs."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    got = []
    for path in hf_list_dataset_files(repo, token):
        safe = re.sub(r"[^A-Za-z0-9._/-]", "_", path)
        got.append(hf_download_file(repo, path, dest_dir / safe.replace("/", "__"), token))
    return got



def sample_sources(cfg, run, tok, rank, failed):
    """Lazily yield (source_name, text) for every dataset source.
    HF shards download one-by-one DURING parsing and stop as soon as the
    per-repo sample limit is reached; already-downloaded shards are reused."""
    max_s = int(cfg.get("max_samples") or 0)
    ds_list = cfg.get("hf_datasets") or []
    if cfg.get("dataset_type") == "hf" and ds_list:
        tok_src = cfg.get("hf_token") or os.environ.get("HF_TOKEN", "")
        prio = {".jsonl": 0, ".csv": 1, ".parquet": 2, ".json": 3, ".txt": 4}
        for repo in ds_list:
            got = 0
            try:
                files = hf_list_dataset_files(repo, tok_src)
                by_stem = {}
                for p in files:
                    stem = re.sub(r"\.(jsonl|json|csv|parquet|txt)$", "", p, flags=re.I).lower()
                    ext = "." + p.rsplit(".", 1)[-1].lower()
                    cur = by_stem.get(stem)
                    if cur is None or prio.get(ext, 9) < prio.get("." + cur.rsplit(".", 1)[-1].lower(), 9):
                        by_stem[stem] = p
                files = sorted(by_stem.values())
                per_file = max(50, -(-max_s // max(1, len(files)))) if (max_s and files) else None
                for idx, path in enumerate(files):
                    if max_s and got >= max_s:
                        break
                    safe = re.sub(r"[^A-Za-z0-9._/-]", "_", path)
                    out_path = BASE / "uploads" / "hf" / safe.replace("/", "__")
                    if rank == 0:
                        if out_path.exists() and out_path.stat().st_size > 0:
                            run.set_state(message=f"reusing {repo} shard {idx+1}/{len(files)} (already downloaded)")
                        else:
                            run.set_state(message=f"downloading {repo} shard {idx+1}/{len(files)}…")
                            try:
                                hf_download_file(repo, path, out_path, tok_src)
                            except Exception as e:
                                failed.append((repo + "/" + path, str(e)[:150]))
                                run.set_state(message=f"download failed: {path} — excluded")
                                continue
                    else:
                        t0 = time.time()
                        while not (out_path.exists() and out_path.stat().st_size > 0):
                            if time.time() - t0 > 1800:
                                raise RuntimeError(f"waiting for {out_path.name} timed out")
                            time.sleep(5)
                    n = 0
                    for text in iter_dataset(out_path, tok, cfg["max_seq"]):
                        yield (repo, text)
                        got += 1
                        n += 1
                        if max_s and got >= max_s:
                            break
                        if per_file and n >= per_file:
                            break
                    log(f"hf {repo}: shard {idx+1}/{len(files)} -> {n} samples (total {got})")
                    run.set_state(message=f"{repo}: {got}/{max_s or 'unlimited'} samples")
            except Exception as e:
                failed.append((repo, str(e)[:150]))
                run.set_state(message=f"repo excluded: {repo} ({str(e)[:100]})")
    else:
        name = {"url": "downloaded", "file": Path(cfg["dataset_value"]).name}.get(
            cfg["dataset_type"], "dataset")
        if rank == 0:
            path = fetch_dataset(cfg)
        else:
            t0 = time.time()
            while not Path(cfg["dataset_value"]).exists() and time.time() - t0 < 1200:
                time.sleep(5)
            path = Path(cfg["dataset_value"])
        n = 0
        for text in iter_dataset(path, tok, cfg["max_seq"]):
            if max_s and n >= max_s:
                break
            n += 1
            yield (name, text)


def pack_from_samples(sample_gen, run, tok, cfg):
    """Stream samples -> tokenize -> pack blocks. Memory O(max_seq)."""
    counts = {}
    order = []
    blocks = []
    src_of_block = []
    ids = []
    cur = None
    total_samples = 0
    eos = tok.eos_token_id or 0
    max_seq = cfg["max_seq"]
    HARD = 400_000
    hard = False
    for sname, text in sample_gen:
        if cur is not None and sname != cur and len(ids) >= max_seq:
            blocks.append(ids[:max_seq]); src_of_block.append(cur)
            counts[cur] = counts.get(cur, 0) + 1
            ids = []
        cur = sname
        ids.extend(tok(text, add_special_tokens=True, truncation=True,
                       max_length=max_seq)["input_ids"])
        ids.append(eos)
        while len(ids) >= max_seq:
            blocks.append(ids[:max_seq]); src_of_block.append(cur)
            counts[cur] = counts.get(cur, 0) + 1
            ids = ids[max_seq:]
            if len(blocks) >= HARD:
                hard = True
                break
        total_samples += 1
        if sname not in order:
            order.append(sname)
        if hard:
            break
        if total_samples % 2000 == 0:
            run.set_state(message=f"packed {total_samples} samples… (dataset: {cur})")
    if not hard and cur is not None and len(ids) >= max_seq // 2:
        blocks.append(ids[:max_seq]); src_of_block.append(cur)
        counts[cur] = counts.get(cur, 0) + 1
    ok_srcs = [s for s in order if counts.get(s, 0) > 0]
    return blocks, src_of_block, counts, ok_srcs, total_samples


def iter_dataset(path, tokenizer, max_seq):
    """Memory-safe streaming parser: jsonl/csv/parquet/txt yield sample texts one by one."""
    suf = path.suffix.lower()
    if suf == ".jsonl":
        with path.open(errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    it = json.loads(line)
                except ValueError:
                    continue
                t = item_to_text(it, tokenizer)
                if t and t.strip():
                    yield t
    elif suf == ".csv":
        import csv
        with path.open(errors="ignore") as f:
            for row in csv.DictReader(f):
                t = item_to_text(row, tokenizer)
                if t and t.strip():
                    yield t
    elif suf == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise RuntimeError("parquet needs pyarrow - pip install pyarrow")
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                t = item_to_text(row, tokenizer)
                if t and t.strip():
                    yield t
    elif suf == ".json":
        if path.stat().st_size > (1 << 30):
            raise RuntimeError(".json over 1GB - split it into .jsonl (one object per line)")
        items = json.loads(path.read_text(errors="ignore"))
        if isinstance(items, dict):
            items = [items]
        for it in items:
            t = item_to_text(it, tokenizer)
            if t and t.strip():
                yield t
    else:
        with path.open(errors="ignore") as f:
            while True:
                chunk = f.read(max_seq * 4)
                if not chunk:
                    break
                if chunk.strip():
                    yield chunk


def pack_sources(cfg, run, sources, tok, rank, world):
    """Stream every source -> tokenize -> pack blocks (memory O(max_seq)).
    Returns (blocks, src_of_block, counts, ok_srcs, failed, total_samples)."""
    failed = []
    counts = {}
    order = []
    blocks = []
    src_of_block = []
    ids = []
    cur = None
    total_samples = 0
    eos = tok.eos_token_id or 0
    max_seq = cfg["max_seq"]
    max_s = int(cfg.get("max_samples") or 0)
    HARD_BLOCKS = 400_000
    hard_hit = False
    for sname, spath, slim in sources:
        try:
            n = 0
            lim = slim if slim else max_s
            for text in iter_dataset(Path(spath), tok, max_seq):
                if cur is not None and sname != cur and len(ids) >= max_seq:
                    blocks.append(ids[:max_seq]); src_of_block.append(cur)
                    counts[cur] = counts.get(cur, 0) + 1
                    ids = []
                cur = sname
                ids.extend(tok(text, add_special_tokens=True, truncation=True,
                               max_length=max_seq)["input_ids"])
                ids.append(eos)
                while len(ids) >= max_seq:
                    blocks.append(ids[:max_seq]); src_of_block.append(cur)
                    counts[cur] = counts.get(cur, 0) + 1
                    ids = ids[max_seq:]
                    if len(blocks) >= HARD_BLOCKS:
                        hard_hit = True
                        break
                n += 1
                total_samples += 1
                if sname not in order:
                    order.append(sname)
                if n % 5000 == 0:
                    run.set_state(message=f"packing {sname}: {n} samples…",
                                  current_dataset=sname)
                if hard_hit or (lim and n >= lim):
                    break
            log(f"source {'ok' if n else 'EMPTY'}: {sname} — {n} samples")
        except Exception as e:
            failed.append((sname, str(e)[:150]))
            log(f"source FAILED, excluded: {sname} — {e}")
        if hard_hit or len(blocks) >= HARD_BLOCKS:
            break
    if not hard_hit and len(ids) >= max_seq // 2:
        blocks.append(ids[:max_seq]); src_of_block.append(cur)
        counts[cur] = counts.get(cur, 0) + 1
    ok_srcs = [s for s in order if counts.get(s, 0) > 0]
    if hard_hit:
        run.set_state(message="hard cap 400k blocks reached - train on this, then continue from checkpoint for more")
    return blocks, src_of_block, counts, ok_srcs, failed, total_samples


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
def train_real(cfg, run, rank=0, world=1):
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

    failed = []
    def sample_gen():
        yield from sample_sources(cfg, run, tok, rank, failed)
    blocks, src_of_block, counts, ok_srcs, total_samples = \
        pack_from_samples(sample_gen(), run, tok, cfg)
    log(f"packed blocks (len={cfg['max_seq']}): {len(blocks)}; ok sources: {ok_srcs}")
    if not blocks:
        raise RuntimeError("dataset too small for one block — lower max_seq or add data")

    # step boundaries per source (epochs apply to the combined stream)
    counts = {}
    for b_src in src_of_block:
        counts[b_src] = counts.get(b_src, 0) + 1
    eff = max(1, cfg["batch_size"] * cfg["grad_accum"] * world)
    steps_per_src, acc = [], 0
    for s in ok_srcs:
        acc += max(1, round(counts.get(s, 0) * cfg["epochs"] / eff))
        steps_per_src.append((s, acc))
    def src_for_step(step):
        for s, upto in steps_per_src:
            if step <= upto:
                return s
        return ok_srcs[-1]
    total_steps = max(1, (len(blocks) * cfg["epochs"]) // eff)
    run.set_state(message=f"{len(all_samples)} samples → {len(blocks)} blocks · {world} GPU",

                  total_steps=total_steps,
                  datasets_overview=[{"name": s, "blocks": counts.get(s, 0)} for s in ok_srcs],
                  datasets_failed=[{"name": n, "error": e} for n, e in failed],
                  current_dataset=ok_srcs[0])

    qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    if world > 1:
        from accelerate import PartialState
        dev = PartialState().device
        log(f"DDP rank {rank}/{world} on {dev}")
        placement = {"": dev}
    else:
        placement = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=qconf, device_map=placement,
        torch_dtype=torch.float16, attn_implementation="sdpa")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    resume = cfg.get("resume_from")
    resume_ckpt = None
    if resume and resume.startswith("hf:"):
        cfg["hf_repo"] = resume[3:] if len(resume) > 3 else cfg.get("hf_repo")
        resume_ckpt = hf_fetch_checkpoint(cfg, run)
        resume = str(resume_ckpt)
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

    ckpt_upload_on = bool(cfg.get("hf_token") or os.environ.get("HF_TOKEN")) and \
        (cfg.get("hf_upload") or cfg.get("ckpt_upload"))

    class CB(TrainerCallback):
        def __init__(self):
            self.last_time_save = time.time()

        def on_log(self, args, state, control, logs=None, **kw):
            if rank != 0:
                return
            if logs and "loss" in logs and not run.stopping:
                step = state.global_step or 0
                run.set_state(status="training", phase="train", step=step,
                              loss=round(float(logs["loss"]), 4),
                              lr=float(logs.get("learning_rate", 0) or 0),
                              epoch=round(float(logs.get("epoch", 0) or 0), 3))
                run.progress_point(step, round(float(logs["loss"]), 4),
                                   float(logs.get("learning_rate", 0) or 0),
                                   round(float(logs.get("epoch", 0) or 0), 3))
                try:
                    run.set_state(current_dataset=src_for_step(step))
                except NameError:
                    pass

        def on_step_end(self, args, state, control, **kw):
            if rank != 0:
                return
            st = run.state
            cmd = st.get("command")
            if cmd:
                run.set_state(command=None)
                if cmd == "save":
                    control.should_save = True
                elif cmd == "upload":
                    import threading
                    threading.Thread(target=_do_manual_upload, args=(cfg, run),
                                     daemon=True).start()
            minutes = cfg.get("save_minutes") or 0
            if minutes and time.time() - self.last_time_save >= minutes * 60:
                control.should_save = True

        def on_save(self, args, state, control, **kw):
            if rank != 0:
                return
            self.last_time_save = time.time()
            ckdir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            try:
                shutil.copy(Path(cfg["_run_dir"]) / "train_config.json", ckdir / "train_config.json")
            except Exception:
                pass
            run.set_state(last_ckpt_local=str(ckdir), last_ckpt_step=state.global_step)
            log(f"checkpoint saved: {ckdir.name}")
            if ckpt_upload_on:
                import threading
                threading.Thread(target=upload_checkpoint, args=(cfg, run, ckdir),
                                 daemon=True).start()

    targs = TrainingArguments(
        output_dir=str(run.dir / "ckpt"),
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        logging_steps=5, save_steps=cfg["save_steps"],
        save_total_limit=int(cfg.get("keep_ckpts") or 2),
        fp16=True, optim="paged_adamw_8bit", gradient_checkpointing=True,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        report_to=[], remove_unused_columns=False, dataloader_drop_last=True)

    trainer = Trainer(model=model, args=targs, train_dataset=BlockDataset(blocks),
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
                      callbacks=[CB()])
    if resume_ckpt:
        # restore optimizer/scheduler/RNG and step counter (weights come from the adapter above)
        try:
            trainer._load_optimizer_and_scheduler(resume_ckpt)
            trainer._load_rng_state(resume_ckpt)
            ts = json.loads((Path(resume_ckpt) / "trainer_state.json").read_text())
            trainer.state.global_step = ts.get("global_step", 0)
            log(f"resumed optimizer/scheduler state at step {trainer.state.global_step}")
        except Exception as e:
            log(f"optimizer state restore skipped: {e}")
    trainer.train()

    adapter_dir = run.dir / "adapter"
    if rank == 0:
        trainer.save_model(str(adapter_dir))
        tok.save_pretrained(str(adapter_dir))
        run.set_state(status="merging", phase="merge", adapter_dir=str(adapter_dir),
                      message="training done — merging adapter")
    else:
        log(f"rank {rank} finished training — exiting (merge runs on rank 0)")
        sys.exit(0)

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


# ---------------------------------------------------------------- checkpoints
def _hf_api(tok):
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise RuntimeError("huggingface_hub not installed — pip install huggingface_hub") from e
    return HfApi(token=tok)


def hf_repo_id(cfg, run):
    tok = cfg.get("hf_token") or os.environ.get("HF_TOKEN", "")
    user = cfg.get("hf_user") or os.environ.get("HF_USER", "")
    repo = cfg.get("hf_repo")
    if not repo:
        if user:
            repo = user + "/" + cfg["run_name"]
        elif tok:
            try:
                repo = _hf_api(tok).whoami()["name"] + "/" + cfg["run_name"]
            except Exception as e:
                log("whoami failed (%s) - set HF username in Settings" % e)
    return tok, repo


def upload_checkpoint(cfg, run, ckpt_dir):
    """Upload a trainer checkpoint dir to HF so training can resume anywhere.
    Retries 3x; only marks uploaded after files are verified on the repo."""
    tok, repo = hf_repo_id(cfg, run)
    if not tok or not repo:
        run.set_state(hf_ckpt_error="no HF token/repo — checkpoint kept local")
        return False
    ckpt_dir = Path(ckpt_dir)
    if not (ckpt_dir / "optimizer.pt").exists():
        run.set_state(hf_ckpt_error="checkpoint dir has no optimizer state")
        return False
    api = _hf_api(tok)
    for attempt in range(1, 4):
        try:
            run.set_state(hf_uploading=True, hf_ckpt_error=None,
                          message=f"uploading checkpoint to HF (try {attempt}/3)…")
            api.upload_folder(folder_path=str(ckpt_dir), repo_id=repo,
                              path_in_repo="checkpoint", private=True,
                              commit_message=f"checkpoint step {run.state.get('last_ckpt_step')}")
            files = api.list_repo_files(repo)
            need = ("checkpoint/optimizer.pt", "checkpoint/scheduler.pt",
                    "checkpoint/trainer_state.json")
            if not all(n in files for n in need):
                raise RuntimeError(f"verification failed: missing {[n for n in need if n not in files]}")
            run.set_state(hf_uploading=False,
                          last_ckpt_uploaded=run.state.get("last_ckpt_step"),
                          message=f"checkpoint uploaded ✓ (step {run.state.get('last_ckpt_step')})")
            log(f"checkpoint uploaded to HF: {repo} (step {run.state.get('last_ckpt_step')})")
            return True
        except Exception as e:
            log(f"HF checkpoint upload attempt {attempt} failed: {e}")
            run.set_state(hf_uploading=False,
                          hf_ckpt_error=str(e)[:300],
                          message=f"HF upload failed (try {attempt}/3)")
            time.sleep(8 * attempt)
    return False


def hf_fetch_checkpoint(cfg, run):
    """Download the latest checkpoint from HF. Returns local dir or raises."""
    tok, repo = hf_repo_id(cfg, run)
    if not repo:
        raise RuntimeError("HF resume: no repo configured")
    run.set_state(message=f"downloading checkpoint from HF: {repo}…")
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo, token=tok or None,
                              local_dir=str(run.dir / "hf_ckpt"),
                              allow_patterns=["checkpoint/*"])
    cands = sorted(Path(local).glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)
    if not cands:
        cands = [Path(local) / "checkpoint"]
    ckpt = cands[-1]
    if not (ckpt / "trainer_state.json").exists():
        raise RuntimeError("HF checkpoint incomplete (no trainer_state.json)")
    log(f"checkpoint from HF: {ckpt} (step {json.loads((ckpt / 'trainer_state.json').read_text())['global_step']})")
    return ckpt


def _do_manual_upload(cfg, run):
    ck = run.state.get("last_ckpt_local")
    if not ck or not Path(ck).exists():
        run.set_state(hf_ckpt_error="no local checkpoint yet")
        return
    upload_checkpoint(cfg, run, ck)


# ---------------------------------------------------------------- from scratch
SCRATCH_SIZES = {
    "tiny":   {"hidden_size": 512,  "num_hidden_layers": 8,  "num_attention_heads": 8,  "intermediate_size": 1376},
    "small":  {"hidden_size": 768,  "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 2048},
    "medium": {"hidden_size": 1024, "num_hidden_layers": 16, "num_attention_heads": 16, "intermediate_size": 2816},
}


def train_scratch(cfg, run, rank=0, world=1):
    """Train a small LLM from random weights (no base model). Full-precision params + AMP."""
    import gc
    import torch
    from transformers import (AutoTokenizer, LlamaConfig, LlamaForCausalLM,
                              Trainer, TrainerCallback, TrainingArguments,
                              DataCollatorForLanguageModeling)
    run.set_state(status="training", phase="train", message="building model from random weights…")
    tok = AutoTokenizer.from_pretrained(cfg.get("base_model") or "Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    failed = []
    def sample_gen():
        yield from sample_sources(cfg, run, tok, rank, failed)
    blocks, src_of_block, counts, ok_srcs, total_samples = \
        pack_from_samples(sample_gen(), run, tok, cfg)
    log(f"packed blocks: {len(blocks)}")
    if not blocks:
        raise RuntimeError("dataset too small for one block")

    eff = max(1, cfg["batch_size"] * cfg["grad_accum"] * world)
    total_steps = max(1, (len(blocks) * cfg["epochs"]) // eff)
    steps_per_src, acc = [], 0
    for s in ok_srcs:
        acc += max(1, round(counts.get(s, 0) * cfg["epochs"] / eff))
        steps_per_src.append((s, acc))
    def src_for_step(step):
        for s, upto in steps_per_src:
            if step <= upto:
                return s
        return ok_srcs[-1]
    run.set_state(total_steps=total_steps, current_dataset=ok_srcs[0],
                  datasets_overview=[{"name": s, "blocks": counts.get(s, 0)} for s in ok_srcs],
                  datasets_failed=[{"name": n, "error": e} for n, e in failed])

    if world > 1:
        from accelerate import PartialState
        dev = PartialState().device
        model = LlamaForCausalLM(config).to(dev)
    else:
        model = LlamaForCausalLM(config).to("cuda" if torch.cuda.is_available() else "cpu")

    class CB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if rank != 0:
                return
            if logs and "loss" in logs and not run.stopping:
                step = state.global_step or 0
                run.set_state(status="training", phase="train", step=step,
                              loss=round(float(logs["loss"]), 4),
                              lr=float(logs.get("learning_rate", 0) or 0),
                              epoch=round(float(logs.get("epoch", 0) or 0), 3))
                run.progress_point(step, round(float(logs["loss"]), 4),
                                   float(logs.get("learning_rate", 0) or 0),
                                   round(float(logs.get("epoch", 0) or 0), 3))
                try:
                    run.set_state(current_dataset=src_for_step(step))
                except NameError:
                    pass

    targs = TrainingArguments(
        output_dir=str(run.dir / "ckpt"),
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"] if cfg["lr"] > 3e-4 else 5e-4,
        logging_steps=5, save_steps=cfg["save_steps"],
        save_total_limit=int(cfg.get("keep_ckpts") or 2),
        fp16=True, optim="adamw_torch_fused",
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        report_to=[], remove_unused_columns=False, dataloader_drop_last=True)

    trainer = Trainer(model=model, args=targs, train_dataset=BlockDataset(blocks),
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
                      callbacks=[CB()])
    trainer.train()

    merged = run.dir / "merged"
    if rank == 0:
        model.save_pretrained(str(merged), safe_serialization=True)
        tok.save_pretrained(str(merged))
        run.set_state(status="merging", phase="merge", merged_dir=str(merged),
                      adapter_dir=str(merged),
                      message="from-scratch training done — model saved (no adapter needed)")
    else:
        log(f"rank {rank} finished — exiting")
        sys.exit(0)
    del model, trainer
    gc.collect()


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
        cmd = run.state.get("command")
        if cmd:
            run.set_state(command=None)
            if cmd == "save":
                ck = run.dir / f"checkpoint-{step}"
                ck.mkdir(parents=True, exist_ok=True)
                (ck / "optimizer.pt").write_bytes(b"fake")
                (ck / "trainer_state.json").write_text(json.dumps({"global_step": step}))
                shutil.copy(Path(cfg["_run_dir"]) / "train_config.json", ck / "train_config.json")
                run.set_state(last_ckpt_local=str(ck), last_ckpt_step=step)
                log(f"checkpoint saved: {ck.name}")
                if cfg.get("hf_token"):
                    upload_checkpoint(cfg, run, ck)
            elif cmd == "upload":
                if cfg.get("hf_token"):
                    import threading
                    threading.Thread(target=_do_manual_upload, args=(cfg, run), daemon=True).start()
        loss = max(0.15, 2.6 * (2.718 ** (-step / 40)) + 0.1 + random.uniform(0, 0.06))
        if step in (40, 80):
            ck = run.dir / f"checkpoint-{step}"
            ck.mkdir(parents=True, exist_ok=True)
            (ck / "optimizer.pt").write_bytes(b"fake")
            (ck / "trainer_state.json").write_text(json.dumps({"global_step": step}))
            run.set_state(last_ckpt_local=str(ck), last_ckpt_step=step)
            log(f"checkpoint saved: {ck.name}")
            if cfg.get("hf_token"):
                upload_checkpoint(cfg, run, ck)
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
    rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
    world = int(os.environ.get("WORLD_SIZE", "1") or 1)
    run = Run(cfg)
    run.main_rank = (rank == 0) or world == 1
    try:
        if cfg.get("hf_token"):
            os.environ.setdefault("HF_TOKEN", cfg["hf_token"])
        if FAKE:
            fake_run(cfg, run)
        elif cfg.get("mode") == "scratch":
            train_scratch(cfg, run, rank, world)
        else:
            train_real(cfg, run, rank, world)
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
