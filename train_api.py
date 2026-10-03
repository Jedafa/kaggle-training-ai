"""Training orchestration API — runs, dataset uploads, progress, export."""
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
import zipfile
import asyncio
import urllib.request
import urllib.error
from pathlib import Path

from aiohttp import web

HOME = Path.home()
BASE = HOME / ".kaggle-panel"
RUNS = BASE / "runs"
UPLOADS = BASE / "uploads"
PANEL_DIR = Path(__file__).resolve().parent
TRAINER = PANEL_DIR / "trainer.py"

_live = {}  # run_name -> Popen

ACTIVE_STATUSES = ("preparing", "training", "merging", "gguf", "ollama", "hf_upload", "stopping")
FINAL_STATUSES = ("done", "error", "stopped")


def safe_name(s, fallback="run"):
    s = re.sub(r"[^A-Za-z0-9._-]", "_", str(s or "").strip())
    s = s.strip("._") or fallback
    return s[:64]


def run_dir(name):
    return RUNS / name


def state_path(name):
    return run_dir(name) / "state.json"


def read_state(name):
    try:
        return json.loads(state_path(name).read_text())
    except Exception:
        return {"run": name, "status": "unknown", "phase": "", "step": 0,
                "total_steps": 0, "loss": None, "message": "", "error": None}


def write_state(name, patch):
    d = run_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    st = read_state(name)
    st.update(patch)
    st["updated_at"] = time.time()
    tmp = state_path(name).with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False))
    tmp.replace(state_path(name))
    return st


def env_vars():
    vals = {}
    f = BASE / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                vals[k.strip()] = v.strip().strip('"')
    return vals


def _proc_alive(name):
    proc = _live.get(name)
    if proc is not None:
        return proc.poll() is None
    pf = run_dir(name) / "train.pid"
    if pf.exists():
        try:
            os.kill(int(pf.read_text().strip()), 0)
            return True
        except Exception:
            return False
    return False


def launch_training(body):
    """Validate + spawn a training run. Returns (ok, response_dict)."""
    for n, p in list(_live.items()):
        if p.poll() is None:
            return False, {"error": "already_training", "run": n, "status": 409}

    name = safe_name(body.get("run_name") or f"run-{secrets.token_hex(3)}")
    st = read_state(name)
    if st.get("status") in ACTIVE_STATUSES and _proc_alive(name):
        return False, {"error": "run_in_progress", "run": name, "status": 409}

    dstype = body.get("dataset_type")
    value = str(body.get("dataset_value") or "").strip()
    if dstype == "url":
        if not re.match(r"^https?://", value):
            return False, {"error": "bad_dataset_url", "status": 400}
    elif dstype == "file":
        p = Path(value)
        if not p.exists():
            p = UPLOADS / safe_name(value, "dataset.bin")
        if not p.exists():
            return False, {"error": "dataset_not_found", "status": 400}
        value = str(p)
    elif dstype == "hf":
        repos = [x.strip() for x in re.split(r"[,;\n]+", value) if x.strip()]
        if not repos:
            return False, {"error": "no HF dataset repos given", "status": 400}
        value = ",".join(repos)
    else:
        return False, {"error": "dataset_type must be url|file", "status": 400}

    cfg = {
        "run_name": name,
        "base_model": str(body.get("base_model") or "unsloth/Llama-3.2-3B-Instruct").strip(),
        "dataset_type": dstype,
        "dataset_value": value,
        "hf_datasets": ([x.strip() for x in re.split(r"[,;\n]+", value)] if dstype == "hf" else []),
        "epochs": float(body.get("epochs") or 3),
        "lr": float(body.get("lr") or 2e-4),
        "batch_size": int(body.get("batch_size") or 2),
        "grad_accum": int(body.get("grad_accum") or 4),
        "max_seq": int(body.get("max_seq") or 1024),
        "lora_r": int(body.get("lora_r") or 16),
        "save_steps": int(body.get("save_steps") or 100),
        "save_minutes": float(body.get("save_minutes") or 0) or None,
        "keep_ckpts": int(body.get("keep_ckpts") or 2),
        "max_samples": int(body.get("max_samples") or 5000),
        "ckpt_upload": bool(body.get("ckpt_upload", True)),
        "resume_from": str(body.get("resume_from") or "").strip() or None,
        "gguf_outtype": body.get("gguf_outtype") if body.get("gguf_outtype") in ("q8_0", "f16") else "q8_0",
        "ollama_import": bool(body.get("ollama_import", True)),
        "ollama_name": safe_name(body.get("ollama_name") or name).lower(),
        "hf_upload": bool(body.get("hf_upload", False)),
        "hf_repo": str(body.get("hf_repo") or "").strip() or None,
        "hf_token": str(body.get("hf_token") or "").strip() or env_vars().get("HF_TOKEN", ""),
        "hf_user": str(body.get("hf_user") or "").strip() or env_vars().get("HF_USER", ""),
        "created_at": time.time(),
    }
    d = run_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "train_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    (d / "train.log").write_text("")

    # training takes all VRAM — stop ollama first
    subprocess.run(["pkill", "-x", "ollama"], capture_output=True)

    env = os.environ.copy()
    if cfg["hf_token"]:
        env["HF_TOKEN"] = cfg["hf_token"]
    proc = subprocess.Popen(
        [sys.executable, str(TRAINER), "--config", str(d / "train_config.json")],
        stdout=open(d / "train.log", "ab"), stderr=subprocess.STDOUT,
        start_new_session=True, env=env, cwd=str(PANEL_DIR))
    _live[name] = proc
    write_state(name, {"status": "preparing", "phase": "prepare", "pid": proc.pid,
                       "error": None, "message": "trainer started (ollama stopped to free VRAM)"})
    return True, {"ok": True, "run": name, "note": "ollama stopped for training"}


async def api_train_start(request):
    body = await request.json()
    ok, resp = launch_training(body)
    return web.json_response(resp, status=200 if ok else resp.get("status", 400))


async def api_train_stop(request):
    body = await request.json()
    name = safe_name(body.get("run"))
    killed = False
    proc = _live.pop(name, None)
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            killed = True
        except Exception:
            pass
    pf = run_dir(name) / "train.pid"
    if not killed and pf.exists():
        try:
            os.killpg(os.getpgid(int(pf.read_text().strip())), signal.SIGTERM)
            killed = True
        except Exception:
            pass
    if killed:
        write_state(name, {"status": "stopping", "message": "stop requested"})
        return web.json_response({"ok": True, "run": name})
    return web.json_response({"error": "not running"}, status=404)


async def api_train_status(request):
    name = safe_name(request.query.get("run", ""))
    st = read_state(name)
    if st.get("status") in ACTIVE_STATUSES and not _proc_alive(name):
        # trainer died without writing a final state
        st = write_state(name, {"status": "error",
                                "error": "trainer exited unexpectedly — see train.log"})
    points = []
    pf = run_dir(name) / "progress.jsonl"
    if pf.exists():
        lines = pf.read_text().splitlines()[-400:]
        for line in lines:
            try:
                points.append(json.loads(line))
            except ValueError:
                pass
    log_tail = []
    lf = run_dir(name) / "train.log"
    if lf.exists():
        log_tail = lf.read_text(errors="ignore").splitlines()[-40:]
    return web.json_response({"state": st, "points": points, "log": log_tail})


async def api_train_list(request):
    runs = []
    if RUNS.exists():
        for d in sorted(RUNS.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            st = read_state(d.name)
            st["name"] = d.name
            st["running"] = _proc_alive(d.name)
            runs.append(st)
    datasets = sorted(f.name for f in UPLOADS.iterdir()) if UPLOADS.exists() else []
    return web.json_response({"runs": runs, "datasets": datasets})


async def api_train_upload(request):
    reader = await request.multipart()
    field = None
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "file":
            field = part
            break
    if field is None:
        return web.json_response({"error": "no file field"}, status=400)
    fname = safe_name(field.filename or "dataset.bin", "dataset.bin")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / fname
    size = 0
    with dest.open("wb") as f:
        while True:
            chunk = await field.read_chunk(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            f.write(chunk)
    return web.json_response({"ok": True, "name": fname, "size": size})


async def api_train_delete(request):
    body = await request.json()
    name = safe_name(body.get("run"))
    if name in _live and _live[name].poll() is None:
        return web.json_response({"error": "stop the run first"}, status=409)
    shutil.rmtree(run_dir(name), ignore_errors=True)
    _live.pop(name, None)
    return web.json_response({"ok": True})


def _zip_run(name):
    d = run_dir(name)
    st = read_state(name)
    out = Path("/tmp") / f"{name}-export.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for rel in ("train_config.json", "Modelfile"):
            if (d / rel).exists():
                z.write(d / rel, rel)
        gg = st.get("gguf_path")
        if gg and Path(gg).exists():
            z.write(Path(gg), Path(gg).name)
        ad = d / "adapter"
        if ad.exists():
            for f in sorted(ad.rglob("*")):
                if f.is_file() and "checkpoint" not in f.relative_to(ad).parts[0]:
                    z.write(f, "adapter/" + f.relative_to(ad).as_posix())
    return out


async def api_ds_push(request):
    body = await request.json()
    name = safe_name(body.get("name"), "dataset.bin")
    f = UPLOADS / name
    if not f.exists():
        return web.json_response({"error": "no such dataset"}, status=404)
    tok = env_vars().get("HF_TOKEN", "")
    if not tok:
        return web.json_response({"error": "no HF token — set it in Settings"}, status=400)
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        who = api.whoami()["name"]
        stem = Path(name).stem
        repo = f"{who}/{stem}-dataset"
        api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
        api.upload_file(path_or_fileobj=str(f), path_in_repo="dataset.jsonl",
                        repo_id=repo, repo_type="dataset")
        url = f"https://huggingface.co/datasets/{repo}"
        return web.json_response({"ok": True, "url": url})
    except ImportError:
        return web.json_response({"error": "huggingface_hub not installed"}, status=500)
    except Exception as e:
        return web.json_response({"error": str(e)[:300]}, status=500)


async def api_train_command(request):
    body = await request.json()
    name = safe_name(body.get("run"))
    cmd = body.get("cmd")
    if cmd not in ("save", "upload"):
        return web.json_response({"error": "cmd must be save|upload"}, status=400)
    write_state(name, {"command": cmd})
    return web.json_response({"ok": True, "run": name, "command": cmd})


async def api_train_download(request):
    name = safe_name(request.match_info["run"])
    if not run_dir(name).exists():
        return web.json_response({"error": "no such run"}, status=404)
    out = _zip_run(name)
    return web.FileResponse(out, headers={
        "Content-Disposition": f'attachment; filename="{name}-model.zip"'})


HF_BASE = "https://huggingface.co"
DATA_EXT = (".jsonl", ".json", ".csv", ".parquet", ".txt")


def _hf_headers(token=None):
    h = {"User-Agent": "kaggle-training-ai"}
    if token:
        h["Authorization"] = "Bearer " + token
    return h


def hf_dataset_info(repo, token=None):
    """Inspect an HF dataset repo via REST (no SDK needed). Green/red verdict."""
    repo = str(repo or "").strip()
    if repo.startswith("datasets/"):
        repo = repo[len("datasets/"):]
    repo = repo.strip("/")
    if not repo or "/" not in repo:
        return {"repo": repo, "ok": False, "files": [], "error": "expected format: user/dataset-name"}
    try:
        req = urllib.request.Request(HF_BASE + "/api/datasets/" + repo, headers=_hf_headers(token))
        with urllib.request.urlopen(req, timeout=20) as r:
            json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return {"repo": repo, "ok": False, "files": [], "error": "private/gated dataset — HF token required in Settings"}
        if e.code == 404:
            return {"repo": repo, "ok": False, "files": [], "error": "not found on HuggingFace"}
        return {"repo": repo, "ok": False, "files": [], "error": "HTTP %d" % e.code}
    except Exception as e:
        return {"repo": repo, "ok": False, "files": [], "error": str(e)[:200]}
    try:
        req2 = urllib.request.Request(HF_BASE + "/api/datasets/" + repo + "/tree/main?recursive=true",
                                      headers=_hf_headers(token))
        with urllib.request.urlopen(req2, timeout=25) as r:
            tree = json.loads(r.read().decode())
        files = [f["path"] for f in tree
                 if isinstance(f, dict) and f.get("type") == "file"
                 and f["path"].lower().endswith(DATA_EXT)]
    except Exception as e:
        return {"repo": repo, "ok": False, "files": [], "error": "tree listing: " + str(e)[:150]}
    if not files:
        return {"repo": repo, "ok": True, "files": [],
                "error": "repo exists but has no data files (jsonl/json/csv/parquet/txt)"}
    return {"repo": repo, "ok": True, "files": files[:20], "error": None}


async def api_ds_hf_copy(request):
    """Copy someone's public dataset to the user's own private HF dataset repo."""
    body = await request.json()
    repo = str(body.get("repo") or "").strip()
    tok = env_vars().get("HF_TOKEN", "")
    user = env_vars().get("HF_USER", "")
    if not tok:
        return web.json_response({"error": "no HF token — set it in Settings"}, status=400)
    if not user:
        return web.json_response({"error": "no HF username — set it in Settings"}, status=400)
    info = hf_dataset_info(repo, tok)
    if not info.get("ok") or not info.get("files"):
        return web.json_response({"error": info.get("error") or "no data files"}, status=400)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", repo.split("/")[-1]).strip("-") or "dataset"
    dest_repo = f"{user}/{stem}-dataset"
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        api.create_repo(dest_repo, repo_type="dataset", exist_ok=True, private=True)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            hdr = _hf_headers(tok)
            import urllib.request
            for path in info["files"]:
                url = f"{HF_BASE}/datasets/{repo}/resolve/main/{path}"
                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", path)
                local = Path(td) / safe
                req = urllib.request.Request(url, headers=hdr)
                with urllib.request.urlopen(req, timeout=180) as r, local.open("wb") as f:
                    shutil.copyfileobj(r, f)
                api.upload_file(path_or_fileobj=str(local), path_in_repo=path,
                                repo_id=dest_repo, repo_type="dataset",
                                commit_message=f"copy of {repo}")
        url = f"https://huggingface.co/datasets/{dest_repo}"
        return web.json_response({"ok": True, "url": url, "files": info["files"]})
    except ImportError:
        return web.json_response({"error": "huggingface_hub not installed"}, status=500)
    except Exception as e:
        return web.json_response({"error": str(e)[:300]}, status=500)


async def api_ds_hf_check(request):
    body = await request.json()
    repos = body.get("repos") or []
    if isinstance(repos, str):
        repos = [x.strip() for x in re.split(r"[,;\n]+", repos) if x.strip()]
    if not repos:
        return web.json_response({"error": "no repos given"}, status=400)
    tok = env_vars().get("HF_TOKEN", "")
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, lambda: [hf_dataset_info(r, tok) for r in repos[:20]])
    return web.json_response({"ok": True, "results": results})


def register_train_routes(app):
    app.router.add_post("/api/train/start", api_train_start)
    app.router.add_post("/api/train/stop", api_train_stop)
    app.router.add_get("/api/train/status", api_train_status)
    app.router.add_get("/api/train/list", api_train_list)
    app.router.add_post("/api/train/upload", api_train_upload)
    app.router.add_get("/api/train/download/{run}", api_train_download)
    app.router.add_post("/api/train/delete", api_train_delete)
    app.router.add_post("/api/train/command", api_train_command)
    app.router.add_post("/api/ds/push", api_ds_push)
    app.router.add_post("/api/ds/hf_check", api_ds_hf_check)
    app.router.add_post("/api/ds/hf_copy", api_ds_hf_copy)
