"""AI dataset generator: any OpenAI-compatible API → jsonl dataset → optional auto-train.

Jobs run as asyncio tasks inside the panel; progress in ~/.kaggle-panel/ai_gen/state.json.
The generated dataset lands in ~/.kaggle-panel/uploads/ai-<slug>.jsonl and can be used
directly by the training pipeline (file dataset).
The API key lives only in memory for the duration of the job — never written to disk.
"""
import asyncio
import hashlib
import json
import re
import secrets
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

from train_api import env_vars

BASE = Path.home() / ".kaggle-panel"
GEN = BASE / "ai_gen"
UPLOADS = BASE / "uploads"
DEFAULT_URL = "https://api.openai.com/v1/chat/completions"

_job = {"task": None, "stop": False}


def read_state():
    try:
        return json.loads((GEN / "state.json").read_text())
    except Exception:
        return {"status": "idle", "generated": 0, "total": 0, "message": "", "error": None}


def write_state(**kw):
    GEN.mkdir(parents=True, exist_ok=True)
    st = read_state()
    st.update(kw)
    st["updated_at"] = time.time()
    tmp = GEN / "state.tmp"
    tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False))
    tmp.replace(GEN / "state.json")
    return st


# ---------------------------------------------------------------- web reference
class _TextExtract(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1

    def handle_data(self, d):
        if not self.skip:
            self.parts.append(d)


def fetch_url_text(url, limit=4000):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (training-ai panel)"})
    raw = urllib.request.urlopen(req, timeout=20).read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", "ignore")
    p = _TextExtract()
    p.feed(text)
    out = re.sub(r"\s+", " ", " ".join(p.parts)).strip()
    return out[:limit]


def build_system_prompt(cfg, refs):
    parts = [
        "You are a high-quality training-dataset generator. "
        "You produce question/answer pairs suitable for supervised fine-tuning of an LLM.",
        "USER REQUIREMENTS (topic, style, language, special rules):",
        cfg["prompt"].strip() or "(no specific topic — make a general assistant dataset)",
    ]
    if cfg.get("seed"):
        parts.append("STYLE EXAMPLES (match this format and quality):\n" + cfg["seed"].strip()[:2000])
    if refs:
        parts.append("REFERENCE MATERIAL FETCHED FROM THE WEB (use as factual grounding, "
                     "rewrite in your own words, do not copy verbatim):\n" + refs[:12000])
    parts.append(
        "OUTPUT RULES: return ONLY a JSON array of objects "
        '{"instruction": "...", "output": "..."} — no markdown fences, no commentary, '
        "NO thinking or reasoning text before/after the array. "
        "Each output must be a complete, correct, self-contained answer.")
    return "\n\n".join(parts)


def normalize_url(u):
    if u.endswith("/chat/completions") or u.endswith("/api/chat"):
        return u
    if re.search(r"/v1/?$", u):
        return u.rstrip("/") + "/chat/completions"
    return u



def normalize_url(u):
    if u.endswith("/chat/completions") or u.endswith("/api/chat"):
        return u
    if re.search(r"/v1/?$", u):
        return u.rstrip("/") + "/chat/completions"
    return u


async def resolve_chat_url(sess, url, headers):
    """Accepts any provider base URL and returns the chat-completions endpoint.
    Probes /v1/models and /models so DeepSeek/OpenAI/OpenRouter/Groq/local-ollama
    all work no matter which form of the base URL the user pasted."""
    url = url.strip().rstrip("/")
    if url.endswith("/chat/completions") or url.endswith("/api/chat"):
        return url
    if re.search(r"/v1/?$", url):
        return url + "/chat/completions"
    probes = [(url + "/v1/models", url + "/v1/chat/completions"),
              (url + "/models", url + "/chat/completions")]
    if ":11434" in url:
        probes.insert(0, (url + "/api/tags", url + "/api/chat"))
    for probe, chat in probes:
        try:
            async with sess.get(probe, headers=headers) as r:
                if r.status == 200:
                    d = await r.json(content_type=None)
                    if isinstance(d, dict) and ("data" in d or "models" in d):
                        log(f"resolved chat url via {probe}: {chat}")
                        return chat
        except Exception:
            continue
    return url + "/v1/chat/completions"


async def fetch_model_ids(sess, url, headers):
    url = url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        url = url.rsplit("/chat/completions", 1)[0]
    elif url.endswith("/api/chat"):
        url = url.rsplit("/api/chat", 1)[0]
    cands = []
    if ":11434" in url:
        cands.append(url + "/api/tags")
    if re.search(r"/v1/?$", url):
        cands.append(url + "/models")
    else:
        cands += [url + "/v1/models", url + "/models"]
    for cand in cands:
        try:
            async with sess.get(cand, headers=headers) as r:
                if r.status != 200:
                    continue
                d = await r.json(content_type=None)
                items = d.get("data") if isinstance(d, dict) else None
                items = items or (d.get("models") if isinstance(d, dict) else None) or []
                ids = []
                for m in items:
                    if isinstance(m, dict):
                        ids.append(m.get("id") or m.get("name"))
                    elif isinstance(m, str):
                        ids.append(m)
                ids = [x for x in ids if x]
                if ids:
                    return sorted(set(ids))
        except Exception:
            continue
    return []


def _raw_snip(data):
    try:
        return json.dumps(data, ensure_ascii=False)[:250]
    except Exception:
        return str(data)[:250]


def extract_content(data):
    """Pull the text out of an OpenAI-compatible or ollama-native response.
    Surfaces provider errors (OpenRouter returns 200 + error object sometimes)
    and reasoning-only responses with a clear hint."""
    if isinstance(data, dict) and data.get("error"):
        e = data["error"]
        msg = e.get("message") if isinstance(e, dict) else str(e)
        code = e.get("code") if isinstance(e, dict) else None
        raise ValueError(f"API error{(' [' + str(code) + ']') if code else ''}: {msg}")
    ch = data.get("choices")
    if ch:
        msg0 = (ch[0] or {}) if isinstance(ch[0], dict) else {}
        m0 = msg0.get("message") or {}
        c = m0.get("content")
        if isinstance(c, str) and c.strip():
            return c
        reasoning = m0.get("reasoning_content") or m0.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            if msg0.get("finish_reason") == "length":
                # the model burned the whole token budget mid-thinking — nothing to mine
                raise ValueError("reasoning model spent the whole token budget on thinking "
                                 "(finish_reason=length, answer never written) — raise 'Max tokens / batch' "
                                 "to 12000-16000 or use a non-reasoning model")
            # thinking models put the final JSON inside the reasoning text — mine it
            return reasoning
        if msg0.get("finish_reason") == "error":
            raise ValueError("provider failed to generate — try another model on this API. raw: "
                             + _raw_snip(data))
        raise ValueError("empty message content (finish_reason=%s) — raw: %s"
                         % (msg0.get("finish_reason"), _raw_snip(data)))
    msg = data.get("message")  # ollama native /api/chat
    if isinstance(msg, dict) and msg.get("content"):
        return msg["content"]
    raise ValueError("response has no message content — raw response: " + _raw_snip(data))


def parse_items(text):
    """Extract sample dicts even when the model wraps JSON in thinking/prose."""
    text = text.strip()
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "[":
            try:
                val, _end = dec.raw_decode(text, i)
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    return [x for x in val if isinstance(x, dict)]
            except ValueError:
                continue
    items = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{") and line.endswith("}"):
            try:
                items.append(json.loads(line))
            except ValueError:
                pass
    return items


async def generate_job(cfg):
    slug = cfg["slug"]
    UPLOADS.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    ds_path = UPLOADS / f"ai-{slug}.jsonl"

    write_state(status="running", generated=0, total=cfg["count"], error=None,
                dataset_file=None, message="preparing…", slug=slug,
                started_at=time.time(), last_samples=[])

    refs = ""
    for u in cfg.get("ref_urls", [])[:5]:
        try:
            write_state(message=f"fetching web reference: {u[:60]}")
            refs += f"\n--- {u} ---\n" + await asyncio.to_thread(fetch_url_text, u)
        except Exception as e:
            refs += f"\n({u}: fetch failed: {e})\n"

    headers0 = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers0["Authorization"] = "Bearer " + cfg["api_key"]
    async with ClientSession(timeout=ClientTimeout(total=30)) as s0:
        chat_url = await resolve_chat_url(s0, cfg["api_url"], headers0)
    write_state(resolved_url=chat_url, message="API resolved: " + chat_url)
    log("chat url: " + chat_url)

    system = build_system_prompt(cfg, refs)
    seen = set()
    lines = []
    written = 0
    batch_errors = 0
    dup_streak = 0

    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = "Bearer " + cfg["api_key"]

    try:
        async with ClientSession(timeout=ClientTimeout(total=240)) as sess:
            headers = headers0
            api_endpoint = chat_url
            while written < cfg["count"] and not _job["stop"]:
                need = min(cfg["batch"], cfg["count"] - written)
                user = (f"Generate exactly {need} new samples. Total already generated: {written}.")
                if lines:
                    recent = [json.loads(x)["instruction"] for x in lines[-8:]]
                    user += "\nDo not repeat these existing instructions:\n" + "\n".join(recent)
                write_state(message=f"calling {cfg['model']} for {need} samples… "
                                    "(reasoning models can take a few minutes)")
                payload = {"model": cfg["model"],
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}],
                           "temperature": cfg["temperature"],
                           "max_tokens": cfg["max_tokens"], "stream": False}
                if "openrouter.ai" in cfg["api_url"]:
                    payload["reasoning"] = {"enabled": False}
                try:
                    async with sess.post(
                        api_endpoint, json=payload,
                        headers=headers) as r:
                        if r.status != 200:
                            body = (await r.text())[:200]
                            raise RuntimeError(f"API {r.status}: {body}")
                        data = await r.json()
                    content = extract_content(data)
                except Exception as e:
                    emsg = str(e)
                    if "token budget on thinking" in emsg and cfg["max_tokens"] < 32000:
                        cfg["max_tokens"] = min(cfg["max_tokens"] * 2, 32000)
                        write_state(message=f"reasoning ate the budget — retrying batch with max_tokens={cfg['max_tokens']}")
                        log(f"escalating max_tokens to {cfg['max_tokens']}")
                        continue
                    batch_errors += 1
                    write_state(message=f"batch error ({batch_errors}): {emsg[:160]} — retrying")
                    if batch_errors >= 5:
                        if written > 0:
                            log(f"API keeps failing — finishing with {written} samples")
                            break
                        raise RuntimeError(f"API keeps failing: {e}")
                    await asyncio.sleep(3 * batch_errors)  # backoff
                    continue

                items = parse_items(content)
                if not items:
                    batch_errors += 1
                    write_state(message=f"unparsable batch ({batch_errors}) — retrying")
                    if batch_errors >= 5:
                        raise RuntimeError("generator keeps returning unparsable output")
                    continue
                batch_errors = 0
                new = 0
                for it in items:
                    ins = str(it.get("instruction") or it.get("prompt") or "").strip()
                    out = str(it.get("output") or it.get("response") or it.get("completion") or "").strip()
                    if not ins or not out:
                        continue
                    h = hashlib.md5((ins + out).lower().encode()).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    lines.append(json.dumps({"instruction": ins, "output": out},
                                            ensure_ascii=False))
                    written += 1
                    new += 1
                    if written >= cfg["count"]:
                        break
                if new == 0:
                    dup_streak += 1
                    write_state(message=f"batch had only duplicates ({dup_streak})")
                    if dup_streak >= 5:
                        if written > 0:
                            log("only duplicates — finishing with what we have")
                            break
                        raise RuntimeError("generator produces only duplicates — "
                                           "raise temperature or change the prompt")
                else:
                    dup_streak = 0
                ds_path.write_text("\n".join(lines) + "\n")
                last = [json.loads(x) for x in lines[-2:]]
                write_state(generated=written, dataset_file=str(ds_path),
                            message=f"batch ok (+{new})", last_samples=last)
                log_line = f"[ai-gen {time.strftime('%H:%M:%S')}] {written}/{cfg['count']} (+{new})"
                print(log_line, flush=True)
                await asyncio.sleep(0.5)

        if _job["stop"]:
            write_state(status="stopped", message=f"stopped at {written}/{cfg['count']}",
                        dataset_file=str(ds_path) if written else None)
            return

        if written == 0:
            raise RuntimeError("0 samples generated")

        write_state(status="done", message=f"dataset ready: {ds_path.name} ({written} samples)",
                    dataset_file=str(ds_path))
        print(f"[ai-gen] done: {ds_path}", flush=True)

        if cfg.get("save_ds_hf"):
            save_ds_hf(cfg, run_state_writer=write_state, ds_path=ds_path, count=written)

        if cfg.get("auto_train"):
            write_state(message="dataset ready → launching training…")
            from train_api import launch_training
            body = dict(cfg.get("train") or {})
            body["run_name"] = body.get("run_name") or slug
            body["dataset_type"] = "file"
            body["dataset_value"] = str(ds_path)
            ok, resp = launch_training(body)
            write_state(auto_train=resp)
    except Exception as e:
        write_state(status="error", error=str(e)[:400],
                    message="generation failed", dataset_file=str(ds_path) if written else None)


def save_ds_hf(cfg, run_state_writer, ds_path, count):
    """Push the generated dataset to a private HF dataset repo. Never raises."""
    tok = cfg.get("hf_token") or os.environ.get("HF_TOKEN", "")
    user = cfg.get("hf_user") or os.environ.get("HF_USER", "")
    if not tok:
        run_state_writer(ds_hf_error="no HF token — dataset kept local")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        repo_user = user or (api.whoami()["name"] if not user else user)
        repo = f"{repo_user}/{cfg['slug']}-dataset"
        api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
        api.upload_file(path_or_fileobj=str(ds_path), path_in_repo="dataset.jsonl",
                        repo_id=repo, repo_type="dataset",
                        commit_message=f"{count} samples — generated by training-ai")
        url = f"https://huggingface.co/datasets/{repo}"
        run_state_writer(ds_hf_url=url, message=f"dataset on HF ✓ {url}")
        print(f"[ai-gen] dataset uploaded: {url}", flush=True)
    except Exception as e:
        run_state_writer(ds_hf_error=str(e)[:300])
        print(f"[ai-gen] HF dataset upload failed: {e}", flush=True)


async def api_ai_start(request):
    body = await request.json()
    task = _job.get("task")
    if task and not task.done():
        return web.json_response({"error": "generation_already_running"}, status=409)

    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return web.json_response({"error": "prompt required"}, status=400)
    api_url = normalize_url(str(body.get("api_url") or DEFAULT_URL).strip() or DEFAULT_URL)
    if not re.match(r"^https?://", api_url):
        return web.json_response({"error": "bad api_url"}, status=400)

    slug = re.sub(r"[^a-z0-9-]+", "-", str(body.get("dataset_name") or "ds").lower()).strip("-")[:40]
    slug = slug or f"ds-{secrets.token_hex(2)}"

    cfg = {
        "slug": slug,
        "api_url": api_url,
        "api_key": str(body.get("api_key") or "").strip(),
        "model": str(body.get("model") or "gpt-4o-mini").strip(),
        "prompt": prompt,
        "seed": str(body.get("seed") or ""),
        "ref_urls": [u.strip() for u in re.split(r"[,\s]+", str(body.get("ref_urls") or "")) if u.strip().startswith("http")][:5],
        "count": max(1, min(int(body.get("count") or 50), 5000)),
        "batch": max(1, min(int(body.get("batch") or 10), 50)),
        "temperature": min(max(float(body.get("temperature") or 0.9), 0), 2),
        "max_tokens": max(256, min(int(body.get("max_tokens") or 8000), 32000)),
        "auto_train": bool(body.get("auto_train", False)),
        "save_ds_hf": bool(body.get("save_ds_hf", True)),
        "hf_token": str(body.get("hf_token") or "").strip(),
        "hf_user": env_vars().get("HF_USER", ""),
        "train": body.get("train") or {},
    }
    _job["stop"] = False
    _job["task"] = asyncio.create_task(generate_job(cfg))
    return web.json_response({"ok": True, "slug": slug})


async def api_ai_stop(request):
    _job["stop"] = True
    write_state(message="stop requested")
    return web.json_response({"ok": True})


async def api_ai_status(request):
    return web.json_response(read_state())


async def api_ai_test(request):
    body = await request.json()
    api_url = str(body.get("api_url") or DEFAULT_URL).strip()
    headers = {"Content-Type": "application/json"}
    if body.get("api_key"):
        headers["Authorization"] = "Bearer " + str(body["api_key"]).strip()
    try:
        async with ClientSession(timeout=ClientTimeout(total=60)) as sess:
            chat_url = await resolve_chat_url(sess, api_url, headers)
            models = await fetch_model_ids(sess, api_url, headers)
            async with sess.post(chat_url, headers=headers, json={
                "model": body.get("model") or "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                "max_tokens": 10, "stream": False}) as r:
                text = await r.text()
                if r.status != 200:
                    return web.json_response({"ok": False, "error": f"HTTP {r.status}: {text[:200]}"})
                try:
                    data = json.loads(text)
                except ValueError:
                    return web.json_response({"ok": False, "error":
                        "response is not JSON — URL must point to a chat completions endpoint"}, status=200)
                try:
                    reply = extract_content(data) or ""
                except ValueError as e:
                    return web.json_response({"ok": False, "error": str(e), "resolved": chat_url, "models": models[:20]})
                return web.json_response({"ok": True, "reply": reply[:120], "resolved": chat_url, "models": models[:20]})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)[:250]})


async def api_ai_models(request):
    body = await request.json()
    headers = {}
    if body.get("api_key"):
        headers["Authorization"] = "Bearer " + str(body["api_key"]).strip()
    try:
        async with ClientSession(timeout=ClientTimeout(total=30)) as sess:
            models = await fetch_model_ids(sess, str(body.get("api_url") or "").strip(), headers)
        return web.json_response({"ok": True, "models": models[:200]})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)[:250]})


def register_ai_routes(app):
    app.router.add_post("/api/ai/start", api_ai_start)
    app.router.add_post("/api/ai/stop", api_ai_stop)
    app.router.add_get("/api/ai/status", api_ai_status)
    app.router.add_post("/api/ai/test", api_ai_test)
    app.router.add_post("/api/ai/models", api_ai_models)
