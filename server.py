#!/usr/bin/env python3
"""Kaggle Web Panel — remote control panel for a Kaggle machine.

Serves the web UI, a shared PTY terminal over WebSocket, stats and
action APIs. Designed to be exposed through a Cloudflare tunnel.
"""
import asyncio
import base64
import fcntl
import json
import os
import pty
import re
import secrets
import shutil
import signal
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

from aiohttp import WSMsgType, web
import aiohttp
import psutil

import train_api

HOME = Path.home()
BASE = HOME / ".kaggle-panel"
BIN_DIR = BASE / "bin"
SCRIPTS = BASE / "scripts"
ENV_FILE = BASE / ".env"
TUNNEL_LOG = BASE / "tunnel.log"
OLLAMA_LOG = BASE / "ollama.log"
PANEL_LOG = BASE / "panel.log"
STATIC = Path(__file__).resolve().parent / "static"

PORT = int(os.environ.get("PANEL_PORT", "7860"))
HOST = os.environ.get("PANEL_HOST", "127.0.0.1")
AUTH_TOKEN = os.environ.get("PANEL_PASSWORD") or ""  # finalized in main()

REPO_DIR = Path(__file__).resolve().parent
RUNS = BASE / "runs"
UPLOADS = BASE / "uploads"
TRAIN_TOOLS = BASE / "tools"

TUNNEL_SH = r"""#!/usr/bin/env bash
# Managed by kaggle-web panel. Usage: tunnel.sh {start|stop|restart}
BASE="$HOME/.kaggle-panel"
LOG="$BASE/tunnel.log"
set -a; [ -f "$BASE/.env" ] && source "$BASE/.env"; set +a
PORT="${PANEL_PORT:-7860}"
BIN="${CLOUDFLARED:-}"
[ -x "$BIN" ] || BIN="$BASE/bin/cloudflared"
[ -x "$BIN" ] || BIN="$(command -v cloudflared || true)"
start() {
  if pgrep -f "cloudflared tunnel" >/dev/null 2>&1; then
    echo "[tunnel] already running (pid $(pgrep -f 'cloudflared tunnel' | head -1))"
    return 0
  fi
  if [ ! -x "$BIN" ] && [ -z "$(command -v cloudflared || true)" ]; then
    echo "[tunnel] cloudflared not found — install it first"
    return 1
  fi
  echo "[tunnel] starting..."
  if [ -n "$CF_TOKEN" ]; then
    echo "[tunnel] mode: named tunnel (token). Domain is managed in Cloudflare Zero Trust dashboard."
    nohup "$BIN" tunnel --protocol http2 --no-autoupdate run --token "$CF_TOKEN" >"$LOG" 2>&1 &
  elif [ -n "$TUNNEL_UUID" ] && [ -n "$CF_DOMAIN" ] && [ -f "$BASE/tunnel-credentials.json" ]; then
    echo "[tunnel] mode: config file, hostname $CF_DOMAIN -> http://localhost:$PORT"
    nohup "$BIN" tunnel --protocol http2 --no-autoupdate --config "$BASE/config.yml" run >"$LOG" 2>&1 &
  else
    echo "[tunnel] mode: quick (random *.trycloudflare.com URL)"
    nohup "$BIN" tunnel --protocol http2 --no-autoupdate --url "http://127.0.0.1:$PORT" >"$LOG" 2>&1 &
  fi
  disown
  echo "[tunnel] started (pid $!). Log: $LOG"
}
stop() {
  if pkill -f "cloudflared tunnel" 2>/dev/null; then echo "[tunnel] stopped"; else echo "[tunnel] not running"; fi
}
case "$1" in
  start) start ;;
  stop) stop ;;
  restart) stop; sleep 1; start ;;
  *) echo "usage: tunnel.sh {start|stop|restart}" ;;
esac
"""

RESTART_SH = r"""#!/usr/bin/env bash
# Managed by kaggle-web panel. Restarts the panel server.
BASE="$HOME/.kaggle-panel"
set -a; [ -f "$BASE/.env" ] && source "$BASE/.env"; set +a
REPO="{repo}"
PY="{py}"
echo "[panel] restarting..."
pkill -f "python.*server\.py" 2>/dev/null
sleep 1
cd "$REPO" || exit 1
nohup "$PY" server.py >"$BASE/panel.log" 2>&1 &
disown
echo "[panel] restarted on port ${{PANEL_PORT:-7860}}"
""".format(repo=str(REPO_DIR), py=sys.executable)

UPDATE_SH = r"""#!/usr/bin/env bash
# Managed by kaggle-web panel. Pulls the latest panel from GitHub and restarts.
BASE="$HOME/.kaggle-panel"
REPO="{repo}"
echo "[panel] updating from git..."
cd "$REPO" || {{ echo "[panel] repo dir not found"; exit 1; }}
if git pull --ff-only; then
  bash "$BASE/scripts/restart.sh"
else
  echo "[panel] git pull failed — panel not restarted"
fi
""".format(repo=str(REPO_DIR))


def sh_quote(v) -> str:
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("$", "\\$").replace("`", "\\`")
    return '"%s"' % s


BASHRC = r'''# Managed by kaggle-web panel
[ -f /etc/bash.bashrc ] && source /etc/bash.bashrc
[ -f ~/.bashrc ] && source ~/.bashrc
PS1='\[\e[1;35m\]⚡ panel\[\e[0m\] \[\e[1;34m\]\w\[\e[0m\] ❯ '
'''


def setup_files():
    for d in (BASE, BIN_DIR, SCRIPTS, RUNS, UPLOADS, TRAIN_TOOLS):
        d.mkdir(parents=True, exist_ok=True)
    (SCRIPTS / "tunnel.sh").write_text(TUNNEL_SH)
    (SCRIPTS / "restart.sh").write_text(RESTART_SH)
    (SCRIPTS / "update.sh").write_text(UPDATE_SH)
    (SCRIPTS / "bashrc").write_text(BASHRC)
    for name in ("tunnel.sh", "restart.sh", "update.sh"):
        os.chmod(SCRIPTS / name, 0o755)


def load_env_file():
    """Merge .env values into process env (panel port may change after restart)."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


# ---------------------------------------------------------------- PTY shell
class Shell:
    def __init__(self):
        self.pid = None
        self.fd = None
        self.clients = set()
        self.loop = None
        self.history = bytearray()
        self.seq = 0          # total bytes ever written to history
        self.trimmed = 0      # bytes dropped from the front of history
        self.sid = None       # session id (changes on every respawn)

    def ensure(self):
        if self.pid:
            return
        env = os.environ.copy()
        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"
        env["LANG"] = "C.UTF-8"
        self.pid, self.fd = pty.fork()
        if self.pid == 0:  # child
            try:
                os.execvpe("bash", ["bash", "--rcfile", str(SCRIPTS / "bashrc"), "-i"], env)
            except Exception:
                os._exit(1)
        # fresh session: drop the previous session's scrollback
        self.sid = secrets.token_hex(4)
        self.history.clear()
        self.seq = 0
        self.trimmed = 0
        self.resize(120, 32)
        try:
            self.loop.add_reader(self.fd, self._on_read)
        except Exception:
            pass

    def _on_read(self):
        try:
            data = os.read(self.fd, 65536)
        except (OSError, BlockingIOError):
            data = b""
        if not data:
            self._on_exit()
            return
        self.history.extend(data)
        self.seq += len(data)
        if len(self.history) > 400_000:
            drop = len(self.history) - 400_000
            del self.history[:drop]
            self.trimmed += drop
        msg = json.dumps({"type": "out", "sid": self.sid, "seq": self.seq,
                          "data": base64.b64encode(data).decode()})
        for ws in list(self.clients):
            asyncio.create_task(self._safe_send(ws, msg))

    async def replay(self, ws):
        """Send buffered terminal output to a newly attached client."""
        if self.history:
            msg = json.dumps({"type": "out", "sid": self.sid, "seq": self.seq,
                              "data": base64.b64encode(bytes(self.history)).decode()})
            await self._safe_send(ws, msg)

    def snapshot(self, offset: int, client_sid: str):
        """Output chunk for the HTTP fallback channel."""
        sid = self.sid or ""
        reset = client_sid != sid
        start = 0 if reset else max(offset, self.trimmed)
        data = bytes(self.history[start - self.trimmed:]) if start < self.seq else b""
        return {"sid": sid, "seq": self.seq, "reset": reset,
                "data": base64.b64encode(data).decode()}

    async def _safe_send(self, ws, msg):
        try:
            await ws.send_str(msg)
        except Exception:
            self.clients.discard(ws)

    def _on_exit(self):
        if self.fd is not None:
            try:
                self.loop.remove_reader(self.fd)
                os.close(self.fd)
            except Exception:
                pass
        self.fd = None
        self.pid = None
        for ws in list(self.clients):
            asyncio.create_task(
                self._safe_send(ws, json.dumps({"type": "exit"}))
            )

    def write(self, data: str):
        self.ensure()
        try:
            os.write(self.fd, data.encode())
        except OSError:
            self._on_exit()

    def resize(self, cols: int, rows: int):
        if not self.fd:
            return
        try:
            winsize = struct.pack("HHHH", max(rows, 2), max(cols, 2), 0, 0)
            fcntl.ioctl(self.fd, termios.TIOCSWINSZ, winsize)
            os.kill(self.pid, signal.SIGWINCH)
        except Exception:
            pass


shell = Shell()

# ---------------------------------------------------------------- helpers
_login_fails = {}


def auth_ok(request) -> bool:
    tok = (
        request.cookies.get("panel_auth")
        or request.headers.get("X-Auth")
        or request.query.get("auth")
    )
    if not tok:
        # OpenAI-compatible clients send the key as a bearer token
        authz = request.headers.get("Authorization", "")
        if authz.startswith("Bearer "):
            tok = authz[7:].strip()
    return bool(tok) and secrets.compare_digest(str(tok), AUTH_TOKEN)


@web.middleware
async def auth_middleware(request, handler):
    public = request.path == "/api/login"
    guarded = (request.path.startswith("/api/")
               or request.path.startswith("/ws")
               or request.path.startswith("/ollama"))
    if guarded and not public:
        if not auth_ok(request):
            return web.json_response({"error": "unauthorized"}, status=401)
    resp = await handler(request)
    if request.path.startswith("/static") or request.path == "/":
        resp.headers.setdefault("Cache-Control", "no-cache")
    return resp


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,"
             "memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
        ).stdout.strip()
    except Exception:
        return []
    gpus = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            try:
                gpus.append({
                    "name": p[0], "util": float(p[1]), "mem_used": float(p[2]),
                    "mem_total": float(p[3]), "temp": float(p[4]),
                })
            except ValueError:
                continue
    return gpus


_net_prev = {"t": 0.0, "sent": 0, "recv": 0}


def collect_stats():
    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    du = psutil.disk_usage("/")
    net = psutil.net_io_counters()
    now = time.time()
    dt = now - _net_prev["t"] if _net_prev["t"] else 0
    if dt > 0:
        net_up = max((net.bytes_sent - _net_prev["sent"]) / dt, 0)
        net_down = max((net.bytes_recv - _net_prev["recv"]) / dt, 0)
    else:
        net_up = net_down = 0.0
    _net_prev.update(t=now, sent=net.bytes_sent, recv=net.bytes_recv)
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = 0.0
    return {
        "cpu": cpu,
        "cpu_cores": psutil.cpu_count(logical=True),
        "ram_used": vm.used, "ram_total": vm.total, "ram_pct": vm.percent,
        "disk_used": du.used, "disk_total": du.total, "disk_pct": du.percent,
        "gpus": gpu_info(),
        "net_up": net_up, "net_down": net_down,
        "load": [round(load1, 2), round(load5, 2), round(load15, 2)],
        "uptime": time.time() - psutil.boot_time(),
        "host": os.uname().nodename,
    }


def read_env_vars():
    vals = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                vals[k.strip()] = v.strip().strip('"')
    return vals


def write_env_file(data: dict):
    old = read_env_vars()

    def pick(new_key, old_key):
        """New value if the key was sent explicitly, otherwise keep the old one."""
        if new_key in data:
            return str(data[new_key] or "").strip()
        return old.get(old_key, "")

    port = str(data.get("panel_port") or old.get("PANEL_PORT") or PORT)
    password = (data.get("panel_password") or "").strip() or old.get("PANEL_PASSWORD", "")
    lines = [
        "# kaggle-web panel settings (mini .env)",
        "PANEL_PORT=" + sh_quote(port),
    ]
    if password:
        lines.append("PANEL_PASSWORD=" + sh_quote(password))
    lines.append("CF_TOKEN=" + sh_quote(pick("cf_token", "CF_TOKEN")))
    lines.append("CF_DOMAIN=" + sh_quote(pick("cf_domain", "CF_DOMAIN")))
    lines.append("TUNNEL_UUID=" + sh_quote(pick("tunnel_uuid", "TUNNEL_UUID")))
    lines.append("HF_TOKEN=" + sh_quote(pick("hf_token", "HF_TOKEN")))
    ENV_FILE.write_text("\n".join(lines) + "\n")

    uuid = pick("tunnel_uuid", "TUNNEL_UUID")
    domain = pick("cf_domain", "CF_DOMAIN")
    creds = (data.get("credentials") or "").strip()
    if uuid and domain and creds:
        (BASE / "tunnel-credentials.json").write_text(creds)
        (BASE / "config.yml").write_text(
            "tunnel: %s\n"
            "credentials-file: %s\n"
            "ingress:\n"
            "  - hostname: %s\n"
            "    service: http://localhost:%s\n"
            "  - service: http_status:404\n" % (uuid, BASE / "tunnel-credentials.json", domain, port)
        )


def collect_status():
    ollama_bin = shutil.which("ollama") or os.path.exists("/usr/local/bin/ollama")
    zstd_bin = bool(shutil.which("zstd"))
    cf_bin = (BIN_DIR / "cloudflared").exists() or bool(shutil.which("cloudflared"))

    def pgrep(pat):
        try:
            return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0
        except Exception:
            return False

    tunnel_on = pgrep("cloudflared tunnel")
    serve_on = pgrep("ollama serve")
    url = None
    try:
        m = re.findall(r"https://(?!api\.)[a-zA-Z0-9-]+\.trycloudflare\.com", TUNNEL_LOG.read_text(errors="ignore"))
        url = m[-1] if m else None
    except OSError:
        pass
    models = []
    if ollama_bin and serve_on:
        try:
            out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=8).stdout
            for line in out.splitlines()[1:]:
                parts = line.split()
                if parts:
                    models.append({"name": parts[0], "size": parts[2] if len(parts) > 2 else ""})
        except Exception:
            pass
    return {
        "ollama_installed": bool(ollama_bin), "zstd_installed": zstd_bin,
        "cloudflared_installed": bool(cf_bin), "tunnel_running": tunnel_on,
        "tunnel_url": url, "tunnel_mode": ("token" if read_env_vars().get("CF_TOKEN") else
                                           ("config" if read_env_vars().get("TUNNEL_UUID") else "quick")),
        "ollama_serve": serve_on, "models": models,
    }


# ---------------------------------------------------------------- routes
async def index(request):
    return web.FileResponse(STATIC / "index.html",
                            headers={"Cache-Control": "no-store"})


async def api_login(request):
    body = await request.json()
    ip = request.remote or "?"
    fails = _login_fails.get(ip, {"n": 0, "t": 0.0})
    if fails["n"] >= 5 and time.time() - fails["t"] < 60:
        return web.json_response({"error": "too_many_attempts"}, status=429)
    if secrets.compare_digest(str(body.get("password", "")), AUTH_TOKEN):
        _login_fails.pop(ip, None)
        resp = web.json_response({"ok": True, "token": AUTH_TOKEN})
        resp.set_cookie("panel_auth", AUTH_TOKEN, max_age=7 * 24 * 3600,
                        httponly=True, samesite="Lax")
        return resp
    fails["n"] += 1
    fails["t"] = time.time()
    _login_fails[ip] = fails
    return web.json_response({"error": "wrong_password"}, status=401)


async def api_verify(request):
    return web.json_response({"ok": True})


async def api_stats(request):
    return web.json_response(collect_stats())


async def api_status(request):
    return web.json_response(collect_status())


async def api_env_get(request):
    v = read_env_vars()
    return web.json_response({
        "panel_port": v.get("PANEL_PORT") or str(PORT),
        "has_password": bool(v.get("PANEL_PASSWORD")),
        "cf_token_set": bool(v.get("CF_TOKEN")),
        "cf_domain": v.get("CF_DOMAIN") or "",
        "tunnel_uuid": v.get("TUNNEL_UUID") or "",
        "hf_token_set": bool(v.get("HF_TOKEN")),
        "mode": "token" if v.get("CF_TOKEN") else ("config" if v.get("TUNNEL_UUID") else "quick"),
    })


async def api_env_post(request):
    global AUTH_TOKEN
    data = await request.json()
    write_env_file(data)
    new_pass = (data.get("panel_password") or "").strip()
    if new_pass:
        AUTH_TOKEN = new_pass
    resp = web.json_response({"ok": True, "token": AUTH_TOKEN})
    resp.set_cookie("panel_auth", AUTH_TOKEN, max_age=7 * 24 * 3600,
                    httponly=True, samesite="Lax")
    return resp


async def api_inject(request):
    data = await request.json()
    cmd = str(data.get("cmd", ""))[:20000]
    shell.write(cmd + "\n")
    return web.json_response({"ok": True})


async def api_restart(request):
    subprocess.Popen(["bash", str(SCRIPTS / "restart.sh")],
                     start_new_session=True,
                     stdout=open(PANEL_LOG, "ab"), stderr=subprocess.STDOUT)
    async def _die():
        await asyncio.sleep(1.2)
        os._exit(0)
    asyncio.create_task(_die())
    return web.json_response({"ok": True})


async def api_input(request):
    data = await request.json()
    if "cols" in data or "rows" in data:
        try:
            shell.resize(int(data.get("cols", 80)), int(data.get("rows", 24)))
        except (TypeError, ValueError):
            pass
        return web.json_response({"ok": True})
    chunk = str(data.get("data", ""))[:8192]
    if chunk:
        shell.write(chunk)
    return web.json_response({"ok": True})


async def api_output(request):
    shell.ensure()  # spawn the shell on first poll too (WS may be blocked)
    try:
        offset = int(request.query.get("offset", "0"))
    except ValueError:
        offset = 0
    client_sid = request.query.get("sid", "")
    return web.json_response(shell.snapshot(offset, client_sid))


async def ws_term(request):
    ws = web.WebSocketResponse(heartbeat=25, max_msg_size=1 << 22)
    await ws.prepare(request)
    shell.loop = asyncio.get_running_loop()
    shell.ensure()  # always have a live shell — prompt shows even after server restart
    shell.clients.add(ws)
    await shell.replay(ws)
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                d = json.loads(msg.data)
            except ValueError:
                continue
            if d.get("type") == "input":
                shell.write(str(d.get("data", ""))[:8192])
            elif d.get("type") == "resize":
                shell.resize(int(d.get("cols", 80)), int(d.get("rows", 24)))
    finally:
        shell.clients.discard(ws)
    return ws


# ---------------------------------------------------------------- ollama proxy
http_client = None


async def ollama_proxy(request):
    """Reverse-proxy /ollama/* -> 127.0.0.1:11434/* so one domain serves both
    the panel and the Ollama API (password-protected, streaming-safe)."""
    tail = request.match_info.get("path", "")
    url = "http://127.0.0.1:11434/" + tail
    if request.query_string:
        url += "?" + request.query_string
    body = await request.read() if request.can_read_body else None
    headers = {k.lower(): v for k, v in request.headers.items()
               if k.lower() in ("content-type", "accept", "authorization")}
    # force plain streaming: if Ollama gzips the response, its Go server
    # buffers SSE chunks and the client sees nothing until the very end
    headers["accept-encoding"] = "identity"
    try:
        async with http_client.request(request.method, url, data=body,
                                       headers=headers) as up:
            resp = web.StreamResponse(
                status=up.status,
                headers={"Content-Type": up.headers.get("Content-Type",
                                                        "application/octet-stream"),
                         "Cache-Control": "no-cache",
                         "X-Accel-Buffering": "no"})
            await resp.prepare(request)
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp
    except (OSError, asyncio.TimeoutError):
        return web.json_response(
            {"error": "ollama is not running — start it in the panel"},
            status=502)


def make_app():
    app = web.Application(middlewares=[auth_middleware],
                          client_max_size=2 * 1024 * 1024 * 1024)
    app.router.add_get("/", index)
    app.router.add_static("/static/", STATIC)
    app.router.add_post("/api/login", api_login)
    app.router.add_get("/api/verify", api_verify)
    app.router.add_get("/api/stats", api_stats)
    app.router.add_get("/api/status", api_status)
    app.router.add_get("/api/env", api_env_get)
    app.router.add_post("/api/env", api_env_post)
    app.router.add_post("/api/inject", api_inject)
    app.router.add_post("/api/input", api_input)
    app.router.add_get("/api/output", api_output)
    app.router.add_post("/api/restart", api_restart)
    app.router.add_get("/ws/term", ws_term)
    app.router.add_route("*", "/ollama/{path:.*}", ollama_proxy)
    app.router.add_route("*", "/ollama", ollama_proxy)
    train_api.register_train_routes(app)
    return app


def main():
    setup_files()
    load_env_file()
    global PORT, AUTH_TOKEN
    PORT = int(os.environ.get("PANEL_PORT", PORT))
    AUTH_TOKEN = os.environ.get("PANEL_PASSWORD") or secrets.token_urlsafe(9)

    async def _on_start(app):
        global http_client
        shell.loop = asyncio.get_running_loop()
        http_client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None))

    print("=" * 58)
    print("  ⚡ Kaggle Web Panel")
    print("  listening on  http://%s:%d" % (HOST, PORT))
    print("  password      %s" % AUTH_TOKEN)
    print("=" * 58)
    app = make_app()
    app.on_startup.append(_on_start)
    web.run_app(app, host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
