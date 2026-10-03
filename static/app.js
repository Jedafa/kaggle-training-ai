/* ============ Kaggle Training AI — frontend ============ */
"use strict";

const I18N = {
  en: {
    subtitle: "train · serve · ship your own model", password: "Password", login: "Sign in",
    wrong_pass: "Wrong password", login_hint: "The password is printed in the Kaggle notebook output.",
    nav_title: "workspace", nav_training: "Training", nav_terminal: "Terminal", nav_models: "Models", nav_settings: "Settings",
    ds_title: "Dataset", ds_sub: "jsonl / json / csv / parquet / txt — instruction+output, messages[] or raw text",
    ds_link: "Link", ds_file: "File", ds_choose: "Choose file…",
    ds_hf_check: "Check repos", ds_hf_hint: "one repo per line · green = ready · red = excluded",
    ds_hf_need: "Add at least one repo", ds_hf_none_ready: "No ready datasets — fix red ones or remove them",
    ds_copy: "copy to my HF",
    cfg_title: "Model & run", cfg_sub: "QLoRA 4-bit — runs on T4 · adapter → merge → GGUF → Ollama → HuggingFace",
    f_base: "Base model (HuggingFace id or local path)", f_runname: "Run / model name",
    f_epochs: "Epochs", f_lr: "Learning rate", f_batch: "Batch size", f_accum: "Grad accum",
    f_seq: "Max seq len", f_lora: "LoRA rank", f_savesteps: "Save every N steps",
    f_resume: "Continue from (adapter of a previous run)", f_resume_none: "— fresh training —",
    f_ollama: "Import to Ollama", f_hf: "Upload to HuggingFace",
    btn_start: "Start training", btn_stop: "Stop",
    prog_title: "Progress", eta: "eta",
    exp_title: "Export", exp_gguf: "GGUF", exp_download: "Download model (.zip)",
    term_hint: "menu actions are typed here",
    runs_title: "Training runs", runs_sub: "every run keeps its adapter, merged model and GGUF — resume anytime",
    chat_title: "Quick chat test", chat_sub: "any ollama model — same /v1 API your apps use",
    chat_ph: "ask something…", chat_empty: "answer appears here…", chat_thinking: "thinking…",
    chat_running: "generation…", chat_done: "done in", ollama_title: "Ollama models",
    set_tunnel: "Cloudflare tunnel", mode: "Mode", mode_quick: "Quick (random URL)",
    mode_token: "Named tunnel (token)", mode_config: "Config file (domain + UUID)",
    f_token: "Tunnel token", f_domain: "Domain", f_uuid: "Tunnel UUID", f_creds: "Credentials JSON",
    hint_token: "Domain is configured in the Cloudflare Zero Trust dashboard → Public hostname → service http://localhost:PORT",
    btn_tunnel_start: "Start tunnel", btn_tunnel_stop: "Stop", set_panel: "Panel & services",
    f_port: "Panel port", f_pass: "Panel password", f_hftoken: "HuggingFace token", f_hfuser: "HuggingFace username",
    save: "Save", btn_update: "Update panel (git pull)",
    rail_server: "server", rail_training: "training",
    cpu: "CPU", ram: "Memory", disk: "Disk", load: "Load", uptime: "Uptime", host: "Host",
    connected: "connected", disconnected: "disconnected", session_end: "Session ended — press Enter to reconnect",
    toast_saved: "Settings saved", toast_saved_pass: "Saved — new password is active",
    toast_inject: "Sent to terminal", toast_fail: "Request failed", toast_copied: "Copied ✓",
    toast_starting: "Training started", toast_stopping: "Stopping…", toast_uploaded: "Uploaded",
    toast_no_dataset: "Choose a dataset first", toast_no_name: "Enter a run name",
    restart_note: "Port changed — restarting panel…",
    st_preparing: "preparing", st_training: "training", st_merging: "merging", st_gguf: "gguf",
    st_ollama: "ollama", st_hf: "hf upload", st_done: "done", st_error: "error", st_stopped: "stopped",
    st_stopping: "stopping", st_unknown: "—", gpu: "GPU", no_gpu: "not detected",
    delete_run: "Delete", confirm_delete: "Delete this run?", continue_run: "Continue",
    ckpt_saved: "ckpt saved", ckpt_up: "on HF", ckpt_save_btn: "Save checkpoint",
    ckpt_up_btn: "Upload checkpoint", ckpt_saving: "saving checkpoint…",
    ckpt_uploading: "uploading checkpoint…", ckpt_resume_hf: "resume from last checkpoint",
    download_btn: "Download", open_chat: "Chat", hf_set: "HF token ✓", hf_unset: "HF token not set",
    steps: "steps", adapter: "adapter", gguf_m: "gguf",
    ai_title: "AI dataset generator", ai_sub: "any OpenAI-compatible API writes the dataset for you — then auto-trains",
    ai_url: "API URL (chat/completions)", ai_key: "API key", ai_model: "Generator model",
    ai_prompt: "What dataset to make (topic, style, language, rules)",
    ai_count: "Samples", ai_temp: "Temperature",
    ai_refs: "Web references (urls, comma-separated — optional)",
    ai_maxtok: "Max tokens / batch",
    ai_autotrain: "Auto-start training when done", ai_keynote: "key stays in memory only",
    ai_generate: "Generate dataset", ai_test: "Test API", ai_use: "Use this dataset ↓", ai_fetch: "⤓ models",
    ai_need_prompt: "Describe the dataset first",
    ai_savehf: "Save dataset to HuggingFace", ds_push: "↑ HF",
  },
  ru: {
    subtitle: "обучи · запусти · выпусти свою модель", password: "Пароль", login: "Войти",
    wrong_pass: "Неверный пароль", login_hint: "Пароль выведен в результатах ячейки Kaggle-ноутбука.",
    nav_title: "рабочая область", nav_training: "Обучение", nav_terminal: "Терминал", nav_models: "Модели", nav_settings: "Настройки",
    ds_title: "Датасет", ds_sub: "jsonl / json / csv / parquet / txt — instruction+output, messages[] или сырой текст",
    ds_link: "Ссылка", ds_file: "Файл", ds_choose: "Выбрать файл…",
    ds_hf_check: "Проверить репо", ds_hf_hint: "один репо на строку · зелёный = готов · красный = исключён",
    ds_hf_need: "Добавь хотя бы один репо", ds_hf_none_ready: "Нет готовых датасетов — исправь красные или убери их",
    ds_copy: "копия на мой HF",
    cfg_title: "Модель и запуск", cfg_sub: "QLoRA 4-bit — работает на T4 · адаптер → merge → GGUF → Ollama → HuggingFace",
    f_base: "Базовая модель (HuggingFace id или локальный путь)", f_runname: "Имя рана / модели",
    f_epochs: "Эпохи", f_lr: "Learning rate", f_batch: "Batch size", f_accum: "Grad accum",
    f_seq: "Макс. длина seq", f_lora: "LoRA rank", f_savesteps: "Сохранять каждые N шагов",
    f_resume: "Продолжить с (адаптер прошлого рана)", f_resume_none: "— с нуля —",
    f_ollama: "Импорт в Ollama", f_hf: "Залить на HuggingFace",
    btn_start: "Начать обучение", btn_stop: "Остановить",
    prog_title: "Прогресс", eta: "осталось",
    exp_title: "Экспорт", exp_gguf: "GGUF", exp_download: "Скачать модель (.zip)",
    term_hint: "действия печатаются здесь",
    runs_title: "Прогоны обучения", runs_sub: "каждый ран хранит адаптер, merged-модель и GGUF — дообучай когда хочешь",
    chat_title: "Быстрый тест-чат", chat_sub: "любая ollama-модель — тот же /v1 API, что и в твоих приложениях",
    chat_ph: "спроси что-нибудь…", chat_empty: "ответ появится здесь…", chat_thinking: "думает…",
    chat_running: "генерация…", chat_done: "готово за", ollama_title: "Модели Ollama",
    set_tunnel: "Cloudflare туннель", mode: "Режим", mode_quick: "Быстрый (случайный URL)",
    mode_token: "Именованный туннель (токен)", mode_config: "Файл конфига (домен + UUID)",
    f_token: "Токен туннеля", f_domain: "Домен", f_uuid: "UUID туннеля", f_creds: "JSON учётных данных",
    hint_token: "Домен настраивается в панели Cloudflare Zero Trust → Public hostname → service http://localhost:ПОРТ",
    btn_tunnel_start: "Запустить туннель", btn_tunnel_stop: "Остановить", set_panel: "Панель и сервисы",
    f_port: "Порт панели", f_pass: "Пароль панели", f_hftoken: "HuggingFace токен", f_hfuser: "HuggingFace ник",
    save: "Сохранить", btn_update: "Обновить панель (git pull)",
    rail_server: "сервер", rail_training: "обучение",
    cpu: "Процессор", ram: "Память", disk: "Диск", load: "Нагрузка", uptime: "Время работы", host: "Хост",
    connected: "подключено", disconnected: "отключено", session_end: "Сессия завершена — Enter для переподключения",
    toast_saved: "Настройки сохранены", toast_saved_pass: "Сохранено — новый пароль активен",
    toast_inject: "Отправлено в терминал", toast_fail: "Ошибка запроса", toast_copied: "Скопировано ✓",
    toast_starting: "Обучение запущено", toast_stopping: "Останавливаю…", toast_uploaded: "Загружено",
    toast_no_dataset: "Сначала выбери датасет", toast_no_name: "Введи имя рана",
    restart_note: "Порт изменён — перезапускаю панель…",
    st_preparing: "подготовка", st_training: "обучение", st_merging: "склейка", st_gguf: "gguf",
    st_ollama: "ollama", st_hf: "загрузка на hf", st_done: "готово", st_error: "ошибка", st_stopped: "остановлено",
    st_stopping: "останавливаю", st_unknown: "—", gpu: "GPU", no_gpu: "не обнаружен",
    delete_run: "Удалить", confirm_delete: "Удалить этот ран?", continue_run: "Дообучить",
    ckpt_saved: "чекпоинт", ckpt_up: "на HF", ckpt_save_btn: "Сохранить чекпоинт",
    ckpt_up_btn: "Залить чекпоинт", ckpt_saving: "сохраняю чекпоинт…",
    ckpt_uploading: "заливаю чекпоинт…", ckpt_resume_hf: "продолжить с чекпоинта",
    download_btn: "Скачать", open_chat: "Чат", hf_set: "HF токен ✓", hf_unset: "HF токен не задан",
    steps: "шагов", adapter: "адаптер", gguf_m: "gguf",
    ai_title: "ИИ-генератор датасета", ai_sub: "любой OpenAI-совместимый API пишет датасет за тебя — потом авто-обучение",
    ai_url: "API URL (chat/completions)", ai_key: "API ключ", ai_model: "Модель-генератор",
    ai_prompt: "Какой датасет сделать (тема, стиль, язык, правила)",
    ai_count: "Сэмплов", ai_temp: "Temperature",
    ai_refs: "Референсы из интернета (url через запятую — опционально)",
    ai_maxtok: "Макс. токенов на батч",
    ai_autotrain: "Автостарт обучения после генерации", ai_keynote: "ключ хранится только в памяти",
    ai_generate: "Сгенерировать датасет", ai_test: "Проверить API", ai_use: "Использовать этот датасет ↓", ai_fetch: "⤓ модели",
    ai_need_prompt: "Сначала опиши датасет",
    ai_savehf: "Сохранить датасет на HuggingFace", ds_push: "↑ HF",
  },
  zh: {
    subtitle: "训练 · 部署 · 发布你自己的模型", password: "密码", login: "登录",
    wrong_pass: "密码错误", login_hint: "密码显示在 Kaggle notebook 的输出中。",
    nav_title: "工作区", nav_training: "训练", nav_terminal: "终端", nav_models: "模型", nav_settings: "设置",
    ds_title: "数据集", ds_sub: "jsonl / json / csv / parquet / txt — instruction+output、messages[] 或纯文本",
    ds_link: "链接", ds_file: "文件", ds_choose: "选择文件…",
    ds_hf_check: "检查仓库", ds_hf_hint: "每行一个仓库 · 绿色 = 可训练 · 红色 = 排除",
    ds_hf_need: "先添加至少一个仓库", ds_hf_none_ready: "没有可用的数据集 — 修正红色项或移除",
    ds_copy: "复制到我的 HF",
    cfg_title: "模型与运行", cfg_sub: "QLoRA 4-bit — T4 可跑 · 适配器 → 合并 → GGUF → Ollama → HuggingFace",
    f_base: "基础模型（HuggingFace id 或本地路径）", f_runname: "运行 / 模型名称",
    f_epochs: "轮数", f_lr: "学习率", f_batch: "批次大小", f_accum: "梯度累积",
    f_seq: "最大序列长度", f_lora: "LoRA 秩", f_savesteps: "每 N 步保存",
    f_resume: "从上次运行继续（适配器）", f_resume_none: "— 全新训练 —",
    f_ollama: "导入 Ollama", f_hf: "上传到 HuggingFace",
    btn_start: "开始训练", btn_stop: "停止",
    prog_title: "进度", eta: "剩余",
    exp_title: "导出", exp_gguf: "GGUF", exp_download: "下载模型 (.zip)",
    term_hint: "菜单操作会在此输入",
    runs_title: "训练运行", runs_sub: "每次运行保留适配器、合并模型和 GGUF — 随时继续训练",
    chat_title: "快速对话测试", chat_sub: "任意 ollama 模型 — 与你的应用相同的 /v1 API",
    chat_ph: "问点什么…", chat_empty: "回答会显示在这里…", chat_thinking: "思考中…",
    chat_running: "生成中…", chat_done: "完成用时", ollama_title: "Ollama 模型",
    set_tunnel: "Cloudflare 隧道", mode: "模式", mode_quick: "快速模式（随机网址）",
    mode_token: "命名隧道（Token）", mode_config: "配置文件（域名 + UUID）",
    f_token: "隧道 Token", f_domain: "域名", f_uuid: "隧道 UUID", f_creds: "凭据 JSON",
    hint_token: "域名需在 Cloudflare Zero Trust 控制台配置 → Public hostname → service http://localhost:端口",
    btn_tunnel_start: "启动隧道", btn_tunnel_stop: "停止", set_panel: "面板与服务",
    f_port: "面板端口", f_pass: "面板密码", f_hftoken: "HuggingFace 令牌", f_hfuser: "HuggingFace 用户名",
    save: "保存", btn_update: "更新面板 (git pull)",
    rail_server: "服务器", rail_training: "训练",
    cpu: "处理器", ram: "内存", disk: "磁盘", load: "负载", uptime: "运行时间", host: "主机",
    connected: "已连接", disconnected: "已断开", session_end: "会话已结束 — 按 Enter 重连",
    toast_saved: "设置已保存", toast_saved_pass: "已保存 — 新密码已生效",
    toast_inject: "已发送到终端", toast_fail: "请求失败", toast_copied: "已复制 ✓",
    toast_starting: "训练已开始", toast_stopping: "正在停止…", toast_uploaded: "已上传",
    toast_no_dataset: "请先选择数据集", toast_no_name: "请输入运行名称",
    restart_note: "端口已更改 — 正在重启面板…",
    st_preparing: "准备中", st_training: "训练中", st_merging: "合并中", st_gguf: "gguf",
    st_ollama: "ollama", st_hf: "上传 hf", st_done: "完成", st_error: "错误", st_stopped: "已停止",
    st_stopping: "停止中", st_unknown: "—", gpu: "GPU", no_gpu: "未检测到",
    delete_run: "删除", confirm_delete: "删除此次运行？", continue_run: "继续训练",
    ckpt_saved: "检查点", ckpt_up: "HF 上", ckpt_save_btn: "保存检查点",
    ckpt_up_btn: "上传检查点", ckpt_saving: "正在保存检查点…",
    ckpt_uploading: "正在上传检查点…", ckpt_resume_hf: "从检查点继续",
    download_btn: "下载", open_chat: "对话", hf_set: "HF 令牌 ✓", hf_unset: "未设置 HF 令牌",
    steps: "步数", adapter: "适配器", gguf_m: "gguf",
    ai_title: "AI 数据集生成器", ai_sub: "任何 OpenAI 兼容 API 替你写数据集 — 然后自动训练",
    ai_url: "API 地址（chat/completions）", ai_key: "API 密钥", ai_model: "生成模型",
    ai_prompt: "要做什么数据集（主题、风格、语言、规则）",
    ai_count: "样本数", ai_temp: "温度",
    ai_refs: "网络参考（网址，逗号分隔 — 可选）",
    ai_maxtok: "每批最大令牌数",
    ai_autotrain: "完成后自动开始训练", ai_keynote: "密钥只存于内存",
    ai_generate: "生成数据集", ai_test: "测试 API", ai_use: "使用此数据集 ↓", ai_fetch: "⤓ 模型",
    ai_need_prompt: "先描述数据集",
    ai_savehf: "保存数据集到 HuggingFace", ds_push: "↑ HF",
  },
};

const $ = (s) => document.querySelector(s);
window.__jsErrors = [];
window.addEventListener("error", (e) => {
  const msg = String(e.message || e).slice(0, 140);
  window.__jsErrors.push(msg);
  try {
    const el = document.createElement("div");
    el.className = "toast";
    el.style.borderColor = "var(--coral)";
    el.textContent = "⚠ JS: " + msg;
    document.getElementById("toasts").appendChild(el);
    setTimeout(() => el.remove(), 8000);
  } catch {}
});
const state = {
  token: localStorage.getItem("panel_token") || "",
  lang: localStorage.getItem("panel_lang") || "en",
  theme: localStorage.getItem("panel_theme") || "dark",
  view: "training",
  ws: null, wsUp: false, decoder: null, fit: null, term: null, welcomed: false,
  outSid: null, outSeq: 0,
  dsMode: "url", uploadedName: "", uploadedSize: 0,
  runName: "", lastPoints: [], trainPollTimer: null,
};

const BASE_PRESETS = ["unsloth/Llama-3.2-3B-Instruct", "unsloth/Qwen2.5-3B-Instruct",
  "unsloth/Llama-3.2-1B-Instruct", "unsloth/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"];

function t(key) { return (I18N[state.lang] && I18N[state.lang][key]) || I18N.en[key] || key; }
function applyLang() {
  document.documentElement.lang = state.lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.placeholder = t(el.dataset.i18nPh); });
  document.querySelectorAll(".lang-switch button").forEach((b) => b.classList.toggle("active", b.dataset.lang === state.lang));
  localStorage.setItem("panel_lang", state.lang);
}
function applyTheme() {
  document.documentElement.dataset.theme = state.theme;
  $("#theme-toggle").textContent = state.theme === "dark" ? "◐" : "◑";
  localStorage.setItem("panel_theme", state.theme);
  if (state.term) state.term.options.theme = { background: "transparent", foreground: "#d0d6e0", cursor: "#e4f222", selectionBackground: "rgba(228,242,34,.25)" };
}
function toast(msg) {
  const el = document.createElement("div");
  el.className = "toast"; el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 2800);
}

async function api(path, opts = {}) {
  opts.headers = Object.assign({ "X-Auth": state.token, "Content-Type": "application/json" }, opts.headers || {});
  const r = await fetch(path, opts);
  if (r.status === 401 && !path.startsWith("/api/login")) { showLogin(); throw new Error("unauthorized"); }
  return r;
}
const apiGet = (p) => api(p).then((r) => r.json());
const apiPost = (p, body) => api(p, { method: "POST", body: JSON.stringify(body) }).then((r) => r.json());

/* ---------------- auth ---------------- */
function showLogin() { $("#app").classList.add("hidden"); $("#login").classList.remove("hidden"); $("#login-pass").focus(); }
function showApp() { $("#login").classList.add("hidden"); $("#app").classList.remove("hidden"); setView("training"); }
async function tryVerify() {
  if (!state.token) { showLogin(); return; }
  try { const r = await fetch("/api/verify", { headers: { "X-Auth": state.token } }); r.ok ? showApp() : showLogin(); }
  catch { showLogin(); }
}
$("#login-btn").addEventListener("click", doLogin);
$("#login-pass").addEventListener("keydown", (e) => { if (e.key === "Enter") doLogin(); });
async function doLogin() {
  try {
    const r = await fetch("/api/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ password: $("#login-pass").value }) });
    if (!r.ok) throw new Error();
    const d = await r.json();
    state.token = d.token; localStorage.setItem("panel_token", d.token);
    $("#login-err").classList.add("hidden"); $("#login-pass").value = "";
    showApp();
  } catch {
    $("#login-err").classList.remove("hidden");
    const c = $("#login-card"); c.classList.remove("shake"); void c.offsetWidth; c.classList.add("shake");
  }
}
$("#logout").addEventListener("click", () => {
  localStorage.removeItem("panel_token"); state.token = "";
  if (state.ws) { try { state.ws.close(); } catch {} }
  showLogin();
});

/* ---------------- views ---------------- */
function setView(v) {
  state.view = v;
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  document.querySelectorAll(".view").forEach((el) => el.classList.add("hidden"));
  $("#view-" + v).classList.remove("hidden");
  if (v === "terminal") initTerminal();
  if (v === "models") refreshModels();
  if (v === "settings") loadSettings();
}
document.querySelectorAll(".nav-item").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));

/* ---------------- terminal (PTY ws + http fallback) ---------------- */
function decodeB64(b64) {
  return state.decoder.decode(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)), { stream: true });
}
function applyOut(d) {
  if (!state.term || typeof d.seq !== "number") return;
  if (d.sid !== state.outSid) { state.term.reset(); state.decoder = new TextDecoder("utf-8"); state.outSid = d.sid; state.outSeq = 0; if (!d.data) return; }
  if (d.seq <= state.outSeq) return;
  if (d.data) state.term.write(decodeB64(d.data));
  state.outSeq = d.seq;
}
async function pollOutput() {
  if (!state.term) return;
  try {
    const r = await fetch(`/api/output?offset=${state.outSeq}&sid=${encodeURIComponent(state.outSid || "")}`, { headers: { "X-Auth": state.token } });
    if (r.status === 401) { showLogin(); return; }
    applyOut(await r.json());
  } catch {}
}
function sendResize() {
  if (!state.term) return;
  if (state.ws && state.wsUp) state.ws.send(JSON.stringify({ type: "resize", cols: state.term.cols, rows: state.term.rows }));
  else apiPost("/api/input", { cols: state.term.cols, rows: state.term.rows }).catch(() => {});
}
function initTerminal() {
  if (state.term) { setTimeout(() => state.fit && state.fit.fit(), 60); return; }
  state.decoder = new TextDecoder("utf-8");
  state.term = new Terminal({
    fontFamily: '"JetBrains Mono", ui-monospace, Menlo, Consolas, monospace',
    fontSize: 13, cursorBlink: true, scrollback: 5000,
    theme: { background: "transparent", foreground: "#d0d6e0", cursor: "#e4f222" },
  });
  window.__term = state.term;
  state.fit = new FitAddon.FitAddon();
  state.term.loadAddon(state.fit);
  state.term.open($("#terminal"));
  applyTheme();
  state.term.onData((data) => {
    if (state.ws && state.wsUp) state.ws.send(JSON.stringify({ type: "input", data }));
    else apiPost("/api/input", { data }).catch(() => {});
  });
  new ResizeObserver(() => { try { state.fit.fit(); sendResize(); } catch {} }).observe($("#terminal"));
  connectWS();
}
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/term?auth=${encodeURIComponent(state.token)}`);
  state.ws = ws;
  ws.onopen = () => {
    state.wsUp = true; setConn(true);
    if (!state.welcomed) { state.welcomed = true; state.term.write(`\x1b[38;5;226m⚡ training-ai\x1b[0m \x1b[90m— ${t("term_hint")}\x1b[0m\r\n\r\n`); }
    sendResize();
  };
  ws.onmessage = (ev) => {
    let d; try { d = JSON.parse(ev.data); } catch { return; }
    if (d.type === "out") applyOut(d);
    else if (d.type === "exit") state.term.write(`\r\n\x1b[33m${t("session_end")}\x1b[0m\r\n`);
  };
  ws.onclose = () => { state.wsUp = false; setConn(false); setTimeout(connectWS, 1800); };
  ws.onerror = () => ws.close();
}
function setConn(up) { $("#conn-dot").classList.toggle("on", up); $("#conn-label").textContent = up ? t("connected") : t("disconnected"); }
async function inject(cmd) {
  try { await apiPost("/api/inject", { cmd }); toast(t("toast_inject")); }
  catch { toast("⚠ " + t("toast_fail")); }
}

/* ---------------- stats rail ---------------- */
function fmtBytes(n) {
  if (n > 1 << 30) return (n / (1 << 30)).toFixed(1) + " GB";
  if (n > 1 << 20) return (n / (1 << 20)).toFixed(0) + " MB";
  return (n / 1024).toFixed(0) + " KB";
}
function setBar(id, pct) { const b = $(id); b.style.width = Math.min(pct, 100) + "%"; b.classList.toggle("hot", pct > 85); }
function renderStats(d) {
  setConn(true);
  $("#s-cpu-v").textContent = d.cpu.toFixed(0) + "%"; setBar("#s-cpu", d.cpu);
  $("#s-ram-v").textContent = fmtBytes(d.ram_used) + " / " + fmtBytes(d.ram_total); setBar("#s-ram", d.ram_pct);
  $("#s-disk-v").textContent = fmtBytes(d.disk_used) + " / " + fmtBytes(d.disk_total); setBar("#s-disk", d.disk_pct);
  $("#gpu-list").innerHTML = (d.gpus && d.gpus.length)
    ? d.gpus.map((g, i) => `<div class="stat"><div class="stat-head"><span>GPU${i} · ${g.name.replace("Tesla ", "")}</span><b>${g.temp.toFixed(0)}°C</b></div><div class="bar-mono"><i class="${g.util > 5 ? "lime" : ""}" style="width:${g.util}%"></i></div><div class="stat-head" style="margin-top:3px"><span>VRAM</span><b>${fmtBytes(g.mem_used * 1048576)} / ${fmtBytes(g.mem_total * 1048576)}</b></div><div class="bar-mono"><i style="width:${g.mem_used / g.mem_total * 100}%"></i></div></div>`).join("")
    : `<div class="stat"><div class="stat-head"><span data-i18n="gpu">GPU</span><b>${t("no_gpu")}</b></div></div>`;
  $("#s-load").textContent = d.load.join(" / ");
  $("#s-uptime").textContent = fmtUptime(d.uptime);
  $("#s-host").textContent = d.host;
}
function fmtUptime(s) { const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60); return `${h}h ${m}m`; }

/* ---------------- tunnel / settings ---------------- */
const BASE_DIR = "$HOME/.kaggle-panel";
async function loadSettings() {
  try {
    const d = await apiGet("/api/env");
    $("#env-mode").value = d.mode || "quick";
    $("#env-token").value = ""; $("#env-token").placeholder = d.cf_token_set ? "•••• (saved)" : "eyJhIjoi...";
    $("#env-domain").value = d.cf_domain || "";
    $("#env-uuid").value = d.tunnel_uuid || "";
    $("#env-creds").value = "";
    $("#env-port").value = d.panel_port || "7860";
    state.portBefore = d.panel_port || "7860";
    $("#env-pass").value = ""; $("#env-hf").value = "";
    $("#env-hfuser").value = d.hf_user || "";
    $("#env-hf").placeholder = d.hf_token_set ? "•••• (saved)" : "hf_...";
    $("#hf-state").textContent = d.hf_token_set ? t("hf_set") : t("hf_unset");
    $("#hf-state").className = "badge " + (d.hf_token_set ? "b-green" : "");
    applyModeVis();
  } catch {}
}
function applyModeVis() {
  const m = $("#env-mode").value;
  document.querySelectorAll(".mode-token").forEach((el) => el.classList.toggle("hidden", m !== "token"));
  document.querySelectorAll(".mode-config").forEach((el) => el.classList.toggle("hidden", m !== "config"));
}
$("#env-mode").addEventListener("change", applyModeVis);
$("#env-save").addEventListener("click", async () => {
  const mode = $("#env-mode").value;
  const body = {
    cf_token: mode === "token" ? ($("#env-token").value.trim() || "__KEEP__") : "",
    cf_domain: $("#env-domain").value.trim(),
    tunnel_uuid: mode === "config" ? $("#env-uuid").value.trim() : "",
    credentials: mode === "config" && $("#env-creds").value.trim() ? $("#env-creds").value.trim() : "__KEEP__",
    panel_port: $("#env-port").value.trim() || "7860",
    panel_password: $("#env-pass").value.trim(),
    hf_token: $("#env-hf").value.trim() || "__KEEP__",
    hf_user: $("#env-hfuser").value.trim() || "__KEEP__",
  };
  if (body.cf_token === "__KEEP__") delete body.cf_token;
  if (body.credentials === "__KEEP__") delete body.credentials;
  if (body.hf_token === "__KEEP__") delete body.hf_token;
  if (body.hf_user === "__KEEP__") delete body.hf_user;
  if (body.hf_user === "__KEEP__") delete body.hf_user;
  try {
    const d = await apiPost("/api/env", body);
    if (d.token) { state.token = d.token; localStorage.setItem("panel_token", d.token); }
    toast(body.panel_password ? t("toast_saved_pass") : t("toast_saved"));
    if (state.portBefore && body.panel_port !== state.portBefore) {
      toast("↻ " + t("restart_note"));
      setTimeout(() => apiPost("/api/restart", {}).catch(() => {}), 1200);
    }
    loadSettings();
  } catch { toast("⚠ " + t("toast_fail")); }
});
$("#btn-tunnel-start").addEventListener("click", () => inject(`bash ${BASE_DIR}/scripts/tunnel.sh start`));
$("#btn-tunnel-stop").addEventListener("click", () => inject(`bash ${BASE_DIR}/scripts/tunnel.sh stop`));
$("#tunnel-copy").addEventListener("click", () => { const u = $("#tunnel-url").textContent; if (u && navigator.clipboard) navigator.clipboard.writeText(u).then(() => toast(t("toast_copied"))); });
$("#btn-update").addEventListener("click", () => inject(`bash ${BASE_DIR}/scripts/update.sh`));

/* ---------------- training: dataset ---------------- */
const modelInput = $("#cfg-base");
BASE_PRESETS.forEach((m) => {
  const b = document.createElement("button");
  b.className = "chip"; b.textContent = m.split("/").pop().replace("-Instruct", "");
  b.title = m;
  b.addEventListener("click", () => { modelInput.value = m; document.querySelectorAll("#base-presets .chip").forEach((c) => c.classList.remove("active")); b.classList.add("active"); });
  $("#base-presets").appendChild(b);
});
$("#ds-mode-url").addEventListener("click", () => setDsMode("url"));
$("#ds-mode-file").addEventListener("click", () => setDsMode("file"));
$("#ds-mode-hf").addEventListener("click", () => setDsMode("hf"));
function setDsMode(m) {
  state.dsMode = m;
  $("#ds-mode-url").classList.toggle("active", m === "url");
  $("#ds-mode-file").classList.toggle("active", m === "file");
  $("#ds-mode-hf").classList.toggle("active", m === "hf");
  $("#ds-url-wrap").classList.toggle("hidden", m !== "url");
  $("#ds-file-wrap").classList.toggle("hidden", m !== "file");
  $("#ds-hf-wrap").classList.toggle("hidden", m !== "hf");
}
$("#ds-hf-check").addEventListener("click", async () => {
  const repos = $("#ds-hf-repos").value.trim();
  if (!repos) { toast(t("ds_hf_need")); return; }
  $("#ds-hf-check").disabled = true;
  try {
    const d = await apiPost("/api/ds/hf_check", { repos });
    state.hfResults = d.results || [];
    renderHfList();
  } catch { toast("⚠ " + t("toast_fail")); }
  $("#ds-hf-check").disabled = false;
});
function renderHfList() {
  const box = $("#ds-hf-list");
  box.innerHTML = (state.hfResults || []).map((r, i) => {
    const ok = r.ok && (r.files || []).length;
    return `<div><span style="display:flex;align-items:center;gap:8px">
      <input type="checkbox" class="ds-hf-keep" data-i="${i}" ${ok ? "checked" : "disabled"} style="width:auto">
      <span class="badge ${ok ? "b-green" : "b-coral"}">${ok ? "✓ " + (r.files || []).length + " files" : "✗"}</span>
      <b style="font-family:var(--mono);font-size:12px">${r.repo}</b></span>
      <span class="small muted">${r.error || (r.files || []).slice(0, 3).join(", ")}</span></div>`;
  }).join("");
  box.querySelectorAll(".ds-hf-keep").forEach((cb) => cb.addEventListener("change", () => {
    state.hfResults[+cb.dataset.i]._keep = cb.checked;
  }));
}
$("#ds-file").addEventListener("change", async () => {
  const f = $("#ds-file").files[0];
  if (!f) return;
  $("#ds-file-name").textContent = f.name + " · " + fmtBytes(f.size);
  const fd = new FormData();
  fd.append("file", f);
  try {
    const r = await fetch("/api/train/upload", { method: "POST", headers: { "X-Auth": state.token }, body: fd });
    const d = await r.json();
    if (!r.ok || !d.ok) throw new Error(d.error || "fail");
    state.uploadedName = d.name; state.uploadedSize = d.size;
    $("#ds-file-name").textContent = `${d.name} · ${fmtBytes(d.size)} ✓`;
    $("#ds-push").classList.remove("hidden");
    $("#ds-hf-link").classList.add("hidden");
    toast(t("toast_uploaded"));
  } catch { toast("⚠ " + t("toast_fail")); }
});

/* ---------------- training: start / stop / status ---------------- */
$("#btn-train-start").addEventListener("click", async () => {
  const name = $("#cfg-name").value.trim();
  if (!name) { toast(t("toast_no_name")); return; }
  let dsValue = state.dsMode === "url" ? $("#ds-url").value.trim() : state.uploadedName;
  let dsType = state.dsMode;
  if (state.dsMode === "hf") {
    const kept = (state.hfResults || []).filter((r) => r.ok && r._keep !== false && (r.files || []).length).map((r) => r.repo);
    if (!kept.length) { toast(t("ds_hf_none_ready")); return; }
    dsType = "hf"; dsValue = kept.join(",");
  }
  if (!dsValue) { toast(t("toast_no_dataset")); return; }
  const body = {
    run_name: name,
    base_model: modelInput.value.trim(),
    dataset_type: state.dsMode,
    dataset_value: dsValue,
    epochs: +$("#cfg-epochs").value || 3,
    lr: parseFloat($("#cfg-lr").value) || 2e-4,
    batch_size: +$("#cfg-batch").value || 2,
    grad_accum: +$("#cfg-accum").value || 4,
    max_seq: +$("#cfg-seq").value || 1024,
    lora_r: +$("#cfg-lora").value || 16,
    save_steps: +$("#cfg-save").value || 100,
    resume_from: $("#cfg-resume").value || null,
    ollama_import: $("#cfg-ollama").checked,
    ollama_name: $("#cfg-ollama-name").value.trim() || name,
    hf_upload: $("#cfg-hf").checked,
    hf_repo: $("#cfg-hf-repo").value.trim() || null,
  };
  try {
    const d = await apiPost("/api/train/start", body);
    if (d.error) { toast("⚠ " + d.error); return; }
    state.runName = d.run;
    toast(t("toast_starting"));
    startTrainPoll();
  } catch { toast("⚠ " + t("toast_fail")); }
});
$("#btn-train-stop").addEventListener("click", async () => {
  try { await apiPost("/api/train/stop", { run: state.runName }); toast(t("toast_stopping")); } catch {}
});
$("#btn-ck-save").addEventListener("click", async () => {
  try { await apiPost("/api/train/command", { run: state.runName, cmd: "save" }); toast(t("ckpt_saving")); } catch {}
});
$("#btn-ck-upload").addEventListener("click", async () => {
  try { await apiPost("/api/train/command", { run: state.runName, cmd: "upload" }); toast(t("ckpt_uploading")); } catch {}
});

const BADGE_CLS = { preparing: "b-iris", training: "b-lime", merging: "b-teal", gguf: "b-teal", ollama: "b-teal", hf_upload: "b-teal", done: "b-green", error: "b-coral", stopped: "", stopping: "b-iris", interrupted: "b-coral" };
function badgeFor(status) { return `badge ${BADGE_CLS[status] || ""}`; }

function fmtEta(sec) { if (!sec || sec < 0) return "—"; const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60); return h ? `${h}h ${m}m` : `${m}m ${Math.floor(sec % 60)}s`; }

function renderTrain(d) {
  const st = d.state || {};
  const status = st.status || "unknown";
  const active = ["preparing", "training", "merging", "gguf", "ollama", "hf_upload", "stopping"].includes(status);
  $("#train-badge").textContent = t("st_" + (BADGE_CLS[status] ? status : "unknown"));
  $("#train-badge").className = badgeFor(status);
  $("#btn-train-start").classList.toggle("hidden", active);
  $("#btn-train-stop").classList.toggle("hidden", !active);
  $("#p-step").textContent = `${st.step || 0} / ${st.total_steps || 0}`;
  $("#p-loss").textContent = st.loss != null ? st.loss.toFixed(4) : "—";
  $("#p-lr").textContent = st.lr ? st.lr.toExponential(1) : "—";
  $("#p-epoch").textContent = st.epoch || "—";
  const pct = st.total_steps ? Math.min(100, (st.step || 0) / st.total_steps * 100) : 0;
  $("#p-bar").style.width = pct + "%";
  if (active && st.step && st.total_steps && status === "training") {
    const pts = state.lastPoints;
    if (pts.length > 1) {
      const dt = pts[pts.length - 1].ts - pts[0].ts, ds = pts[pts.length - 1].step - pts[0].step;
      if (ds > 0) $("#p-eta").textContent = fmtEta((st.total_steps - st.step) * (dt / ds));
    }
  } else $("#p-eta").textContent = "—";
  $("#p-ds").textContent = st.current_dataset || "—";
  $("#p-ds").title = st.current_dataset || "";
  $("#p-ck").textContent = st.last_ckpt_step ? "step " + st.last_ckpt_step : "—";
  $("#p-ckup").textContent = st.hf_uploading ? "…" : (st.last_ckpt_uploaded ? "step " + st.last_ckpt_uploaded : "—");
  const cke = $("#p-ck-err");
  if (st.hf_ckpt_error) { cke.classList.remove("hidden"); cke.textContent = "⚠ " + st.hf_ckpt_error; } else cke.classList.add("hidden");
  $("#btn-ck-save").classList.toggle("hidden", !active);
  $("#btn-ck-upload").classList.toggle("hidden", !active);
  drawLossChart($("#loss-chart"), state.lastPoints);
  const logBox = $("#train-log");
  const stick = logBox.scrollHeight - logBox.scrollTop - logBox.clientHeight < 40;
  logBox.innerHTML = (d.log || []).map((l) => `<span class="${/error|traceback|failed/i.test(l) ? "err" : ""}">${l.replace(/</g, "&lt;")}</span>`).join("\n");
  if (stick) logBox.scrollTop = logBox.scrollHeight;
  // export
  const gg = st.gguf_path;
  $("#e-gguf").textContent = gg ? gg.split("/").pop() : (st.merged_dir ? "—" : "—");
  $("#e-ollama").innerHTML = st.ollama_ready ? `<span style="color:var(--pulse)">✓ ${st.ollama_model}</span>` : (st.ollama_model || "—");
  $("#e-hf").innerHTML = st.hf_url ? `<a href="${st.hf_url}" target="_blank" rel="noopener" style="color:var(--lavender)">${st.hf_url.replace("https://huggingface.co/", "")} ↗</a>` : (st.status === "done" ? "—" : "—");
  const dl = $("#e-download");
  if (gg && ["gguf", "done", "error", "stopped"].includes(status)) { dl.classList.remove("hidden"); dl.href = `/api/train/download/${st.run}?t=${state.token}`; }
  else dl.classList.add("hidden");
  const hl = $("#e-hf-link");
  if (st.hf_url) { hl.classList.remove("hidden"); hl.href = st.hf_url; } else hl.classList.add("hidden");
  // rail
  $("#rt-name").textContent = st.run || "—";
  $("#rt-loss").textContent = st.loss != null ? st.loss.toFixed(3) : "—";
  $("#rt-bar").style.width = pct + "%";
  $("#rt-step").textContent = `${st.step || 0} / ${st.total_steps || 0}`;
  $("#rt-badge").textContent = t("st_" + (BADGE_CLS[status] ? status : "unknown"));
  $("#rt-badge").className = badgeFor(status);
  // continue-from select + stop button need run name
  if (st.run && !state.runName) state.runName = st.run;
  return active;
}

async function pollTrain() {
  if (!state.runName) {
    try {
      const d = await apiGet("/api/train/list");
      const ACTIVE = ["preparing", "training", "merging", "gguf", "ollama", "hf_upload", "stopping"];
      const running = (d.runs || []).find((r) => r.running || ACTIVE.includes(r.status));
      state.runName = running ? running.name : ((d.runs || [])[0] || {}).name || "";
    } catch {}
    if (!state.runName) { renderTrain({ state: { status: "unknown" }, points: [], log: [] }); return; }
  }
  try {
    const d = await apiGet(`/api/train/status?run=${encodeURIComponent(state.runName)}`);
    state.lastPoints = d.points || [];
    renderTrain(d);
  } catch {}
}

/* ---------------- loss chart ---------------- */
function drawLossChart(canvas, points) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = 220;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const padL = 46, padR = 12, padT = 14, padB = 26;
  const iw = w - padL - padR, ih = h - padT - padB;
  ctx.font = "10px 'JetBrains Mono', monospace";
  if (!points || points.length < 2) {
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--ash") || "#62666d";
    ctx.fillText("— " + t("prog_title") + " —", w / 2 - 40, h / 2);
    return;
  }
  const losses = points.map((p) => p.loss).filter((x) => x != null);
  let lo = Math.min(...losses), hi = Math.max(...losses);
  const pad = (hi - lo) * 0.08 || 0.1; lo = Math.max(0, lo - pad); hi += pad;
  const maxStep = points[points.length - 1].step || 1;
  const X = (s) => padL + (s / maxStep) * iw;
  const Y = (v) => padT + (1 - (v - lo) / (hi - lo)) * ih;
  ctx.strokeStyle = "rgba(128,132,140,.18)"; ctx.lineWidth = 1;
  ctx.fillStyle = "#62666d";
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * i / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    ctx.fillText(v.toFixed(2), 8, y + 3);
  }
  for (let i = 0; i <= 4; i++) {
    const s = maxStep * i / 4;
    ctx.fillText(Math.round(s).toString(), Math.min(X(s), w - 30), h - 8);
  }
  ctx.strokeStyle = "#e4f222"; ctx.lineWidth = 1.6; ctx.beginPath();
  let started = false;
  for (const p of points) {
    if (p.loss == null) continue;
    const x = X(p.step), y = Y(p.loss);
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke();
  const lastP = points[points.length - 1];
  if (lastP.loss != null) {
    ctx.fillStyle = "#e4f222";
    ctx.beginPath(); ctx.arc(X(lastP.step), Y(lastP.loss), 3, 0, Math.PI * 2); ctx.fill();
  }
}

/* ---------------- models view ---------------- */
async function refreshModels() {
  try {
    const [list, status] = await Promise.all([apiGet("/api/train/list"), apiGet("/api/status")]);
    renderRuns(list.runs || []);
    renderOllama(status.models || []);
    fillResumeSelect((list.runs || []).filter((r) => r.adapter_dir));
  } catch {}
}
function renderRuns(runs) {
  const box = $("#runs-list");
  if (!runs.length) { box.innerHTML = `<span class="muted small">—</span>`; return; }
  box.innerHTML = runs.map((r) => {
    const cls = BADGE_CLS[r.status] || "";
    return `<div class="run-card">
      <div class="run-head"><span class="run-name">${r.name}</span><span class="badge ${cls}">${t("st_" + (BADGE_CLS[r.status] ? r.status : "unknown"))}</span></div>
      <div class="run-meta">
        <span>⬢ ${r.base_model || "—"}</span>
        <span>${t("steps")}: ${r.step || 0}/${r.total_steps || 0}</span>
        ${r.loss != null ? `<span>loss: ${r.loss}</span>` : ""}
        ${r.ollama_ready ? `<span style="color:var(--pulse)">ollama ✓ ${r.ollama_model}</span>` : ""}
      </div>
      ${r.error ? `<div class="small" style="color:var(--coral)">${String(r.error).slice(0, 140)}</div>` : ""}
      <div class="row wrap">
        ${r.ollama_ready ? `<button class="btn btn-ghost" data-chat="${r.ollama_model}">${t("open_chat")}</button>` : ""}
        <a class="btn btn-ghost" href="/api/train/download/${r.name}?t=${state.token}">${t("download_btn")}</a>
        ${r.adapter_dir ? `<button class="btn btn-ghost" data-continue="${r.name}">${t("continue_run")}</button>` : ""}
        <button class="btn btn-danger" data-del="${r.name}">${t("delete_run")}</button>
      </div></div>`;
  }).join("");
  box.querySelectorAll("[data-continue]").forEach((b) => b.addEventListener("click", () => {
    $("#cfg-resume").value = "runs/" + b.dataset.continue + "/adapter";
    state.runName = b.dataset.continue;
    setView("training");
    toast("↻ " + b.dataset.continue);
  }));
  box.querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", async () => {
    if (!confirm(t("confirm_delete"))) return;
    try { await apiPost("/api/train/delete", { run: b.dataset.del }); refreshModels(); } catch {}
  }));
  box.querySelectorAll("[data-chat]").forEach((b) => b.addEventListener("click", () => {
    $("#chat-model").value = b.dataset.chat;
    $("#chat-input").focus();
  }));
}
function renderOllama(models) {
  $("#ollama-list").innerHTML = models.length
    ? models.map((m) => `<div><b>${m.name}</b><span>${m.size || ""}</span></div>`).join("")
    : `<div><b>—</b></div>`;
  const sel = $("#chat-model");
  const cur = sel.value;
  sel.innerHTML = models.map((m) => `<option>${m.name}</option>`).join("") || "<option>—</option>";
  if (cur && models.some((m) => m.name === cur)) sel.value = cur;
}
function fillResumeSelect(runs) {
  const sel = $("#cfg-resume");
  const cur = sel.value;
  const repo = $("#cfg-hf-repo").value.trim();
  let opts = `<option value="">${t("f_resume_none")}</option>`;
  for (const r of runs) {
    if (r.last_ckpt_step && r.last_ckpt_local) {
      opts += `<option value="${r.last_ckpt_local}">${r.name} — ${t("ckpt_resume_step")} ${r.last_ckpt_step}</option>`;
    } else if (r.adapter_dir) {
      opts += `<option value="runs/${r.name}/adapter">${r.name} (${t("adapter")})</option>`;
    }
  }
  if (repo) opts += `<option value="hf:${repo}">HF: ${repo} (${t("ckpt_resume_hf")})</option>`;
  sel.innerHTML = opts;
  if (cur) sel.value = cur;
}

/* ---------------- chat test (SSE stream) ---------------- */
$("#chat-send").addEventListener("click", sendChat);
$("#chat-input").addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });
async function sendChat() {
  const model = $("#chat-model").value;
  const q = $("#chat-input").value.trim();
  if (!q || !model || model === "—") return;
  const out = $("#chat-out");
  $("#chat-input").value = "";
  out.innerHTML = `<span class="thinking">${t("chat_thinking")}</span>`;
  const maxTok = +$("#chat-max").value || 256;
  const t0 = timeNow();
  try {
    const r = await fetch("/ollama/v1/chat/completions", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages: [{ role: "user", content: q }], stream: true, max_tokens: maxTok }),
    });
    if (!r.ok || !r.body) throw new Error("HTTP " + r.status);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "", content = "", thinking = false, n = 0;
    out.innerHTML = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        const s = line.trim();
        if (!s.startsWith("data:")) continue;
        const payload = s.slice(5).trim();
        if (payload === "[DONE]") continue;
        try {
          const d = JSON.parse(payload);
          const delta = (d.choices && d.choices[0] && d.choices[0].delta) || {};
          if (delta.reasoning && !content && !thinking) { thinking = true; out.innerHTML = `<span class="thinking">${t("chat_thinking")}</span>`; }
          if (delta.content) {
            if (thinking) { thinking = false; out.innerHTML = ""; }
            content += delta.content; n++;
            out.textContent = content;
            out.scrollTop = out.scrollHeight;
          }
        } catch {}
      }
    }
    if (!content) out.innerHTML = `<span class="muted small">—</span>`;
    out.insertAdjacentHTML("beforeend", `<div class="small muted mono" style="margin-top:8px">${t("chat_done")} ${(timeNow() - t0).toFixed(1)}s</div>`);
  } catch (e) {
    out.innerHTML = `<span style="color:var(--coral)">⚠ ${e.message}</span>`;
  }
}
function timeNow() { return performance.now() / 1000; }

/* ---------------- AI dataset generator ---------------- */
$("#ai-fetch").addEventListener("click", async () => {
  const btn = $("#ai-fetch"); btn.disabled = true;
  try {
    const d = await apiPost("/api/ai/models", { api_url: $("#ai-url").value.trim(), api_key: $("#ai-key").value.trim() });
    if (!d.ok) { toast("⚠ " + (d.error || "fail")); return; }
    const dl = $("#ai-model-list");
    dl.innerHTML = (d.models || []).map((m) => `<option value="${m}">`).join("");
    toast((d.models || []).length + " models fetched");
  } catch { toast("⚠ " + t("toast_fail")); }
  btn.disabled = false;
});
$("#ai-test").addEventListener("click", async () => {
  try {
    const d = await apiPost("/api/ai/test", { api_url: $("#ai-url").value.trim(), api_key: $("#ai-key").value.trim(), model: $("#ai-model").value.trim() });
    if (d.ok) {
      toast("✓ " + (d.reply || "ok"));
      if (d.resolved) console.info("resolved:", d.resolved);
      if (d.models && d.models.length) {
        $("#ai-model-list").innerHTML = d.models.map((m) => `<option value="${m}">`).join("");
      }
    } else toast("⚠ " + (d.error || "fail"));
  } catch { toast("⚠ " + t("toast_fail")); }
});
$("#ai-start").addEventListener("click", async () => {
  const prompt = $("#ai-prompt").value.trim();
  if (!prompt) { toast(t("ai_need_prompt")); return; }
  try {
    const d = await apiPost("/api/ai/start", {
      prompt,
      api_url: $("#ai-url").value.trim(),
      api_key: $("#ai-key").value.trim(),
      model: $("#ai-model").value.trim(),
      count: +$("#ai-count").value || 50,
      temperature: +$("#ai-temp").value || 0.9,
      max_tokens: +$("#ai-maxtok").value || 8000,
      ref_urls: $("#ai-refs").value.trim(),
      auto_train: $("#ai-autotrain").checked,
      save_ds_hf: $("#ai-savehf").checked,
      hf_token: $("#env-hf").value.trim(),
      dataset_name: ($("#cfg-name").value.trim() || "ai") + "-ds",
      train: {
        base_model: modelInput.value.trim(),
        epochs: +$("#cfg-epochs").value || 3,
        lr: parseFloat($("#cfg-lr").value) || 2e-4,
        batch_size: +$("#cfg-batch").value || 2,
        grad_accum: +$("#cfg-accum").value || 4,
        max_seq: +$("#cfg-seq").value || 1024,
        lora_r: +$("#cfg-lora").value || 16,
        save_steps: +$("#cfg-save").value || 100,
        save_minutes: +$("#cfg-savemin").value || 0,
        keep_ckpts: +$("#cfg-keepck").value || 2,
        ckpt_upload: true,
        ollama_import: $("#cfg-ollama").checked,
        ollama_name: $("#cfg-ollama-name").value.trim() || $("#cfg-name").value.trim(),
        hf_upload: $("#cfg-hf").checked,
        hf_repo: $("#cfg-hf-repo").value.trim() || null,
      },
    });
    if (d.error) { toast("⚠ " + d.error); return; }
    toast(t("toast_starting"));
  } catch { toast("⚠ " + t("toast_fail")); }
});
$("#ai-stop").addEventListener("click", async () => {
  try { await apiPost("/api/ai/stop", {}); toast(t("toast_stopping")); } catch {}
});
$("#ai-use-ds").addEventListener("click", () => {
  const st = window.__aiState || {};
  if (!st.dataset_file) return;
  state.dsMode = "file";
  state.uploadedName = st.dataset_file.split("/").pop();
  setDsMode("file");
  $("#ds-file-name").textContent = state.uploadedName + " · AI ✓";
  toast("↓ " + state.uploadedName);
});
$("#ds-push").addEventListener("click", async () => {
  if (!state.uploadedName) return;
  $("#ds-push").disabled = true;
  try {
    const d = await apiPost("/api/ds/push", { name: state.uploadedName });
    if (d.ok) {
      toast("↑ HF ✓");
      const a = $("#ds-hf-link");
      a.classList.remove("hidden"); a.href = d.url; a.textContent = "HF ↗";
    } else toast("⚠ " + d.error);
  } catch { toast("⚠ " + t("toast_fail")); }
  $("#ds-push").disabled = false;
});

function renderAi(d) {
  const st = d || {};
  const active = st.status === "running";
  $("#ai-badge").textContent = st.status === "idle" ? "—" : t("st_" + (BADGE_CLS[st.status] ? st.status : "unknown")) || st.status;
  $("#ai-badge").className = "badge " + (BADGE_CLS[st.status] || "");
  $("#ai-start").classList.toggle("hidden", active);
  $("#ai-stop").classList.toggle("hidden", !active);
  const pct = st.total ? Math.min(100, (st.generated || 0) / st.total * 100) : 0;
  $("#ai-bar").style.width = pct + "%";
  $("#ai-progress").textContent = st.status === "running" || st.generated
    ? `${st.generated || 0} / ${st.total || 0} — ${st.message || ""}` : (st.message || "—");
  window.__aiState = st;
  const done = ["done", "stopped", "interrupted", "error"].includes(st.status) && st.dataset_file;
  $("#ai-use-ds").classList.toggle("hidden", !done);
  const prev = $("#ai-preview");
  if (st.last_samples && st.last_samples.length) {
    prev.classList.remove("hidden");
    prev.innerHTML = st.last_samples.map((s) =>
      `<div style="margin-bottom:6px"><b style="color:var(--mist)">${String(s.instruction).slice(0, 80)}</b><br>${String(s.output).slice(0, 120)}…</div>`).join("");
  } else prev.classList.add("hidden");
  if (st.status === "done") {
    state.dsMode = "file";
    state.uploadedName = (st.dataset_file || "").split("/").pop();
    if (st.auto_train && st.auto_train.ok) state.runName = st.auto_train.run || state.runName;
    if (st.ds_hf_url) toast("↑ HF ✓ " + st.ds_hf_url.replace("https://huggingface.co/datasets/", ""));
  }
}

/* ---------------- boot ---------------- */
applyLang();
applyTheme();
tryVerify();
setInterval(() => apiGet("/api/stats").then(renderStats).catch(() => setConn(false)), 2000);
setInterval(pollTrain, 2500);
setInterval(() => { if (state.view === "models") refreshModels(); }, 6000);
setInterval(pollOutput, 1000);
setInterval(() => apiGet("/api/ai/status").then(renderAi).catch(() => {}), 2000);
apiGet("/api/stats").then(renderStats).catch(() => {});
pollTrain();
apiGet("/api/ai/status").then(renderAi).catch(() => {});
