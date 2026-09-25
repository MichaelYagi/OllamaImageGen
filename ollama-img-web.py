#!/usr/bin/env python3
"""
ollama-img-web: web UI for Ollama image models (x/z-image-turbo, x/flux2-klein).

Generation runs in a background worker on the server, one job at a time.
You can close the tab, refresh, or switch devices; finished images are saved
to disk and show up in the gallery whenever you come back.

The Ollama URL can be set three ways (later wins):
  1. OLLAMA_URL env var
  2. --ollama / -o command-line flag
  3. The "Ollama server" field in the web page (saved in that browser)

Other options:
  --port / -p   or PORT env        Port to listen on (default 8080)
  --images      or IMAGES_DIR env  Where images are saved (default ./images next to this script)
  --reset-auth                     Choose a new username/password
  APP_USER / APP_PASSWORD env      Override the saved login (e.g. for systemd/Docker)

Login: on first launch you're asked in the terminal for a username and
password. They're saved (password hashed) to ~/.config/ollama-img-web/auth.json.
  GEN_TIMEOUT env                  Seconds to wait on Ollama per image (default 900)

Run:
  python3 ollama-img-web.py
  tmole 8080
"""
import argparse
import base64
import getpass
import hashlib
import io
import json
import os
import queue
import re
import secrets
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
PORT = int(os.environ.get("PORT", "8080"))
GEN_TIMEOUT = int(os.environ.get("GEN_TIMEOUT", "900"))
IMAGES_DIR = os.environ.get("IMAGES_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "images"))
AUTH_FILE = os.path.join(os.path.expanduser("~"), ".config", "ollama-img-web", "auth.json")
PBKDF2_ROUNDS = 200_000
auth = {}                   # {"user", "salt", "hash"} or {"user", "password"} from env
auth_ok_cache = set()       # Authorization headers already verified (avoids re-hashing every request)


def hash_password(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, PBKDF2_ROUNDS).hex()


def setup_auth(reset=False):
    """Load login from env or AUTH_FILE; on first launch (or --reset-auth) ask in the terminal."""
    global auth
    env_user, env_pw = os.environ.get("APP_USER"), os.environ.get("APP_PASSWORD")
    if env_user and env_pw:
        auth = {"user": env_user, "password": env_pw}
        print(f"Login: using APP_USER/APP_PASSWORD from the environment (user '{env_user}')")
        return
    if not reset and os.path.exists(AUTH_FILE):
        with open(AUTH_FILE) as f:
            auth = json.load(f)
        print(f"Login: user '{auth['user']}' (change with --reset-auth)")
        return
    if not sys.stdin.isatty():
        sys.exit("No login is set up yet. Run once in a terminal to choose one, "
                 "or set APP_USER and APP_PASSWORD.")

    print("\nFirst launch: choose a login for the web page.")
    while True:
        user = input("  Username: ").strip()
        if user and ":" not in user:
            break
        print("  Username can't be empty or contain ':'.")
    while True:
        pw = getpass.getpass("  Password: ")
        if not pw:
            print("  Password can't be empty.")
            continue
        if getpass.getpass("  Confirm password: ") == pw:
            break
        print("  Passwords didn't match, try again.")
    salt = secrets.token_bytes(16)
    auth = {"user": user, "salt": salt.hex(), "hash": hash_password(pw, salt)}
    os.makedirs(os.path.dirname(AUTH_FILE), exist_ok=True)
    fd = os.open(AUTH_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(auth, f)
    print(f"  Saved to {AUTH_FILE}\n")


def check_login(user, pw):
    if not secrets.compare_digest(user, auth.get("user", "")):
        return False
    if "password" in auth:
        return secrets.compare_digest(pw, auth["password"])
    return secrets.compare_digest(hash_password(pw, bytes.fromhex(auth["salt"])), auth["hash"])
MODELS = ("x/z-image-turbo", "x/flux2-klein")
ID_RE = re.compile(r"^[0-9a-f]{32}$")
IMG_CACHE_MAX = 64          # images kept in RAM
img_cache = {}              # id -> bytes (insertion-ordered, oldest evicted)
img_cache_lock = threading.Lock()


def read_image(jid):
    with img_cache_lock:
        data = img_cache.get(jid)
    if data is None:
        with open(os.path.join(IMAGES_DIR, jid + ".png"), "rb") as f:
            data = f.read()
        with img_cache_lock:
            img_cache[jid] = data
            while len(img_cache) > IMG_CACHE_MAX:
                img_cache.pop(next(iter(img_cache)))
    return data

jobs = {}                 # id -> job dict
jobs_lock = threading.Lock()
work = queue.Queue()


# ---------------------------------------------------------------- jobs

def public(job):
    j = {k: job[k] for k in ("id", "prompt", "model", "ollama", "status", "created", "error")}
    end = job.get("finished") or time.time()
    j["secs"] = round(end - job["started"], 1) if job.get("started") else None
    if job["status"] == "queued":
        with jobs_lock:
            j["ahead"] = sum(1 for o in jobs.values()
                             if o["status"] in ("queued", "running") and o["created"] < job["created"])
    return j


def save_meta(job):
    with open(os.path.join(IMAGES_DIR, job["id"] + ".json"), "w") as f:
        json.dump(job, f)


def load_saved():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    for name in os.listdir(IMAGES_DIR):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(IMAGES_DIR, name)) as f:
                job = json.load(f)
            if job.get("status") == "done" and os.path.exists(os.path.join(IMAGES_DIR, job["id"] + ".png")):
                jobs[job["id"]] = job
        except Exception:
            pass


def run_job(job):
    payload = json.dumps({"model": job["model"], "prompt": job["prompt"], "stream": False}).encode()
    req = urllib.request.Request(f"{job['ollama']}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=GEN_TIMEOUT) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama returned HTTP {e.code}: {e.read().decode(errors='replace')[:1000]}")
    except Exception as e:
        raise RuntimeError(f"Could not reach Ollama at {job['ollama']}: {e}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError("Ollama returned malformed JSON: " + raw[:500].decode(errors="replace"))
    if not data.get("image"):
        raise RuntimeError(data.get("error") or data.get("response") or "Model returned no image and no response text.")
    png = base64.b64decode(data["image"])
    if not png:
        raise RuntimeError("Decoded image is empty.")
    with open(os.path.join(IMAGES_DIR, job["id"] + ".png"), "wb") as f:
        f.write(png)


def worker():
    while True:
        job = work.get()
        if job["status"] != "queued":        # deleted while waiting
            continue
        job["status"], job["started"] = "running", time.time()
        print(f"[job {job['id'][:8]}] {job['model']}: {job['prompt'][:60]!r}", file=sys.stderr)
        try:
            run_job(job)
            job["status"] = "done"
            save_meta_job = dict(job)
        except Exception as e:
            job["status"], job["error"] = "error", str(e)
            save_meta_job = None
            print(f"[job {job['id'][:8]}] failed: {e}", file=sys.stderr)
        job["finished"] = time.time()
        if save_meta_job:
            save_meta_job["finished"] = job["finished"]
            save_meta(save_meta_job)


def delete_job(jid):
    """Delete one job and its files. Returns None on success, or a reason string."""
    with jobs_lock:
        job = jobs.get(jid)
        if not job:
            return "not found"
        if job["status"] == "running":
            return "running"
        job["status"] = "deleted"          # worker skips it if still queued
        del jobs[jid]
    with img_cache_lock:
        img_cache.pop(jid, None)
    for ext in (".png", ".json"):
        try:
            os.remove(os.path.join(IMAGES_DIR, jid + ext))
        except FileNotFoundError:
            pass
    return None


def resolve_ollama(url):
    """Return a clean base URL, falling back to the server default. Raises ValueError if invalid."""
    url = (url or "").strip().rstrip("/")
    if not url:
        return OLLAMA_URL
    if "://" not in url:
        url = "http://" + url
    p = urllib.parse.urlparse(url)
    if p.scheme not in ("http", "https") or not p.netloc:
        raise ValueError(f"'{url}' is not a valid http(s) URL.")
    return f"{p.scheme}://{p.netloc}{p.path.rstrip('/')}"


# ---------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Light table</title>
<style>
  :root {
    --bg: #d9dee4; --panel: #eef1f4; --ink: #1b2430; --muted: #5d6b7a;
    --line: #b7c0ca; --accent: #2f55d4; --bad: #b3261e;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #151a21; --panel: #1e252e; --ink: #e3e8ee; --muted: #93a0ae;
            --line: #33404d; --accent: #7d9bff; --bad: #ff8a80; }
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  html, body { overflow-x: hidden; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 24px 20px 48px;
         display: grid; grid-template-columns: 340px minmax(0, 1fr); gap: 24px; }
  main > * { min-width: 0; }
  @media (max-width: 820px) { main { grid-template-columns: minmax(0, 1fr); padding: 16px 12px 40px; } }
  h1 { font-size: 1.25rem; margin: 0 0 16px; font-weight: 650; }
  form { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
         padding: 16px; display: flex; flex-direction: column; gap: 14px; align-self: start; }
  label { font-size: .875rem; color: var(--muted); }
  textarea { width: 100%; min-height: 150px; resize: vertical; padding: 10px;
             font: inherit; color: var(--ink); background: var(--bg);
             border: 1px solid var(--line); border-radius: 6px; }
  fieldset { border: 0; padding: 0; margin: 0; display: flex; gap: 6px; }
  fieldset label { flex: 1; }
  fieldset input { position: absolute; opacity: 0; }
  fieldset span { display: block; text-align: center; padding: 8px 6px; border-radius: 6px;
                  border: 1px solid var(--line); color: var(--ink); cursor: pointer; font-size: .9rem; }
  fieldset input:checked + span { background: var(--ink); color: var(--panel); border-color: var(--ink); }
  fieldset input:focus-visible + span, button:focus-visible, textarea:focus-visible, a:focus-visible
    { outline: 2px solid var(--accent); outline-offset: 2px; }
  button { padding: 10px; font: inherit; font-weight: 600; border: 0; border-radius: 6px;
           background: var(--accent); color: #fff; cursor: pointer; }
  button:disabled { opacity: .55; cursor: progress; }
  #ollama { width: 100%; padding: 8px 10px; font: inherit; font-size: .9rem;
            color: var(--ink); background: var(--bg); border: 1px solid var(--line); border-radius: 6px; }
  #ollama:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  button.ghost { background: none; color: var(--ink); border: 1px solid var(--line); padding: 8px 12px; font-weight: 500; }
  .hint { font-size: .8rem; color: var(--muted); margin: 0; }
  .conn { display: flex; align-items: center; gap: 7px; margin: 6px 0 0; font-size: .8rem; color: var(--muted); }
  .conn .dot { width: 9px; height: 9px; border-radius: 50%; flex: none; background: #8a96a3; }
  .conn.ok .dot { background: #1f9d55; box-shadow: 0 0 0 3px rgba(31,157,85,.18); }
  .conn.warn .dot { background: #d99a06; box-shadow: 0 0 0 3px rgba(217,154,6,.18); }
  .conn.bad .dot { background: #d93025; box-shadow: 0 0 0 3px rgba(217,48,37,.18); }
  .conn.checking .dot { animation: pulse 1s ease-in-out infinite; }
  @keyframes pulse { 50% { opacity: .35; } }
  @media (prefers-reduced-motion: reduce) { .conn.checking .dot { animation: none; } }
  .stage { min-height: min(420px, 60vh); overflow: hidden; border: 1px dashed var(--line); border-radius: 10px;
           display: grid; place-items: center; padding: 12px; }
  .stage a { display: block; max-width: 100%; cursor: zoom-in; }
  .stage img { max-width: 100%; max-height: 75vh; border-radius: 4px; display: block; }
  .status { color: var(--muted); text-align: center; margin: 0; }
  .status.err { color: var(--bad); white-space: pre-wrap; text-align: left; font-size: .9rem; }
  .meta { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-top: 10px;
          font-size: .85rem; color: var(--muted); }
  .meta .prompt { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  dialog { border: 1px solid var(--line); border-radius: 12px; padding: 20px; width: min(400px, calc(100vw - 32px));
           background: var(--panel); color: var(--ink); box-shadow: 0 20px 50px rgba(0,0,0,.3); }
  dialog::backdrop { background: rgba(10, 14, 20, .55); }
  dialog[open] { animation: pop .14s ease-out; }
  @keyframes pop { from { opacity: 0; transform: translateY(6px) scale(.98); } }
  @media (prefers-reduced-motion: reduce) { dialog[open] { animation: none; } }
  dialog form { display: block; padding: 0; border: 0; background: none; }
  dialog h2 { margin: 0 0 8px; font-size: 1.05rem; font-weight: 650; }
  dialog p { margin: 0 0 20px; color: var(--muted); font-size: .92rem; }
  .modal-actions { display: flex; justify-content: flex-end; gap: 8px; }
  .modal-actions button { padding: 8px 16px; }
  .bulk { display: flex; align-items: center; gap: 8px; margin-top: 16px; font-size: .85rem; color: var(--muted); }
  .bulk button { padding: 6px 12px; font-size: .85rem; }
  button.danger { background: var(--bad); color: #fff; }
  .bulk + .strip, .bulk + .bulk + .strip { margin-top: 8px; }
  .strip button { position: relative; }
  .strip.selecting button { border-color: var(--line); }
  .strip.selecting button.picked { border-color: var(--accent); }
  .strip.selecting button::after { content: ""; position: absolute; top: 4px; right: 4px; width: 18px; height: 18px;
    border-radius: 50%; border: 2px solid #fff; background: rgba(0,0,0,.35); box-shadow: 0 0 0 1px rgba(0,0,0,.25); }
  .strip.selecting button.picked::after { content: "✓"; background: var(--accent); color: #fff;
    font-size: 12px; line-height: 14px; text-align: center; font-weight: 700; }
  .strip.selecting button.picked img { opacity: .75; }
  .strip { display: flex; gap: 8px; overflow-x: auto; margin-top: 16px; padding-bottom: 4px; }
  .strip button { padding: 0; background: var(--panel); border: 2px solid transparent; border-radius: 4px; flex: none;
                  width: 76px; height: 76px; display: grid; place-items: center; color: var(--muted); font-size: .75rem; font-weight: 500; }
  .strip button.on { border-color: var(--accent); }
  .strip button.err { color: var(--bad); }
  .strip img { width: 72px; height: 72px; object-fit: cover; border-radius: 3px; display: block; }
</style>
</head>
<body>
<main>
  <form id="f">
    <h1>Generate an image</h1>
    <div>
      <label for="prompt">Prompt</label>
      <textarea id="prompt" required placeholder="A red fox asleep in fresh snow, morning light"></textarea>
    </div>
    <fieldset aria-label="Model">
      <label><input type="radio" name="model" value="x/z-image-turbo" checked><span>Z-Image Turbo</span></label>
      <label><input type="radio" name="model" value="x/flux2-klein"><span>FLUX.2 Klein</span></label>
    </fieldset>
    <div>
      <label for="ollama">Ollama server</label>
      <input id="ollama" type="url" spellcheck="false" autocomplete="off">
      <p class="conn" id="conn" role="status" aria-live="polite"><span class="dot"></span><span id="connText">Checking…</span></p>
    </div>
    <button id="go" type="submit">Generate</button>
    <p class="hint">Ctrl/⌘ + Enter to generate. Jobs run on the server, so you can close this tab and come back later.</p>
  </form>

  <section>
    <div class="stage" id="stage"><p class="status">Your image will appear here.</p></div>
    <div class="meta" id="meta" hidden>
      <span class="prompt" id="info"></span>
    </div>
    <div class="bulk" id="bulk" hidden>
      <button type="button" id="selAll" class="ghost">Select all</button>
      <button type="button" id="selDl" disabled>Download</button>
      <button type="button" id="selDel" class="danger" disabled>Delete</button>
      <button type="button" id="selDone" class="ghost">Cancel</button>
      <span id="bulkCount"></span>
    </div>
    <div class="bulk" id="bulkOff" hidden>
      <button type="button" id="selStart" class="ghost">Select</button>
    </div>
    <div class="strip" id="strip"></div>
  </section>
</main>
<dialog id="modal" aria-labelledby="modalTitle" aria-describedby="modalMsg">
  <form method="dialog">
    <h2 id="modalTitle"></h2>
    <p id="modalMsg"></p>
    <div class="modal-actions">
      <button value="cancel" class="ghost" id="modalCancel">Cancel</button>
      <button value="ok" id="modalOk">OK</button>
    </div>
  </form>
</dialog>
<script>
const $ = s => document.querySelector(s);
const KEY = 'ollama-img-web:url';
const ollamaInput = $('#ollama');
let jobs = [], selected = null, shownKey = '', pollTimer;
let cleared = false;   // true when the user deselected the viewed image on purpose

try { ollamaInput.value = localStorage.getItem(KEY) || ''; } catch {}
fetch('health').then(r => r.json()).then(d => { ollamaInput.placeholder = d.ollama; }).catch(() => {});
ollamaInput.addEventListener('change', () => { try { localStorage.setItem(KEY, ollamaInput.value.trim()); } catch {} clearTimeout(debounce); checkOllama(); });

let checkTimer, checkSeq = 0;
function conn(state, msg) {
  $('#conn').className = 'conn ' + state;
  $('#connText').textContent = msg;
}
async function checkOllama() {
  clearTimeout(checkTimer);
  const seq = ++checkSeq;
  conn('checking', 'Checking…');
  try {
    const r = await fetch('api/check?url=' + encodeURIComponent(ollamaInput.value.trim()));
    const d = await r.json();
    if (seq !== checkSeq) return;          // a newer check superseded this one
    if (!r.ok) throw new Error(d.error);
    const host = d.url.replace(/^https?:\/\//, '');
    if (d.missing.length) conn('warn', `Connected to ${host}, but missing ${d.missing.join(', ')}`);
    else conn('ok', `Connected to ${host}`);
  } catch (err) {
    if (seq !== checkSeq) return;
    conn('bad', err.message || 'Cannot reach Ollama');
  }
  if (!document.hidden) checkTimer = setTimeout(checkOllama, 30000);
}
let debounce;
ollamaInput.addEventListener('input', () => { clearTimeout(debounce); debounce = setTimeout(checkOllama, 600); });

function statusText(j) {
  if (j.status === 'queued') return j.ahead ? `Queued, ${j.ahead} ahead of it.` : 'Starting…';
  if (j.status === 'running') return `Generating with ${j.model}… ${Math.round(j.secs || 0)}s`;
  return 'Generation failed.\n' + (j.error || '');
}

function renderStage() {
  const j = jobs.find(x => x.id === selected);
  if (!j) { shownKey = ''; $('#stage').innerHTML = '<p class="status">Your image will appear here.</p>'; $('#meta').hidden = true; return; }
  const key = j.id + j.status;
  if (j.status === 'done') {
    if (key !== shownKey) {
      const img = new Image(); img.decoding = 'async';
      img.onerror = () => { $('#stage').innerHTML = '<p class="status err">Image file is missing on the server.</p>'; };
      img.src = `images/${j.id}.png`; img.alt = j.prompt;
      const link = document.createElement('a');
      link.href = img.src; link.target = '_blank'; link.rel = 'noopener';
      link.title = 'Open full size'; link.append(img);
      $('#stage').replaceChildren(link);
    }
    $('#info').textContent = `${j.model} · ${j.secs}s · ${j.prompt}`;
  } else {
    let p = $('#stage p');
    if (key !== shownKey || !p) { p = document.createElement('p'); $('#stage').replaceChildren(p); }
    p.className = 'status' + (j.status === 'error' ? ' err' : '');
    p.textContent = statusText(j);
    $('#info').textContent = j.prompt;
  }
  $('#meta').hidden = false;
  shownKey = key;
}

function confirmModal({title, message = '', okText = 'OK', danger = false}) {
  const dlg = $('#modal');
  $('#modalTitle').textContent = title;
  $('#modalMsg').textContent = message;
  $('#modalMsg').hidden = !message;
  $('#modalOk').textContent = okText;
  $('#modalOk').className = danger ? 'danger' : '';
  dlg.returnValue = '';
  return new Promise(resolve => {
    dlg.addEventListener('close', () => resolve(dlg.returnValue === 'ok'), {once: true});
    dlg.showModal();
    $('#modalCancel').focus();           // safe default for destructive actions
  });
}
// clicking the dimmed backdrop cancels
$('#modal').addEventListener('click', e => { if (e.target === e.currentTarget) e.currentTarget.close('cancel'); });

let stripKey = '';
let selecting = false;
const picked = new Set();
const pickable = j => j.status !== 'running';

function renderBulk() {
  // drop picks for jobs that no longer exist or started running
  for (const id of picked) { const j = jobs.find(x => x.id === id); if (!j || !pickable(j)) picked.delete(id); }
  $('#bulkOff').hidden = selecting || jobs.length === 0;
  $('#bulk').hidden = !selecting;
  if (selecting && jobs.length === 0) selecting = false;
  const n = picked.size, total = jobs.filter(pickable).length;
  $('#bulkCount').textContent = n ? `${n} selected` : '';
  $('#selDel').disabled = n === 0;
  $('#selDl').disabled = pickedDone().length === 0;
  $('#selAll').textContent = n && n === total ? 'Select none' : 'Select all';
}

function renderStrip() {
  const strip = $('#strip');
  const key = selected + '|' + selecting + '|' + [...picked].join(',') + '|' + jobs.map(j => j.id + j.status).join(',');
  if (key === stripKey) return;
  stripKey = key;
  strip.classList.toggle('selecting', selecting);
  strip.replaceChildren(...jobs.map(j => {
    const b = document.createElement('button');
    b.type = 'button'; b.title = j.prompt;
    b.className = (!selecting && j.id === selected ? 'on ' : '') + (j.status === 'error' ? 'err ' : '') + (picked.has(j.id) ? 'picked' : '');
    if (selecting) b.setAttribute('aria-pressed', picked.has(j.id));
    if (selecting && !pickable(j)) b.disabled = true;
    if (j.status === 'done') b.innerHTML = `<img src="images/${j.id}.png" alt="" loading="lazy" decoding="async">`;
    else b.textContent = j.status === 'error' ? 'Failed' : j.status === 'running' ? 'Working' : 'Queued';
    b.onclick = () => {
      if (selecting) { picked.has(j.id) ? picked.delete(j.id) : picked.add(j.id); }
      else if (selected === j.id) { selected = null; cleared = true; }
      else { selected = j.id; cleared = false; }
      render();
    };
    return b;
  }));
}

$('#selStart').onclick = () => { selecting = true; picked.clear(); render(); };
$('#selDone').onclick = () => { selecting = false; picked.clear(); render(); };
$('#selAll').onclick = () => {
  const all = jobs.filter(pickable).map(j => j.id);
  if (picked.size === all.length) picked.clear(); else all.forEach(id => picked.add(id));
  render();
};
function pickedDone() {
  return jobs.filter(j => picked.has(j.id) && j.status === 'done').map(j => j.id);
}
$('#selDl').onclick = () => {
  const ids = pickedDone();
  if (!ids.length) return;
  const a = document.createElement('a');
  a.href = ids.length === 1 ? `images/${ids[0]}.png` : 'api/download?ids=' + ids.join(',');
  a.download = ids.length === 1 ? `ollama-${ids[0].slice(0, 8)}.png` : '';
  document.body.append(a); a.click(); a.remove();
};

$('#selDel').onclick = async () => {
  const n = picked.size;
  if (!n) return;
  const ok = await confirmModal({
    title: `Delete ${n} image${n > 1 ? 's' : ''}?`,
    message: "This can't be undone.",
    okText: 'Delete', danger: true,
  });
  if (!ok) return;
  $('#selDel').disabled = true;
  try {
    await fetch('api/jobs/delete', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                    body: JSON.stringify({ids: [...picked]})});
  } catch {}
  picked.clear(); selecting = false;
  await refresh();
};

function render() {
  if (!jobs.some(j => j.id === selected)) selected = cleared ? null : (jobs[0]?.id ?? null);
  renderStage(); renderBulk(); renderStrip();
}

async function refresh() {
  clearTimeout(pollTimer);
  try {
    const r = await fetch('api/jobs');
    if (r.ok) {
      const prevFailed = new Set(jobs.filter(j => j.status === 'error').map(j => j.id));
      jobs = await r.json(); render();
      if (jobs.some(j => j.status === 'error' && !prevFailed.has(j.id))) checkOllama();
    }
  } catch {}
  const busy = jobs.some(j => j.status === 'queued' || j.status === 'running');
  pollTimer = setTimeout(refresh, busy ? 1500 : 15000);
}

$('#prompt').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) $('#f').requestSubmit();
});

$('#f').addEventListener('submit', async e => {
  e.preventDefault();
  const prompt = $('#prompt').value.trim();
  const model = document.querySelector('input[name=model]:checked').value;
  if (!prompt) return;
  $('#go').disabled = true;
  try {
    const r = await fetch('api/generate', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, model, ollama_url: ollamaInput.value.trim()})
    });
    const d = await r.json().catch(() => ({error: `Server returned ${r.status}.`}));
    if (!r.ok) throw new Error(d.error);
    selected = d.id; cleared = false;
    await refresh();
  } catch (err) {
    selected = null; shownKey = '';
    $('#stage').innerHTML = ''; const p = document.createElement('p');
    p.className = 'status err'; p.textContent = 'Could not start generation.\n' + err.message;
    $('#stage').append(p); $('#meta').hidden = true;
  } finally {
    $('#go').disabled = false;
  }
});

document.addEventListener('visibilitychange', () => {
  if (document.hidden) clearTimeout(checkTimer);
  else { refresh(); checkOllama(); }
});
refresh();
checkOllama();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    server_version = "ollama-img-web/2.0"

    def _authorized(self):
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        if header in auth_ok_cache:
            return True
        try:
            user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
        except Exception:
            return False
        if not check_login(user, pw):
            return False
        if len(auth_ok_cache) > 100:
            auth_ok_cache.clear()
        auth_ok_cache.add(header)
        return True

    def _deny(self):
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="ollama-img"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send(self, code, body, ctype="application/json", cache="no-store"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj))

    def do_GET(self):
        if not self._authorized():
            return self._deny()
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if path == "/health":
            return self._json(200, {"ok": True, "ollama": OLLAMA_URL})
        if path == "/api/jobs":
            with jobs_lock:
                snapshot = sorted(jobs.values(), key=lambda j: j["created"], reverse=True)
            return self._json(200, [public(j) for j in snapshot])
        m = re.fullmatch(r"/images/([0-9a-f]{32})\.png", path)
        if m:
            try:
                return self._send(200, read_image(m.group(1)), "image/png", "private, max-age=31536000, immutable")
            except FileNotFoundError:
                return self._send(404, "not found", "text/plain")
        if path == "/api/download":
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            ids = [i for i in qs.get("ids", [""])[0].split(",") if ID_RE.match(i)]
            buf, added = io.BytesIO(), 0
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:   # PNGs are already compressed
                for jid in dict.fromkeys(ids):
                    try:
                        z.writestr(f"ollama-{jid[:8]}.png", read_image(jid))
                        added += 1
                    except FileNotFoundError:
                        pass
            if not added:
                return self._send(404, "no images found", "text/plain")
            name = time.strftime("ollama-images-%Y%m%d-%H%M%S.zip")
            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/check":
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                base = resolve_ollama(qs.get("url", [""])[0])
                with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as r:
                    names = {m.get("name", "") for m in json.loads(r.read()).get("models", [])}
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            except Exception as e:
                reason = getattr(e, "reason", e)
                return self._json(502, {"error": f"Cannot reach Ollama at {base} ({reason})"})
            have = lambda m: any(n == m or n.startswith(m + ":") for n in names)
            return self._json(200, {"url": base, "missing": [m for m in MODELS if not have(m)]})
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        if not self._authorized():
            return self._deny()
        if self.path not in ("/api/generate", "/api/jobs/delete"):
            return self._send(404, "not found", "text/plain")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "Request body must be JSON."})

        if self.path == "/api/jobs/delete":
            if req.get("all"):
                with jobs_lock:
                    ids = list(jobs)
            else:
                ids = [i for i in req.get("ids", []) if isinstance(i, str) and ID_RE.match(i)]
            deleted, skipped = [], []
            for jid in ids:
                (skipped if delete_job(jid) else deleted).append(jid)
            return self._json(200, {"deleted": deleted, "skipped": skipped})

        prompt = str(req.get("prompt", "")).strip()
        model = req.get("model", MODELS[0])
        if not prompt:
            return self._json(400, {"error": "Prompt is empty."})
        if model not in MODELS:
            return self._json(400, {"error": f"Unknown model '{model}'. Use one of: {', '.join(MODELS)}"})
        try:
            base = resolve_ollama(req.get("ollama_url"))
        except ValueError as e:
            return self._json(400, {"error": str(e)})

        job = {"id": uuid.uuid4().hex, "prompt": prompt, "model": model, "ollama": base,
               "status": "queued", "created": time.time(), "started": None, "finished": None, "error": None}
        with jobs_lock:
            jobs[job["id"]] = job
        work.put(job)
        self._json(202, public(job))

    def do_DELETE(self):
        if not self._authorized():
            return self._deny()
        m = re.fullmatch(r"/api/jobs/([0-9a-f]{32})", self.path)
        if not m:
            return self._send(404, "not found", "text/plain")
        reason = delete_job(m.group(1))
        if reason == "not found":
            return self._json(404, {"error": "No such job."})
        if reason == "running":
            return self._json(409, {"error": "Can't delete a job while it's generating."})
        self._json(200, {"deleted": m.group(1)})

    def log_message(self, fmt, *args):
        if "/api/jobs" in (args[0] if args else ""):
            return  # skip polling noise
        sys.stderr.write(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Web UI for Ollama image models")
    ap.add_argument("-o", "--ollama", help=f"Ollama base URL (default {OLLAMA_URL})")
    ap.add_argument("-p", "--port", type=int, default=PORT, help=f"Port to listen on (default {PORT})")
    ap.add_argument("--images", default=IMAGES_DIR, help=f"Image folder (default {IMAGES_DIR})")
    ap.add_argument("--reset-auth", action="store_true", help="Choose a new username/password")
    args = ap.parse_args()
    setup_auth(reset=args.reset_auth)
    try:
        OLLAMA_URL = resolve_ollama(args.ollama)
    except ValueError as e:
        sys.exit(str(e))
    PORT, IMAGES_DIR = args.port, args.images

    load_saved()
    threading.Thread(target=worker, daemon=True).start()
    print(f"Serving on http://0.0.0.0:{PORT}  ->  Ollama at {OLLAMA_URL}")
    print(f"Images saved to {IMAGES_DIR} ({len(jobs)} loaded)")
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            names = {m.get("name", "") for m in json.loads(r.read()).get("models", [])}
        missing = [m for m in MODELS if not any(n == m or n.startswith(m + ":") for n in names)]
        print("Ollama: connected" + (f" (missing models: {', '.join(missing)})" if missing else ", both models available"))
    except Exception as e:
        print(f"Ollama: NOT reachable at {OLLAMA_URL} ({getattr(e, 'reason', e)}). "
              "Starting anyway; you can set a different URL in the page.")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True             # don't wait on open connections at exit

    def _stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)     # systemd/kill get the same clean exit as Ctrl+C

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down…")
        server.server_close()
        with jobs_lock:
            unfinished = sum(1 for j in jobs.values() if j["status"] in ("queued", "running"))
        if unfinished:
            print(f"{unfinished} unfinished job{'s' if unfinished != 1 else ''} discarded.")
        print("Stopped.")