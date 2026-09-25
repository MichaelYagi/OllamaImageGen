# OllamaImageGen

A small web UI for generating images with Ollama's image models, **x/z-image-turbo** and **x/flux2-klein**.

It's a single Python file with no dependencies beyond the standard library. The server serves the page and talks to Ollama for you, so the browser never needs direct access to Ollama. Generation runs in the background on the server, so you can close the tab, lock your phone, or switch devices, and your images will be waiting when you come back.

## Features

- Pick a model, type a prompt, and generate. Press Ctrl/⌘ + Enter as a shortcut.
- Jobs are queued and run one at a time, and the page shows each job's place in the queue.
- Finished images are saved to disk and appear in a gallery strip on every device.
- Click an image to open it full size in a new tab.
- Multi-select lets you download images (a single PNG, or a zip for several) or delete them in bulk.
- A live status dot shows whether Ollama is reachable and both models are installed.
- The Ollama server URL can be set per browser, right on the page.
- HTTP Basic Auth login is set up in the terminal on first launch, and the password is stored hashed.
- Works on mobile and supports dark mode.

## Requirements

- Python 3.8 or newer. No `pip install` needed.
- Ollama **0.32.0 to 0.32.5**, running on **macOS**. See [Setting up Ollama](#setting-up-ollama).
- Optional: [tunnelmole](https://tunnelmole.com) (`tmole`) or any other tunnel, for access from outside your network.

The web server itself can run on any machine that can reach the Mac running Ollama.

## Setting up Ollama

Ollama must run on **macOS**, at a version from **0.32.0 to 0.32.5**. Other versions aren't supported.

### 1. Install a supported version

Install with the official script, setting `OLLAMA_VERSION` to a version in the supported range:

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.32.5 sh
```

Any of `0.32.0`, `0.32.1`, `0.32.2`, `0.32.3`, `0.32.4` or `0.32.5` works. Note that `OLLAMA_VERSION` goes after the `|`, on the `sh` side.

Check the installed version:

```bash
ollama --version
```

### 2. Turn off automatic updates

The Ollama app updates itself by default, which would move it out of the supported range.

1. Open the Ollama app and go to **Settings**.
2. Turn **Auto-download updates** off.

If Ollama does update, run the install command from step 1 again to go back to a supported version. Downloaded models are kept.

### 3. Pull the image models

```bash
ollama pull x/z-image-turbo
ollama pull x/flux2-klein
```

### 4. Allow access from other machines

If the web server runs on a different machine from Ollama, Ollama has to listen on the network, not only on localhost:

```bash
launchctl setenv OLLAMA_HOST "0.0.0.0"
```

Then quit and reopen the Ollama app. You can confirm access from the machine running the web server:

```bash
curl http://<ollama-host>:11434/api/version
```

## Quick start

```bash
git clone git@github.com:MichaelYagi/imagegen.git
cd imagegen
python3 ollama-img-web.py -o http://<ollama-host>:11434
```

Replace `<ollama-host>` with the hostname or LAN address of the machine running Ollama. If Ollama runs on the same machine, you can leave out `-o`.

Open http://localhost:8080 and sign in with the login you chose (see below).

### First launch: choosing a login

The first time you run the server, it asks for a username and password in the terminal:

```
First launch: choose a login for the web page.
  Username: mike
  Password:
  Confirm password:
  Saved to /home/mike/.config/ollama-img-web/auth.json
```

On every later launch, it loads the saved login without asking, and prints one line to confirm:

```
Login: user 'mike' (change with --reset-auth)
```

A few details about the saved login:

- It's saved to `~/.config/ollama-img-web/auth.json`, outside the repo, with permissions `600`. The password is stored as a salted PBKDF2-SHA256 hash, not as plain text.
- To change it, run `python3 ollama-img-web.py --reset-auth`.
- Each machine keeps its own login, because the file lives in that user's home folder.
- If the server starts without a terminal (systemd, Docker, `nohup`) and no login is saved, it refuses to start. Either run it once in a terminal to set a login, or supply `APP_USER` and `APP_PASSWORD` (see [Options](#options)).

The login is set in the terminal, not through a setup page, on purpose. When the server is reachable through a public tunnel, a browser setup page would let whoever opened the URL first choose the login.

## Accessing it remotely

Start the server, then tunnel its port:

```bash
python3 ollama-img-web.py -o http://<ollama-host>:11434
tmole 8080
```

Open the `https://….tunnelmole.net` URL that `tmole` prints, from any device.

- **You only need one tunnel.** Point the server at Ollama over your LAN (`http://<ollama-host>:11434`) and tunnel only the web page. Exposing Ollama through its own tunnel makes its whole API public with no authentication.
- **On the machine running the server, use `http://localhost:8080`.** Going through the tunnel sends every image out to the internet and back.
- **Free tunnelmole URLs change each time `tmole` restarts.**

## Options

The Ollama URL can be set three ways. If more than one is set, the later one in this list wins:

1. The `OLLAMA_URL` environment variable
2. The `--ollama` / `-o` command-line flag
3. The **Ollama server** field on the page, which is saved in that browser. Leave it blank to use the server's default.

| Flag | Env var | Default | Description |
|---|---|---|---|
| `-o`, `--ollama` | `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `-p`, `--port` | `PORT` | `8080` | Port to listen on |
| `--images` | `IMAGES_DIR` | `./images` next to the script | Where images are saved |
| `--reset-auth` | | | Choose a new username and password |
| | `APP_USER`, `APP_PASSWORD` | | Override the saved login. Both must be set. |
| | `GEN_TIMEOUT` | `900` | Seconds to wait on Ollama per image |

Examples:

```bash
# Different port
python3 ollama-img-web.py -o http://<ollama-host>:11434 -p 8081

# Keep images somewhere else
python3 ollama-img-web.py --images ~/ollama-images

# Login from the environment, e.g. for a service
APP_USER=mike APP_PASSWORD='…' python3 ollama-img-web.py
```

When the server starts, it checks Ollama once and prints the result, so a wrong URL shows up immediately:

```
Ollama: connected, both models available
Ollama: NOT reachable at http://<ollama-host>:11434 (Connection refused). Starting anyway; ...
```

## Using the page

### Generating

Type a prompt, pick **Z-Image Turbo** or **FLUX.2 Klein**, and click **Generate**. You can queue more jobs while one is running. They run one at a time on the GPU.

### Ollama status

The dot under the **Ollama server** field shows the connection state:

| Dot | Meaning |
|---|---|
| 🟢 Green | Connected, and both models are installed |
| 🟡 Amber | Connected, but a model is missing (the message says which one) |
| 🔴 Red | Ollama can't be reached (the message says why) |
| ⚪ Pulsing grey | Checking |

The page checks the connection when it loads, shortly after you finish editing the URL, every 30 seconds while the tab is visible, when you return to the tab, and after a failed generation.

### The gallery

Thumbnails of every job appear under the main image. Click one to view it, and click the main image to open it full size in a new tab.

To download or delete images, click **Select** and tap the images you want. Selected images get a blue checkmark.

- **Select all** / **Select none** toggles every image.
- **Download** downloads a single image as a PNG, or several images as one zip. Only finished images are included.
- **Delete** asks for confirmation, then deletes the selected images.
- **Cancel** leaves selection mode without changing anything.

Images that are currently generating can't be selected. Deleting a queued job cancels it before it starts.

## Where things are stored

| What | Where |
|---|---|
| Images | `images/<id>.png`, with a `<id>.json` next to each one holding the prompt, model and timing |
| Login | `~/.config/ollama-img-web/auth.json` |
| Ollama URL set on the page | That browser's local storage |

Finished images survive restarts. Jobs that are still queued or generating when the server stops are lost.

Keep generated images out of git:

```bash
echo "images/" >> .gitignore
```

## Running in the background

To keep the server running after you close the terminal, use `tmux`/`screen`, `nohup`, or a systemd user service. Set up the login first by running the server once in a terminal, or provide `APP_USER` and `APP_PASSWORD`.

Example systemd unit at `~/.config/systemd/user/imagegen.service`:

```ini
[Unit]
Description=imagegen web UI

[Service]
WorkingDirectory=%h/imagegen
ExecStart=/usr/bin/python3 %h/imagegen/ollama-img-web.py -o http://<ollama-host>:11434
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now imagegen
```

## API

Every endpoint requires the Basic Auth login.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | The web page |
| `GET` | `/health` | Server status and the default Ollama URL |
| `GET` | `/api/check?url=…` | Test an Ollama URL. Reports which models are missing. |
| `POST` | `/api/generate` | Queue a job: `{"prompt", "model", "ollama_url"?}`. Returns the job. |
| `GET` | `/api/jobs` | All jobs, newest first |
| `GET` | `/images/<id>.png` | A finished image |
| `GET` | `/api/download?ids=a,b,…` | Zip of the given finished images |
| `POST` | `/api/jobs/delete` | Delete `{"ids": [...]}` or everything with `{"all": true}` |
| `DELETE` | `/api/jobs/<id>` | Delete one job |

Example:

```bash
curl -u mike:PASSWORD http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox asleep in fresh snow", "model": "x/z-image-turbo"}'
```

## Troubleshooting

**`OSError: [Errno 48] Address already in use`** (on Linux, `Errno 98`)
Another program is using the port. Pick another port with `-p 8081` and tunnel that port instead (`tmole 8081`), or find what's using it:
```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

**The login keeps failing (401)**
- Check for leftover env vars with `env | grep APP_`, since they override the saved login. Clear them with `unset APP_USER APP_PASSWORD`.
- Reset the login with `--reset-auth`.
- Browsers remember a failed Basic Auth login, so try a private or incognito window.
- Test the server directly:
  ```bash
  curl -s -o /dev/null -w "%{http_code}\n" -u USER:PASS http://localhost:8080/health
  ```

**The status dot is red**
Check the address in the **Ollama server** field. Make sure Ollama listens on the network and not only on localhost (see [step 4 of Setting up Ollama](#4-allow-access-from-other-machines)), and that no firewall is blocking port 11434.

**Generation stopped working after Ollama updated**
Run `ollama --version` on the Mac. If it's outside 0.32.0 to 0.32.5, reinstall a supported version and turn off **Auto-download updates** (see [Setting up Ollama](#setting-up-ollama)).

**Long generations fail when Ollama is reached through a tunnel**
The page itself isn't affected, because it only sends short requests to check on jobs. The request from the server to Ollama, though, stays open for the whole generation, and some tunnels close long requests. Point the server at Ollama over the LAN instead of through a tunnel.

**Slow on Windows with WSL**
Files under `/mnt/c` are slow to access from WSL. Keep the images on the Linux side:
```bash
python3 ollama-img-web.py --images ~/ollama-images
```

**PyCharm shows files as modified when `git status` is clean**
PyCharm on Windows uses Windows' git, which can disagree with WSL's git about file permissions. Tell git to ignore permission changes:
```bash
git config core.fileMode false
```

## Security notes

- Always use the login when the server is reachable from outside your network. Anyone with the URL can otherwise use your GPU.
- Basic Auth sends the login with every request. That's safe over HTTPS, which tunnelmole provides. On plain `http://`, use it only on your own network.
- The **Ollama server** field lets anyone who is signed in point the server at another address. The server only calls Ollama API paths on that address.
