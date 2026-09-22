# ClineGate

**One local gateway for all your Cline accounts.** ClineGate sits between your
tools and Cline's API: it pools every Cline account you import, load-balances
across them, refreshes their tokens automatically, and serves both
OpenAI-compatible and Anthropic-compatible endpoints on your machine.

Single-file Windows app. No Python, no installer, no setup.

---

## Download (prebuilt)

Grab **`ClineGateway.exe`** from the
[**Releases**](https://github.com/B3hnamR/ClineGate/releases) page and run it.
That's the whole install.

- Everything it needs is created next to the exe on first launch
  (`config.yaml`, `accounts\`, `logs\`, `gateway.db`).
- Keys are generated randomly per install — no shared defaults.
- Closing the window quits the tool. Relaunch the exe to start it again.

> **Windows SmartScreen** may warn on first run because the exe is not yet
> code-signed. Verify the SHA-256 checksum published on the release page if
> you want to be sure you got the exact file we built.

## First account in 60 seconds

1. Open ClineGate → **Accounts** → **🔑 Login with Cline**.
2. A short code appears — click **Open login page** and approve it in your
   browser (a Google account works; new accounts get free starting credit).
3. The account registers, lands in `accounts\`, and starts serving — no
   reload needed.

No Cline install required. Already have the Cline desktop app signed in?
**⤓ Import from Cline app** snapshots that account instead — same result.
Both paths are explained in the app's built-in **Guide** tab, along with
multiple accounts, moving accounts between machines, and connecting clients.

## Use it from your tools

Point any OpenAI- or Anthropic-compatible client at the local gateway (the
**Server** tab shows your keys and exact URLs):

```text
base_url: http://127.0.0.1:8787/v1
api_key:  <client key from the Server tab>
```

Works with the OpenAI SDK, Anthropic SDK, Kilo, Cline's own
openai-compatible provider, or plain curl:

```powershell
curl http://127.0.0.1:8787/v1/chat/completions `
  -H "Authorization: Bearer <client key>" `
  -H "Content-Type: application/json" `
  -d '{ "model": "cline-free/deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": "hi"}] }'
```

## What it does

- **Login with Cline** — add accounts by approving a short device code in your
  browser. No Cline desktop install, no token copying; the account is
  registered, snapshotted, and pooled automatically.
- **Multi-account pool** with load balancing (least-in-flight, round-robin,
  LRU, quota-aware) and per-account in-flight caps. Balance and paid-lane
  state settle the moment an account is added or checked.
- **Automatic token refresh** — access tokens expire hourly; ClineGate renews
  them in the background and retries transparently on a 401.
- **Lane-aware failover** — free models, credit-billed models, and
  subscription models fail differently, and the pool treats them differently:
  a daily cap parks one model, a drained balance retires only the paid lane.
- **Both dialects** — `/v1/chat/completions` (OpenAI) and `/v1/messages`
  (Anthropic), streaming and non-streaming.
- **Live dashboard** — accounts, per-model availability, stats, logs,
  settings, and a setup guide.
- **Self-updating** — a 🔔 appears when a new release is out; one click
  downloads, swaps, and relaunches.

## Updating

Automatic: when a release is published, the running app shows a bell icon.
Click it → **Update now**. Done.

Manual: download the new exe from Releases and replace the old file (keep your
`config.yaml`, `accounts\`, and `gateway.db` — they are your data).

## Build from source

Requires Windows, Python 3.13, and ~5 minutes:

```powershell
git clone https://github.com/B3hnamR/ClineGate.git
cd ClineGate
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt pyinstaller
.\build-exe.ps1
# output: dist\ClineGateway.exe  (~31 MB, single file)
```

Run from source without building:

```powershell
.\.venv\Scripts\python -m cline_gateway.main --port 8787
# dashboard: http://127.0.0.1:8787/dash
```

Tests:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest tests -q
```

## Configuration

Everything lives in `config.yaml` next to the exe (created on first run, safe
to edit — restart applies). The in-app **Settings** tab covers the common
knobs: pool strategy, cooldowns, timeouts, accounts folder, logging, and
update checks. All settings can also be overridden with environment variables:
`CLINE_GATEWAY_<SECTION>__<FIELD>` (e.g. `CLINE_GATEWAY_SERVER__PORT=9999`).

## Security & privacy notes

- The gateway binds to `127.0.0.1` by default and requires API keys on every
  route; the keys are generated per install and shown on the Server tab.
- Account snapshots in `accounts\` contain live credentials. They never leave
  your machine — but treat the folder like a password file, and don't commit
  or share it.
- The update check makes one request to the GitHub API per interval
  (default 6 h). Disable it in Settings → Updates.

## Disclaimer

ClineGate is an unofficial, independent project and is not affiliated with or
endorsed by Cline. "Cline" is a trademark of its respective owner. ClineGate
works by reproducing the Cline desktop client's API contract — using it with
your own accounts is your responsibility; review Cline's terms of service.

## License

MIT — see [LICENSE](LICENSE). Coded by [@B3hnamR](https://t.me/B3hnamR).
