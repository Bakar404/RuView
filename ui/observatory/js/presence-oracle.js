/**
 * Presence Oracle client.
 *
 * Polls the background-reference presence sidecar (scripts/presence_service.py)
 * and publishes the result on `window.__presenceOracle`, which hud-controller.js
 * treats as authoritative for the presence indicator when it is live.
 *
 * Why this exists
 * ---------------
 * The sensing server derives presence from motion:
 *     v2/crates/wifi-densepose-sensing-server/src/main.rs:2681
 *         presence: motion_score > 0.04
 * All motion inputs are change features and the baseline EMA
 * (BASELINE_EMA_ALPHA = 0.003, main.rs:2712) absorbs a static body in ~30 s, so
 * a person sitting still is reported ABSENT. The sidecar instead compares the
 * absolute per-subcarrier amplitude profile to a calibrated empty-room
 * reference, which does not decay.
 *
 * The panel stays hidden if the sidecar is not running, so the observatory
 * behaves exactly as before when it is absent.
 */

// Derive the sidecar host from the page, so this works unchanged whether the
// dashboard is opened on the sensing machine, over an SSH tunnel, or directly
// across the LAN. Hardcoding 127.0.0.1 would resolve to the *viewing* laptop.
// Override for unusual setups with:  localStorage.oracleUrl = 'http://host:5008'
const DEFAULT_URL = `http://${location.hostname || '127.0.0.1'}:5008`;

export class PresenceOracle {
  constructor(baseUrl = DEFAULT_URL) {
    this.base = localStorage.getItem('oracleUrl') || baseUrl;
    this.data = null;
    this.failures = 0;
    this._timer = null;
    this._bindUi();
  }

  start(intervalMs = 1000) {
    if (this._timer) clearInterval(this._timer);
    void this._poll();
    this._timer = setInterval(() => void this._poll(), intervalMs);
  }

  async _poll() {
    let fresh;
    try {
      const r = await fetch(`${this.base}/presence`, {
        signal: AbortSignal.timeout(1500),
        cache: 'no-store',
      });
      if (!r.ok) throw new Error(String(r.status));
      fresh = await r.json();
      this.data = fresh;
      this.failures = 0;
      window.__presenceOracle = fresh;
    } catch {
      this.failures++;
      // Tolerate a couple of dropped polls before disowning the indicator, so a
      // single slow response does not make presence flicker back to the
      // motion-based estimate.
      if (this.failures >= 3) {
        window.__presenceOracle = null;
        this.data = null;
        const block = document.getElementById('oracle-block');
        if (block) block.style.display = 'none';
      }
      return;
    }
    // Render outside the fetch try/catch: a render bug must surface in the
    // console as a real error, not be miscounted as a connection failure.
    this._render(fresh);
  }

  async calibrate(seconds = 120) {
    try {
      await fetch(`${this.base}/calibrate?seconds=${seconds}`, { method: 'POST' });
    } catch { /* sidecar not reachable */ }
  }

  _bindUi() {
    const btn = document.getElementById('oracle-calibrate');
    if (!btn) return;
    btn.addEventListener('click', () => {
      if (!confirm(
        'Calibrate the empty-room reference?\n\n' +
        'The apartment must be EMPTY for the next 2 minutes, or the reference ' +
        'will learn your body as "normal" and presence detection will stop ' +
        'working until you recalibrate.\n\nStart now?')) return;
      void this.calibrate(120);
    });
  }

  _render(d) {
    const block = document.getElementById('oracle-block');
    if (!block) return;
    block.style.display = 'block';

    const badge = document.getElementById('oracle-status');
    const label = {
      ok: 'LIVE', stale: 'STALE REF', calibrating: 'CALIBRATING',
      uncalibrated: 'NO REF',
    }[d.status] || d.status;
    badge.textContent = d.live ? label : 'NO CSI';
    badge.className = 'oracle-badge oracle-badge--' +
      (!d.live ? 'warn' : d.status === 'ok' ? 'ok' : 'warn');

    if (d.status === 'calibrating') {
      document.getElementById('oracle-score').textContent =
        `${Math.round(d.calibrating_remaining)}s left`;
    } else {
      document.getElementById('oracle-score').textContent =
        `${(d.score * 100).toFixed(1)}%`;
    }

    const bar = document.getElementById('oracle-bar');
    bar.style.width = `${Math.min(100, d.score * 100)}%`;
    bar.style.background = d.present ? '#39d98a' : '#4a6a8a';

    const nodes = document.getElementById('oracle-nodes');
    nodes.innerHTML = Object.entries(d.per_node || {})
      .sort((a, b) => Number(a[0]) - Number(b[0]))
      .map(([id, v]) => {
        const pct = (v * 100).toFixed(0);
        const hot = v > d.threshold;
        return `<span class="oracle-node ${hot ? 'oracle-node--hot' : ''}">` +
               `n${id} ${pct}%</span>`;
      }).join('');

    const meta = document.getElementById('oracle-meta');
    const age = d.calibration_age_hours;
    let m = `thr ${(d.threshold * 100).toFixed(0)}% · held ${Math.round(d.held_for)}s`;
    if (age != null) m += ` · ref ${age < 1 ? `${Math.round(age * 60)}m` : `${age.toFixed(1)}h`} old`;
    if (d.status === 'stale') m += ' · RECALIBRATE';
    meta.textContent = m;

    this._sparkline(d.history || [], d.threshold);
  }

  _sparkline(hist, thresh) {
    const c = document.getElementById('oracle-sparkline');
    if (!c || !hist.length) return;
    const ctx = c.getContext('2d');
    const w = c.width, h = c.height;
    ctx.clearRect(0, 0, w, h);

    // Threshold line: everything above it is a detection.
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(0, h - thresh * h);
    ctx.lineTo(w, h - thresh * h);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.beginPath();
    hist.forEach((p, i) => {
      const x = (i / Math.max(1, hist.length - 1)) * w;
      const y = h - Math.min(1, p.score) * h;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = '#39d98a';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}
