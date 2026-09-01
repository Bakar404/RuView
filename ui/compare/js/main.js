/**
 * Ground-truth comparison: camera vs the two CSI presence estimators.
 *
 * Three verdicts are produced independently, and none can observe the others:
 *
 *   CAMERA   MediaPipe pose landmarker on the local video stream. Treated as
 *            ground truth for "a person is in frame". Runs fully offline from
 *            ui/vendor/mediapipe; nothing is uploaded.
 *   SHIPPED  The sensing server's own flag over ws://.../ws/sensing. Defined as
 *            `presence: motion_score > 0.04` (main.rs:2681) where every input is
 *            a *change* feature, and BASELINE_EMA_ALPHA = 0.003 (main.rs:2712)
 *            folds a static body into the baseline in ~30 s.
 *   ORACLE   scripts/presence_service.py: deviation of the absolute
 *            per-subcarrier profile from a calibrated empty-room reference.
 *
 * The interesting statistic is agreement *while the subject is still*, which is
 * the regime where a motion-derived detector is structurally expected to fail.
 * Stillness is measured from the camera alone so it stays independent of both
 * CSI estimators.
 */

import { FilesetResolver, PoseLandmarker }
  from '../../vendor/mediapipe/vision_bundle.mjs';

// Derive both endpoints from the page origin so this page works unchanged on
// the sensing machine, over an SSH tunnel, or from another laptop on the LAN.
// Hardcoding 127.0.0.1 would resolve to the *viewing* laptop, not the rig.
const ORACLE_URL = `http://${location.hostname || '127.0.0.1'}:5008`;
const WS_URL = `ws://${location.host}/ws/sensing`;
const STRIP_SECONDS = 90;

// A landmark counts only if MediaPipe is reasonably sure it saw it.
const VIS_MIN = 0.5;
// Mean normalised landmark motion below this = "still".
const STILL_MOTION = 0.004;

const el = id => document.getElementById(id);

const state = {
  cam: { present: false, still: false, motion: 0, ready: false },
  ship: { present: null, motion_level: null, live: false },
  orac: { present: null, score: 0, status: null, live: false },
  mirror: false,
  history: [],
  stats: {
    all:   { ship: [0, 0], orac: [0, 0] },   // [agree, total]
    still: { ship: [0, 0], orac: [0, 0] },
    missStill: { ship: 0, orac: 0 },
  },
};

/* ------------------------------------------------------------------ camera */

let landmarker = null;
let lastLandmarks = null;
let lastVideoTime = -1;

async function initLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks('/ui/vendor/mediapipe/wasm');
  landmarker = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: '/ui/vendor/mediapipe/pose_landmarker_lite.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

async function listCameras() {
  const sel = el('cam-select');
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    const cams = devs.filter(d => d.kind === 'videoinput');
    if (!cams.length) return;
    const prev = sel.value;
    sel.innerHTML = cams.map((c, i) =>
      `<option value="${c.deviceId}">${c.label || `Camera ${i + 1}`}</option>`).join('');
    // Camo / Continuity expose the iPhone by name; prefer it automatically.
    const iphone = cams.findIndex(c => /camo|iphone|continuity/i.test(c.label));
    if (prev) sel.value = prev;
    else if (iphone >= 0) sel.selectedIndex = iphone;
  } catch { /* enumeration needs permission on some browsers */ }
}

async function startCamera() {
  const sel = el('cam-select');
  const deviceId = sel.value || undefined;
  const stream = await navigator.mediaDevices.getUserMedia({
    video: deviceId ? { deviceId: { exact: deviceId } }
                    : { width: 1280, height: 720 },
    audio: false,
  });
  const v = el('webcam');
  if (v.srcObject) v.srcObject.getTracks().forEach(t => t.stop());
  v.srcObject = stream;
  await v.play();
  el('cam-prompt').style.display = 'none';
  await listCameras();

  if (!landmarker) {
    setPill('pill-cam', 'LOADING MODEL', '');
    await initLandmarker();
  }
  setPill('pill-cam', 'CAMERA ON', 'ok');
  state.cam.ready = true;
  requestAnimationFrame(loop);
}

/** Mean normalised motion of visible landmarks between consecutive frames. */
function landmarkMotion(prev, cur) {
  if (!prev || !cur) return 1;
  let sum = 0, n = 0;
  for (let i = 0; i < cur.length && i < prev.length; i++) {
    if ((cur[i].visibility ?? 1) < VIS_MIN) continue;
    sum += Math.hypot(cur[i].x - prev[i].x, cur[i].y - prev[i].y);
    n++;
  }
  return n ? sum / n : 1;
}

const motionWin = [];

function loop() {
  const v = el('webcam');
  const c = el('overlay');
  if (v.videoWidth) {
    if (c.width !== v.videoWidth) { c.width = v.videoWidth; c.height = v.videoHeight; }
    if (v.currentTime !== lastVideoTime && landmarker) {
      lastVideoTime = v.currentTime;
      let lm = null;
      try {
        const res = landmarker.detectForVideo(v, performance.now());
        lm = res.landmarks && res.landmarks[0] ? res.landmarks[0] : null;
      } catch { /* transient decode failure */ }

      const vis = lm ? lm.filter(p => (p.visibility ?? 1) >= VIS_MIN).length : 0;
      state.cam.present = !!lm && vis >= 8;

      const m = state.cam.present ? landmarkMotion(lastLandmarks, lm) : 1;
      motionWin.push(m);
      if (motionWin.length > 45) motionWin.shift();
      state.cam.motion = motionWin.reduce((a, b) => a + b, 0) / motionWin.length;
      state.cam.still = state.cam.present && state.cam.motion < STILL_MOTION;

      lastLandmarks = lm;
      draw(c, lm);
    }
  }
  requestAnimationFrame(loop);
}

function draw(c, lm) {
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  if (!lm) return;
  ctx.save();
  if (state.mirror) { ctx.translate(c.width, 0); ctx.scale(-1, 1); }

  const P = i => ({ x: lm[i].x * c.width, y: lm[i].y * c.height });
  ctx.strokeStyle = state.cam.still ? '#ffb74d' : '#39d98a';
  ctx.lineWidth = Math.max(2, c.width / 320);
  ctx.lineCap = 'round';
  const conns = PoseLandmarker.POSE_CONNECTIONS || [];
  for (const k of conns) {
    const a = k.start ?? k[0], b = k.end ?? k[1];
    if (lm[a] === undefined || lm[b] === undefined) continue;
    if ((lm[a].visibility ?? 1) < VIS_MIN || (lm[b].visibility ?? 1) < VIS_MIN) continue;
    const p = P(a), q = P(b);
    ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
  }
  ctx.fillStyle = '#4dd0e1';
  for (let i = 0; i < lm.length; i++) {
    if ((lm[i].visibility ?? 1) < VIS_MIN) continue;
    const p = P(i);
    ctx.beginPath(); ctx.arc(p.x, p.y, ctx.lineWidth * 1.1, 0, 7); ctx.fill();
  }
  ctx.restore();
}

/* ------------------------------------------------------------ CSI sources */

function connectWS() {
  let ws;
  try { ws = new WebSocket(WS_URL); } catch { return; }
  ws.onopen = () => setPill('pill-ws', 'SERVER LIVE', 'ok');
  ws.onmessage = e => {
    try {
      const d = JSON.parse(e.data);
      const cls = d.classification || {};
      state.ship.present = !!cls.presence;
      state.ship.motion_level = cls.motion_level || null;
      state.ship.live = true;
    } catch { /* non-JSON frame */ }
  };
  ws.onclose = () => {
    setPill('pill-ws', 'SERVER OFF', 'bad');
    state.ship.live = false;
    setTimeout(connectWS, 3000);
  };
  ws.onerror = () => { try { ws.close(); } catch {} };
}

async function pollOracle() {
  try {
    const r = await fetch(`${ORACLE_URL}/presence`, {
      cache: 'no-store', signal: AbortSignal.timeout(1500),
    });
    const d = await r.json();
    state.orac = {
      present: !!d.present, score: d.score, status: d.status, live: !!d.live,
    };
    setPill('pill-oracle',
      d.live ? `ORACLE ${String(d.status).toUpperCase()}` : 'ORACLE NO CSI',
      d.live && d.status === 'ok' ? 'ok' : 'bad');
    el('orac-sub').textContent =
      `score ${(d.score * 100).toFixed(1)}% · thr ${(d.threshold * 100).toFixed(0)}%`;
  } catch {
    state.orac.live = false;
    setPill('pill-oracle', 'ORACLE OFF', 'bad');
  }
}

/* ------------------------------------------------------------- accounting */

function tick() {
  const cam = state.cam;
  if (cam.ready) {
    const rec = { t: Date.now(), cam: cam.present, still: cam.still,
                  ship: state.ship.live ? state.ship.present : null,
                  orac: state.orac.live ? state.orac.present : null };
    state.history.push(rec);
    const cutoff = Date.now() - STRIP_SECONDS * 1000;
    while (state.history.length && state.history[0].t < cutoff) state.history.shift();

    for (const key of ['ship', 'orac']) {
      if (rec[key] === null) continue;
      const agree = rec[key] === rec.cam;
      const b = state.stats.all[key];
      b[1]++; if (agree) b[0]++;
      if (rec.still) {
        const s = state.stats.still[key];
        s[1]++; if (agree) s[0]++;
        // The failure that matters: a still person the CSI calls absent.
        if (rec.cam && !rec[key]) state.stats.missStill[key]++;
      }
    }
  }
  render();
}

const pct = ([a, n]) => n < 5 ? '--' : `${((a / n) * 100).toFixed(0)}%`;

function render() {
  setVerdict('cam', state.cam.ready ? state.cam.present : null);
  setVerdict('ship', state.ship.live ? state.ship.present : null, state.ship.motion_level);
  setVerdict('orac', state.orac.live ? state.orac.present : null);

  el('cam-sub').textContent = state.cam.ready
    ? `ground truth · motion ${state.cam.motion.toFixed(4)}${state.cam.still ? ' · STILL' : ''}`
    : 'ground truth · pose landmarker';

  el('ag-ship').textContent = pct(state.stats.all.ship);
  el('ag-orac').textContent = pct(state.stats.all.orac);
  el('ag-ship-still').textContent = pct(state.stats.still.ship);
  el('ag-orac-still').textContent = pct(state.stats.still.orac);

  colour('ag-ship', state.stats.all.ship);
  colour('ag-orac', state.stats.all.orac);
  colour('ag-ship-still', state.stats.still.ship);
  colour('ag-orac-still', state.stats.still.orac);

  const ms = state.stats.missStill;
  const stillN = state.stats.still.ship[1];
  el('miss-note').textContent = stillN < 5 ? '' :
    `While still: shipped missed you ${ms.ship}× of ${stillN} samples, ` +
    `background reference ${ms.orac}×.`;

  // Disagreement outline, so the moment of divergence is obvious on screen.
  for (const [k, id] of [['ship', 'v-ship'], ['orac', 'v-orac']]) {
    const v = state[k];
    const dis = state.cam.ready && v.live && v.present !== state.cam.present;
    el(id).classList.toggle('verdict--dis', dis);
  }
  strip();
}

function colour(id, [a, n]) {
  const e = el(id);
  if (n < 5) { e.style.color = 'var(--dim)'; return; }
  const p = a / n;
  e.style.color = p > 0.9 ? 'var(--green)' : p > 0.7 ? 'var(--amber)' : 'var(--red)';
}

function setPill(id, text, kind) {
  const e = el(id);
  e.textContent = text;
  e.className = 'pill' + (kind ? ` pill--${kind}` : '');
}

function setVerdict(key, present, sub) {
  const s = el(`${key}-state`);
  if (present === null) { s.textContent = '--'; s.className = 'verdict-state s-na'; return; }
  s.textContent = present ? 'PRESENT' : 'ABSENT';
  s.className = 'verdict-state ' + (present ? 's-present' : 's-absent');
  if (key === 'ship' && sub) s.title = `motion_level: ${sub}`;
}

function strip() {
  const c = el('strip');
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  const n = state.history.length;
  if (!n) return;
  const h = 22, gap = 6;
  const w = c.width / Math.max(n, STRIP_SECONDS);

  ['cam', 'ship', 'orac'].forEach((key, r) => {
    const y = r * (h + gap);
    ctx.fillStyle = '#141c26';
    ctx.fillRect(0, y, c.width, h);
    for (let i = 0; i < n; i++) {
      const rec = state.history[i];
      const v = rec[key];
      if (v === null) continue;
      const disagree = key !== 'cam' && v !== rec.cam;
      ctx.fillStyle = disagree ? '#ff5c73' : v ? '#39d98a' : '#22303f';
      ctx.fillRect(i * w, y, Math.max(w, 1), h);
      if (rec.still) {
        ctx.fillStyle = 'rgba(255,183,77,.55)';
        ctx.fillRect(i * w, y + h - 3, Math.max(w, 1), 3);
      }
    }
  });
}

/* ------------------------------------------------------------------- wire */

el('btn-start').addEventListener('click', () => startCamera().catch(e => {
  setPill('pill-cam', 'CAMERA ERROR', 'bad');
  alert(`Could not start the camera:\n${e.message}`);
}));
el('btn-start2').addEventListener('click', () => el('btn-start').click());
el('btn-mirror').addEventListener('click', () => {
  state.mirror = !state.mirror;
  el('webcam').style.transform = state.mirror ? 'scaleX(-1)' : '';
});
el('btn-reset').addEventListener('click', () => {
  state.stats = {
    all: { ship: [0, 0], orac: [0, 0] },
    still: { ship: [0, 0], orac: [0, 0] },
    missStill: { ship: 0, orac: 0 },
  };
  state.history = [];
});
el('cam-select').addEventListener('change', () => {
  if (state.cam.ready) startCamera().catch(() => {});
});

listCameras();
connectWS();
pollOracle();
setInterval(pollOracle, 1000);
setInterval(tick, 1000);
