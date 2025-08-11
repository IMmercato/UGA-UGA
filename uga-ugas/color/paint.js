import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js";

// DOM
const video = document.getElementById("webcam");
const uiCanvas = document.getElementById("ui_canvas");
const paintCanvas = document.getElementById("paint_canvas");
const uiCtx = uiCanvas.getContext("2d");
const paintCtx = paintCanvas.getContext("2d");
const playBtn = document.getElementById("webcamButton");
const clearBtn = document.getElementById("clearButton");
const saveBtn = document.getElementById("saveButton");
const bgAudio = document.getElementById("ugasAudio");

let poseLandmarker;
let drawingUtils;
let webcamRunning = false;

// Full-screen sizing
function setCanvasSizeToWindow() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  uiCanvas.width = w;
  uiCanvas.height = h;
  paintCanvas.width = w;
  paintCanvas.height = h;
}
setCanvasSizeToWindow();
window.addEventListener("resize", setCanvasSizeToWindow);

// Painter state per person (max 2)
const MAX_PEOPLE = 2;
const ALPHA = 0.35;        // smoothing
const TOL = 16;            // shoulder tolerance (clear gesture)
const BELLY_TOL = 10;      // pixels above hips threshold
const TOL_SHOULDER = 10;   // left paint off threshold above shoulders
const CLEAR_HOLD_MS = 1000;

const painters = [
  makePainterState("#ff3b3b"),
  makePainterState("#00d2ff"),
];

function makePainterState(initialColor) {
  return {
    r: { x: null, y: null, prev: null, vx: 0, vy: 0, lastT: 0 },
    l: { x: null, y: null, prev: null, vx: 0, vy: 0, lastT: 0 },
    color: initialColor,
    brushSize: 12,
    hoverT: 0,
    hovering: false,
    bothUpSince: null,
  };
}

// Palette (top bar)
const PALETTE = { h: 80, margin: 0 };
const BRUSH = { min: 6, max: 26 };

// Init MediaPipe Pose
(async function init() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numPoses: MAX_PEOPLE
  });
  drawingUtils = new DrawingUtils(uiCtx);
})();

// Controls
playBtn.addEventListener("click", async () => {
  if (!poseLandmarker) return;

  webcamRunning = !webcamRunning;
  playBtn.innerHTML = webcamRunning
    ? '<i class="material-icons">pause</i>'
    : '<i class="material-icons">play_arrow</i>';

  if (webcamRunning) {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    video.srcObject = stream;
    await video.play();
    try { await bgAudio.play(); } catch (_) {}
    lastVideoTime = -1;
    predictWebcam();
  } else {
    stopStream();
    uiCtx.clearRect(0,0,uiCanvas.width, uiCanvas.height);
    try { bgAudio.pause(); } catch (_) {}
  }
});

clearBtn.addEventListener("click", clearPaint);
saveBtn.addEventListener("click", saveImage);

function stopStream() {
  const tracks = video.srcObject ? video.srcObject.getTracks() : [];
  tracks.forEach(t => t.stop());
  video.srcObject = null;
}

// Utils
function lerp(a, b, t) { return a + (b - a) * t; }

function hsvToRgb(h, s, v) {
  let f = (n, k = (n + h / 60) % 6) => v - v * s * Math.max(Math.min(k,4 - k,1),0);
  const r = Math.round(f(5) * 255);
  const g = Math.round(f(3) * 255);
  const b = Math.round(f(1) * 255);
  return `rgb(${r},${g},${b})`;
}

function hueFromX(x) {
  const w = uiCanvas.width;
  return (x / Math.max(w, 1)) * 360;
}

// Landmarks indices
const IDX = {
  lShoulder: 11, rShoulder: 12,
  lElbow: 13, rElbow: 14,
  lWrist: 15, rWrist: 16,
  lHip: 23, rHip: 24
};

let lastVideoTime = -1;

async function predictWebcam() {
  if (!poseLandmarker || !webcamRunning) return;

  const nowMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;

    poseLandmarker.detectForVideo(video, nowMs, (result) => {
      renderFrame(result);
    });
  }
  requestAnimationFrame(predictWebcam);
}

function renderFrame(result) {
  const w = uiCanvas.width;
  const h = uiCanvas.height;

  // UI layer fresh each frame
  uiCtx.clearRect(0,0,w,h);

  // Rainbow palette bar
  drawRainbow(uiCtx, w, PALETTE.h);

  const people = result.landmarks || [];
  const count = Math.min(people.length, MAX_PEOPLE);
  const now = performance.now();

  // Reset clear timers for missing painters
  for (let i = 0; i < MAX_PEOPLE; i++) if (i >= count) painters[i].bothUpSince = null;

  for (let i = 0; i < count; i++) {
    const lm = people[i];
    const p = painters[i];

    // Extract
    const rs = lm[IDX.rShoulder], re = lm[IDX.rElbow], rw = lm[IDX.rWrist];
    const ls = lm[IDX.lShoulder], le = lm[IDX.lElbow], lw = lm[IDX.lWrist];
    const lh = lm[IDX.lHip], rh = lm[IDX.rHip];

    // Visibility checks
    const rOK = visOK(rw, re, rs);
    const lOK = visOK(lw, le, ls);

    // To pixels
    const rPx = toPx(rw, w, h);
    const lPx = toPx(lw, w, h);
    const rShoulderPx = toPx(rs, w, h);
    const lShoulderPx = toPx(ls, w, h);

    // Belly (hips) threshold in pixels: use min visible hip Y; fallback to lower screen
    let hipYs = [];
    if ((lh?.visibility ?? 1) > 0.5) hipYs.push(toPx(lh, w, h).y);
    if ((rh?.visibility ?? 1) > 0.5) hipYs.push(toPx(rh, w, h).y);
    const bellyY = hipYs.length ? Math.min(...hipYs) : Math.round(h * 0.65);

    // Smooth and keep previous for stroke
    smoothPoint(p.r, rPx);
    smoothPoint(p.l, lPx);

    // Palette hover independent of shoulder rule
    const inPalette = lOK && p.l.y !== null && (p.l.y < PALETTE.h + PALETTE.margin);
    handleColorHover(p, inPalette, now);

    // Dynamic brush size from right-hand speed
    updateBrushFromSpeed(p);

    // Gates
    const rAboveBelly = rOK && p.r.y !== null && (p.r.y < bellyY - BELLY_TOL);
    const lAboveBelly = lOK && p.l.y !== null && (p.l.y < bellyY - BELLY_TOL);

    const shouldersMinY = Math.min(rShoulderPx.y, lShoulderPx.y);
    const leftAboveShoulders = lOK && p.l.y !== null && (p.l.y < shouldersMinY - TOL_SHOULDER);

    // Paint with RIGHT wrist if above belly
    if (rAboveBelly && p.r.prev) {
      strokeLine(paintCtx, p.r.prev.x, p.r.prev.y, p.r.x, p.r.y, p.color, p.brushSize);
    }

    // Paint with LEFT wrist if above belly, not in palette, and not above shoulders
    if (lAboveBelly && !inPalette && !leftAboveShoulders && p.l.prev) {
      strokeLine(paintCtx, p.l.prev.x, p.l.prev.y, p.l.x, p.l.y, p.color, Math.max(8, p.brushSize - 2));
    }

    // Both hands up -> clear gesture (unchanged)
    const rightUp = rOK && p.r.y !== null && (p.r.y < rShoulderPx.y - TOL) &&
                    (re && toPx(re, w, h).y < rShoulderPx.y - TOL);
    const leftUp  = lOK && p.l.y !== null && (p.l.y < lShoulderPx.y - TOL) &&
                    (le && toPx(le, w, h).y < lShoulderPx.y - TOL);

    if (rightUp && leftUp) {
      p.bothUpSince = p.bothUpSince ?? now;
      drawClearArc(uiCtx, w, h, Math.min((now - p.bothUpSince) / CLEAR_HOLD_MS, 1));
      if (now - p.bothUpSince > CLEAR_HOLD_MS) {
        clearPaint();
        p.bothUpSince = null;
      }
    } else {
      p.bothUpSince = null;
    }

    // HUD: brush ring at right wrist + color chips
    drawHUDForPainter(uiCtx, p, i);
    // Optional: draw belly line guide for debugging
    // drawBellyLine(uiCtx, bellyY, w);
  }

  // Minimal pose visuals for debbug
  /*for (const landmark of result.landmarks) {
    drawingUtils.drawLandmarks(landmark, {
      radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1),
      color: "rgba(255,255,255,0.55)"
    });
    drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, { color: "rgba(255,255,255,0.18)" });
  }*/
}

// Helpers
function visOK(...marks) {
  return marks.every(m => (m?.visibility ?? 1) > 0.5);
}

function toPx(lm, w, h) {
  return { x: lm.x * w, y: lm.y * h };
}

function smoothPoint(state, target) {
  const now = performance.now();
  if (state.x == null || state.y == null) {
    state.prev = { x: target.x, y: target.y };
    state.x = target.x; state.y = target.y;
    state.vx = 0; state.vy = 0; state.lastT = now;
    return;
  }
  const px = state.x, py = state.y;
  const nx = lerp(state.x, target.x, ALPHA);
  const ny = lerp(state.y, target.y, ALPHA);
  const dt = Math.max((now - state.lastT) / 1000, 1e-3);
  state.vx = (nx - px) / dt;
  state.vy = (ny - py) / dt;
  state.x = nx; state.y = ny;
  state.prev = { x: px, y: py };
  state.lastT = now;
}

function updateBrushFromSpeed(p) {
  const speed = Math.hypot(p.r.vx, p.r.vy); // px/s
  const s = Math.min(speed, 2500) / 2500;
  p.brushSize = Math.round(lerp(BRUSH.min, BRUSH.max, Math.pow(s, 0.6)));
}

function strokeLine(ctx, x0, y0, x1, y1, color, size) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = color;
  const alpha = 0.9 - (Math.max(size - BRUSH.min, 0) / (BRUSH.max - BRUSH.min)) * 0.25;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = size;

  const dx = x1 - x0, dy = y1 - y0;
  const dist = Math.hypot(dx, dy);
  const steps = Math.max(1, Math.floor(dist / 8));

  ctx.beginPath();
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const x = x0 + dx * t;
    const y = y0 + dy * t;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawRainbow(ctx, w, h) {
  const grad = ctx.createLinearGradient(0, 0, w, 0);
  for (let i = 0; i <= 12; i++) {
    const t = i / 12;
    const hue = t * 360;
    grad.addColorStop(t, hsvToRgb(hue, 1, 1));
  }
  ctx.save();
  ctx.globalAlpha = 0.9;
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  const glint = ctx.createLinearGradient(0, 0, 0, h);
  glint.addColorStop(0, "rgba(255,255,255,0.22)");
  glint.addColorStop(1, "rgba(255,255,255,0.02)");
  ctx.fillStyle = glint;
  ctx.fillRect(0, 0, w, h);

  ctx.fillStyle = "rgba(0,0,0,0.25)";
  ctx.fillRect(0, h - 2, w, 2);

  ctx.font = "14px system-ui, sans-serif";
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.fillText("Hover LEFT hand here to pick color", 12, 22);
  ctx.restore();
}

function handleColorHover(p, inPalette, now) {
  if (inPalette && p.l.x != null) {
    const hue = hueFromX(p.l.x);
    const color = hsvToRgb(hue, 1, 1);

    if (!p.hovering) {
      p.hovering = true;
      p.hoverT = now;
    }
    const dwell = now - p.hoverT;

    drawColorPreview(uiCtx, p.l.x, p.l.y, color, Math.min(dwell / 400, 1));

    if (dwell > 350) {
      p.color = color;
    }
  } else {
    p.hovering = false;
  }
}

function drawColorPreview(ctx, x, y, color, t) {
  ctx.save();
  ctx.globalAlpha = 0.95;
  ctx.beginPath();
  ctx.arc(x, y, 20, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.fillStyle = color;
  ctx.arc(x, y, 16, -Math.PI / 2, -Math.PI / 2 + t * Math.PI * 2);
  ctx.closePath();
  ctx.fill();

  ctx.beginPath();
  ctx.arc(x, y, 6, 0, Math.PI * 2);
  ctx.fillStyle = "white";
  ctx.fill();
  ctx.restore();
}

function drawClearArc(ctx, w, h, t) {
  ctx.save();
  const R = 34;
  const cx = w - 40;
  const cy = 40;
  ctx.lineWidth = 6;
  ctx.strokeStyle = `rgba(255,255,255,${0.6 + 0.2 * Math.sin(performance.now()/200)})`;
  ctx.beginPath();
  ctx.arc(cx, cy, R, -Math.PI/2, -Math.PI/2 + t * Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawHUDForPainter(ctx, p, idx) {
  if (p.r.x == null) return;

  ctx.save();
  const x = p.r.x, y = p.r.y;
  ctx.globalAlpha = 0.95;

  // Brush ring
  ctx.beginPath();
  ctx.arc(x, y, p.brushSize/2 + 6, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(0,0,0,0.35)";
  ctx.lineWidth = 6;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(x, y, p.brushSize/2, 0, Math.PI * 2);
  ctx.strokeStyle = p.color;
  ctx.lineWidth = 3;
  ctx.stroke();

  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.fillText(`P${idx+1}`, x + 10, y - 10);
  ctx.restore();

  drawColorChips(ctx, idx, p.color);
}

function drawColorChips(ctx, idx, color) {
  const y = uiCanvas.height - 18;
  const x = 18 + idx * 28;
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, 10, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0,0,0,0.35)";
  ctx.fill();
  ctx.beginPath();
  ctx.arc(x, y, 10, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.95;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(255,255,255,0.75)";
  ctx.stroke();
  ctx.restore();
}

function clearPaint() {
  paintCtx.clearRect(0,0,paintCanvas.width, paintCanvas.height);
}

function saveImage() {
  const w = uiCanvas.width, h = uiCanvas.height;
  const tmp = document.createElement("canvas");
  tmp.width = w; tmp.height = h;
  const tctx = tmp.getContext("2d");

  const bg = tctx.createLinearGradient(0,0,w,h);
  bg.addColorStop(0, "#0b0b13");
  bg.addColorStop(1, "#121225");
  tctx.fillStyle = bg;
  tctx.fillRect(0,0,w,h);

  try {
    tctx.globalAlpha = 0.16;
    tctx.drawImage(video, 0, 0, w, h);
  } catch (_) {}

  tctx.globalAlpha = 1.0;
  tctx.drawImage(paintCanvas, 0, 0);

  const url = tmp.toDataURL("image/png");
  const a = document.createElement("a");
  a.href = url;
  a.download = `hand-paint-${Date.now()}.png`;
  a.click();
}

// Optional debugging guide for belly line
function drawBellyLine(ctx, bellyY, w) {
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.25)";
  ctx.setLineDash([6,6]);
  ctx.beginPath();
  ctx.moveTo(0, bellyY - BELLY_TOL);
  ctx.lineTo(w, bellyY - BELLY_TOL);
  ctx.stroke();
  ctx.restore();
}