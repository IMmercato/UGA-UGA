import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js";

let poseLandmarker;
let webcamRunning = false;

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);
const enableWebcamButton = document.getElementById("webcamButton");
const bgAudio = document.getElementById("ugasAudio");

const videoWidth = 640;
const videoHeight = 480;
canvasElement.width = videoWidth;
canvasElement.height = videoHeight;
video.width = videoWidth;
video.height = videoHeight;

// Tracking + smoothing (for right wrist trail)
let prevWristPx = null;
let prevTimeMs = 0;
let wristTrail = [];
let smoothedWristPx = null;
const ALPHA = 0.5;
const TOL = 20;

const createPoseLandmarker = async () => {
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
    numPoses: 2
  });
};
createPoseLandmarker();

enableWebcamButton.addEventListener("click", () => {
  if (!poseLandmarker) return;

  webcamRunning = !webcamRunning;
  enableWebcamButton.innerText = webcamRunning ? "||" : "ENABLE PREDICTIONS";

  if (webcamRunning) {
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      video.srcObject = stream;
      video.onloadeddata = () => predictWebcam();
    });
    bgAudio.play(); // start music
  } else {
    const tracks = video.srcObject ? video.srcObject.getTracks() : [];
    tracks.forEach((t) => t.stop());
    video.srcObject = null;
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    bgAudio.pause(); // stop music
  }
});

let lastVideoTime = -1;
async function predictWebcam() {
  if (!poseLandmarker || !webcamRunning) return;

  const nowMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;

    poseLandmarker.detectForVideo(video, nowMs, (result) => {
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

      for (const landmark of result.landmarks) {
        drawingUtils.drawLandmarks(landmark, {
          radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
        });
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
      }

      if (result.landmarks.length) {
        const lm = result.landmarks[0];
        const [rShoulder, rElbow, rWrist] = [lm[12], lm[14], lm[16]];
        const [lShoulder, lElbow, lWrist] = [lm[11], lm[13], lm[15]];

        const w = canvasElement.width;
        const h = canvasElement.height;

        const rWristPx = { x: rWrist.x * w, y: rWrist.y * h };
        const rElbowPx = { x: rElbow.x * w, y: rElbow.y * h };
        const rShoulderPx = { x: rShoulder.x * w, y: rShoulder.y * h };

        const lWristPx = { x: lWrist.x * w, y: lWrist.y * h };
        const lElbowPx = { x: lElbow.x * w, y: lElbow.y * h };
        const lShoulderPx = { x: lShoulder.x * w, y: lShoulder.y * h };

        // Smoothing for right wrist (used for speed/trail)
        if (!smoothedWristPx) smoothedWristPx = { ...rWristPx };
        else {
          smoothedWristPx.x = ALPHA * rWristPx.x + (1 - ALPHA) * smoothedWristPx.x;
          smoothedWristPx.y = ALPHA * rWristPx.y + (1 - ALPHA) * smoothedWristPx.y;
        }

        const rVisOK = (rWrist.visibility ?? 1) > 0.5 &&
                       (rElbow.visibility ?? 1) > 0.5 &&
                       (rShoulder.visibility ?? 1) > 0.5;

        const lVisOK = (lWrist.visibility ?? 1) > 0.5 &&
                       (lElbow.visibility ?? 1) > 0.5 &&
                       (lShoulder.visibility ?? 1) > 0.5;

        const isRightHandUp = rVisOK &&
                              rWristPx.y < rShoulderPx.y - TOL &&
                              rElbowPx.y < rShoulderPx.y - TOL;

        const isLeftHandUp = lVisOK &&
                             lWristPx.y < lShoulderPx.y - TOL &&
                             lElbowPx.y < lShoulderPx.y - TOL;

        if (isRightHandUp) {
          canvasCtx.fillStyle = "rgba(255, 220, 0, 0.95)";
          canvasCtx.font = "20px system-ui, sans-serif";
          canvasCtx.fillText("Right hand UP", w - 190, 30);
        }

        if (isLeftHandUp) {
          canvasCtx.fillStyle = "rgba(120, 180, 255, 0.95)";
          canvasCtx.font = "20px system-ui, sans-serif";
          canvasCtx.fillText("Left hand UP", 10, 50);
        }

        // Speed + trail (right wrist)
        const now = performance.now();
        if (prevWristPx) {
          const dx = smoothedWristPx.x - prevWristPx.x;
          const dy = smoothedWristPx.y - prevWristPx.y;
          const dt = Math.max((now - prevTimeMs) / 1000, 1e-3);
          const speed = Math.hypot(dx, dy) / dt;

          canvasCtx.fillStyle = "#fff";
          canvasCtx.font = "14px system-ui, sans-serif";
          canvasCtx.fillText(`Right hand speed: ${speed.toFixed(0)} px/s`, 10, 22);
        }

        prevWristPx = { ...smoothedWristPx };
        prevTimeMs = now;

        wristTrail.push({ x: smoothedWristPx.x, y: smoothedWristPx.y, t: now });
        const TRAIL_MS = 1500;
        while (wristTrail.length && now - wristTrail[0].t > TRAIL_MS) wristTrail.shift();

        if (wristTrail.length > 1) {
          canvasCtx.strokeStyle = "rgba(0, 255, 120, 0.9)";
          canvasCtx.lineWidth = 2;
          canvasCtx.beginPath();
          canvasCtx.moveTo(wristTrail[0].x, wristTrail[0].y);
          for (let i = 1; i < wristTrail.length; i++) {
            canvasCtx.lineTo(wristTrail[i].x, wristTrail[i].y);
          }
          canvasCtx.stroke();
        }

        canvasCtx.fillStyle = "rgba(0, 255, 120, 0.9)";
        canvasCtx.beginPath();
        canvasCtx.arc(smoothedWristPx.x, smoothedWristPx.y, 5, 0, Math.PI * 2);
        canvasCtx.fill();
      }

      canvasCtx.restore();
    });
  }

  // FIX: call the correct function name
  window.requestAnimationFrame(predictWebcam);
}