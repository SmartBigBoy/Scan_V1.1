/* ============================
   iOS Notes Scanner - App Logic
   ============================ */

// ===== State =====
const state = {
    mode: 'scanner',           // 'scanner' | 'adjust' | 'preview'
    pages: [],                 // scanned pages [{dataUrl, filter}]
    activePage: -1,            // index of active page in preview
    stream: null,
    facingMode: 'environment',
    flash: false,
    detectionActive: false,
    detectedCorners: null,     // normalized [0..1] corners for overlay
    lastCorners: null,         // smoothed corners
    capturePending: false,
    animationId: null,
    capturedImageData: null,     // original full-res imageData for adjust screen
    adjustCorners: null,         // detected corners in original image coordinates
    isMobile: /iPhone|iPad|iPod|Android/i.test(navigator.userAgent),
    autoCapture: true,           // auto-capture when document is stable
};

// ===== DOM References =====
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const video = $('#scanner-video');
const overlayCanvas = $('#scanner-overlay');
const overlayCtx = overlayCanvas.getContext('2d');
const detectionFrame = $('#detection-frame');
const scanLines = $('#scan-lines');
const cameraPrompt = $('#camera-prompt');

// ===== Toast =====
let toastEl = null;
function showToast(msg, duration = 2000) {
    if (!toastEl) {
        toastEl = document.createElement('div');
        toastEl.className = 'toast';
        document.body.appendChild(toastEl);
    }
    toastEl.textContent = msg;
    toastEl.classList.add('visible');
    clearTimeout(toastEl._timer);
    toastEl._timer = setTimeout(() => toastEl.classList.remove('visible'), duration);
}

// ===== Screens =====
function showScreen(id) {
    $$('.screen').forEach(s => s.classList.remove('active'));
    $(id).classList.add('active');
    state.mode = id.replace('#screen-', '');
}

// ============================
//  Camera
// ============================

async function initCamera() {
    try {
        if (state.stream) stopCamera();

        const constraints = {
            video: {
                facingMode: state.facingMode,
                width: { ideal: 1920 },
                height: { ideal: 1080 },
            },
            audio: false,
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        state.stream = stream;
        video.srcObject = stream;

        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play();
                resolve(true);
            };
        });
    } catch (err) {
        console.error('Camera error:', err);
        return false;
    }
}

function stopCamera() {
    if (state.stream) {
        state.stream.getTracks().forEach(t => t.stop());
        state.stream = null;
    }
    video.srcObject = null;
}

function toggleFlash() {
    if (!state.stream) return;
    const track = state.stream.getVideoTracks()[0];
    if (!track) return;

    const capabilities = track.getCapabilities();
    if (!capabilities.torch) {
        showToast('此设备不支持闪光灯');
        return;
    }

    state.flash = !state.flash;
    track.applyConstraints({ advanced: [{ torch: state.flash }] }).catch(() => {});
    $('#btn-flash').style.opacity = state.flash ? '1' : '0.6';
}

// ============================
//  Image Processing Helpers
// ============================

function grayscale(data, w, h) {
    const gray = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
        const idx = i * 4;
        gray[i] = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
    }
    return gray;
}

function sobel(gray, w, h) {
    const mag = new Float32Array(w * h);
    // Sobel kernels
    const gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    const gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            let sx = 0, sy = 0;
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const p = (y + ky) * w + (x + kx);
                    const k = (ky + 1) * 3 + (kx + 1);
                    sx += gray[p] * gx[k];
                    sy += gray[p] * gy[k];
                }
            }
            mag[y * w + x] = Math.sqrt(sx * sx + sy * sy);
        }
    }
    return mag;
}

function boxBlur(gray, w, h, radius) {
    const result = new Float32Array(w * h);
    const size = 2 * radius + 1;
    const norm = 1 / (size * size);
    for (let y = radius; y < h - radius; y++) {
        for (let x = radius; x < w - radius; x++) {
            let sum = 0;
            for (let ky = -radius; ky <= radius; ky++) {
                for (let kx = -radius; kx <= radius; kx++) {
                    sum += gray[(y + ky) * w + (x + kx)];
                }
            }
            result[y * w + x] = sum * norm;
        }
    }
    // Copy edge pixels
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            if (y < radius || y >= h - radius || x < radius || x >= w - radius) {
                result[y * w + x] = gray[y * w + x];
            }
        }
    }
    return result;
}

function percentileThreshold(data, pct = 0.85) {
    const sorted = new Float32Array(data);
    sorted.sort();
    return sorted[Math.floor(sorted.length * pct)];
}

function detectBoundaries(edges, gray, w, h, threshold) {
    const topPts = [], bottomPts = [], leftPts = [], rightPts = [];
    const minRun = 2;

    // Top scan: find first run of consecutive edge pixels
    for (let x = 2; x < w - 2; x++) {
        let found = false;
        let run = 0;
        for (let y = 2; y < h - 2; y++) {
            if (edges[y * w + x] > threshold) {
                run++;
                if (run >= minRun) {
                    topPts.push({ x, y: y - minRun + 1 });
                    found = true;
                    break;
                }
            } else {
                run = 0;
            }
        }
        // Fallback: brightness transition (dark→bright = paper edge)
        if (!found) {
            for (let y = 2; y < h - 2; y++) {
                const diff = gray[y * w + x] - gray[(y - 1) * w + x];
                if (diff > 25) {
                    topPts.push({ x, y });
                    break;
                }
            }
        }
    }

    // Bottom scan
    for (let x = 2; x < w - 2; x++) {
        let found = false;
        let run = 0;
        for (let y = h - 3; y > 1; y--) {
            if (edges[y * w + x] > threshold) {
                run++;
                if (run >= minRun) {
                    bottomPts.push({ x, y: y + minRun - 1 });
                    found = true;
                    break;
                }
            } else {
                run = 0;
            }
        }
        if (!found) {
            for (let y = h - 3; y > 1; y--) {
                const diff = gray[(y + 1) * w + x] - gray[y * w + x];
                if (diff > 25) {
                    bottomPts.push({ x, y });
                    break;
                }
            }
        }
    }

    // Left scan
    for (let y = 2; y < h - 2; y++) {
        let found = false;
        let run = 0;
        for (let x = 2; x < w - 2; x++) {
            if (edges[y * w + x] > threshold) {
                run++;
                if (run >= minRun) {
                    leftPts.push({ x: x - minRun + 1, y });
                    found = true;
                    break;
                }
            } else {
                run = 0;
            }
        }
        if (!found) {
            for (let x = 2; x < w - 2; x++) {
                const diff = gray[y * w + x] - gray[y * w + (x - 1)];
                if (diff > 25) {
                    leftPts.push({ x, y });
                    break;
                }
            }
        }
    }

    // Right scan
    for (let y = 2; y < h - 2; y++) {
        let found = false;
        let run = 0;
        for (let x = w - 3; x > 1; x--) {
            if (edges[y * w + x] > threshold) {
                run++;
                if (run >= minRun) {
                    rightPts.push({ x: x + minRun - 1, y });
                    found = true;
                    break;
                }
            } else {
                run = 0;
            }
        }
        if (!found) {
            for (let x = w - 3; x > 1; x--) {
                const diff = gray[y * w + (x + 1)] - gray[y * w + x];
                if (diff > 25) {
                    rightPts.push({ x, y });
                    break;
                }
            }
        }
    }

    return { topPts, bottomPts, leftPts, rightPts };
}

function filterOutliers(pts, threshold = 2.5) {
    if (pts.length < 3) return pts;

    let sum = 0;
    for (const p of pts) sum += p.y;
    const mean = sum / pts.length;

    let sqSum = 0;
    for (const p of pts) sqSum += (p.y - mean) ** 2;
    const std = Math.sqrt(sqSum / pts.length);
    if (std < 1) return pts;

    const filtered = [];
    for (const p of pts) {
        if (Math.abs(p.y - mean) < std * threshold) {
            filtered.push(p);
        }
    }
    return filtered;
}

function fitLine(pts) {
    // Linear regression: y = a * x + b
    if (pts.length < 2) return null;

    const n = pts.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (const p of pts) {
        sumX += p.x;
        sumY += p.y;
        sumXY += p.x * p.y;
        sumX2 += p.x * p.x;
    }

    const denom = n * sumX2 - sumX * sumX;
    if (Math.abs(denom) < 1e-10) return null;

    const a = (n * sumXY - sumX * sumY) / denom;
    const b = (sumY - a * sumX) / n;

    return { a, b };
}

function fitLineRobust(pts) {
    // Use median-based approach (Theil-Sen) for robustness
    if (pts.length < 4) return fitLine(pts);

    // Compute all pairwise slopes
    const slopes = [];
    for (let i = 0; i < Math.min(pts.length, 80); i++) {
        for (let j = i + 1; j < Math.min(pts.length, 80); j++) {
            const dx = pts[j].x - pts[i].x;
            if (Math.abs(dx) > 2) {
                slopes.push((pts[j].y - pts[i].y) / dx);
            }
        }
    }

    if (slopes.length === 0) return fitLine(pts);

    slopes.sort((a, b) => a - b);
    const medianSlope = slopes[Math.floor(slopes.length / 2)];

    // Compute median intercept
    const intercepts = pts.map(p => p.y - medianSlope * p.x);
    intercepts.sort((a, b) => a - b);
    const medianIntercept = intercepts[Math.floor(intercepts.length / 2)];

    return { a: medianSlope, b: medianIntercept };
}

function fitVerticalLine(pts) {
    // For left/right boundaries, swap x and y: x = a * y + b
    if (pts.length < 2) return null;

    const swapped = pts.map(p => ({ x: p.y, y: p.x }));
    const line = fitLine(swapped);
    if (!line) return null;

    // Now a and b are for x = a*y + b
    // Convert back: we need y expressed in terms of x
    // x = a*y + b => y = (x - b) / a
    // But we'll keep it as x = a*y + b
    return { a: line.a, b: line.b, vertical: true };
}

function lineIntersection(l1, l2) {
    if (!l1 || !l2) return null;

    if (l1.vertical && l2.vertical) return null;

    if (l1.vertical) {
        // l1: x = a1*y + b1
        // l2: y = a2*x + b2  =>  y = a2*(a1*y + b1) + b2  =>  y*(1 - a2*a1) = a2*b1 + b2
        const { a: a1, b: b1 } = l1;
        const { a: a2, b: b2 } = l2;
        // x = a1*y + b1, y = a2*x + b2
        // x = a1*(a2*x + b2) + b1 => x*(1 - a1*a2) = a1*b2 + b1
        const denom = 1 - a1 * a2;
        if (Math.abs(denom) < 1e-10) return null;
        const x = (a1 * b2 + b1) / denom;
        const y = a2 * x + b2;
        return { x, y };
    }

    if (l2.vertical) {
        return lineIntersection(l2, l1);
    }

    // Both are horizontal: y = a*x + b
    const { a: a1, b: b1 } = l1;
    const { a: a2, b: b2 } = l2;

    const denom = a1 - a2;
    if (Math.abs(denom) < 1e-10) return null;

    const x = (b2 - b1) / denom;
    const y = a1 * x + b1;
    return { x, y };
}

// ============================
//  Document Detection
// ============================

function quadQualityScore(corners, normW, normH) {
    if (!corners || corners.length < 4) return 0;
    const [tl, tr, br, bl] = corners;
    const w = normW || 1, h = normH || 1;

    // Area score — bigger document = better
    const area = Math.abs(
        (tl.x * tr.y - tr.x * tl.y) +
        (tr.x * br.y - br.x * tr.y) +
        (br.x * bl.y - bl.x * br.y) +
        (bl.x * tl.y - tl.x * bl.y)
    ) / 2;
    const areaScore = Math.min(1, area / (w * h * 0.35));

    // Angle score — closer to 90° = better
    function angle(p1, p2, p3) {
        const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
        const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };
        const dot = v1.x * v2.x + v1.y * v2.y;
        const mag = Math.sqrt(v1.x * v1.x + v1.y * v1.y) *
                    Math.sqrt(v2.x * v2.x + v2.y * v2.y);
        if (mag < 1e-10) return 90;
        return Math.acos(Math.max(-1, Math.min(1, dot / mag)));
    }
    const angles = [angle(bl, tl, tr), angle(tl, tr, br),
                    angle(tr, br, bl), angle(br, bl, tl)];
    let angleScore = 0;
    for (const a of angles) {
        const deg = a * 180 / Math.PI;
        angleScore += Math.max(0, 1 - Math.abs(deg - 90) / 90);
    }
    angleScore /= 4;

    // Aspect ratio score — not too extreme
    const avgW = (Math.hypot(tr.x - tl.x, tr.y - tl.y) +
                  Math.hypot(br.x - bl.x, br.y - bl.y)) / 2;
    const avgH = (Math.hypot(bl.x - tl.x, bl.y - tl.y) +
                  Math.hypot(br.x - tr.x, br.y - tr.y)) / 2;
    const aspect = Math.max(avgW, avgH) / Math.max(1, Math.min(avgW, avgH));
    const aspectScore = Math.max(0, 1 - (aspect - 1) / 6);

    return areaScore * 0.45 + angleScore * 0.35 + aspectScore * 0.20;
}

function detectDocumentCorners(videoElem) {
    const vw = videoElem.videoWidth;
    const vh = videoElem.videoHeight;
    if (!vw || !vh) return null;

    // Downscale for performance
    const maxDim = 240;
    const scale = Math.min(1, maxDim / Math.max(vw, vh));
    const w = Math.round(vw * scale);
    const h = Math.round(vh * scale);

    overlayCanvas.width = w;
    overlayCanvas.height = h;
    overlayCtx.drawImage(videoElem, 0, 0, w, h);
    const imageData = overlayCtx.getImageData(0, 0, w, h);
    const gray = grayscale(imageData.data, w, h);

    // 5×5 box blur to suppress text/background noise
    const blurred = boxBlur(gray, w, h, 2);

    // Sobel edge detection
    const edges = sobel(blurred, w, h);

    // Adaptive threshold using 85th percentile
    const thresh = percentileThreshold(edges, 0.85);

    // Boundary detection with consecutive-run filter + brightness transitions
    const { topPts, bottomPts, leftPts, rightPts } =
        detectBoundaries(edges, gray, w, h, thresh);

    // Relaxed minimum points
    const minPts = 4;
    if (topPts.length < minPts || bottomPts.length < minPts ||
        leftPts.length < minPts || rightPts.length < minPts) {
        return null;
    }

    // Filter outliers (generous threshold keeps most points)
    const topF = filterOutliers(topPts, 3.0);
    const bottomF = filterOutliers(bottomPts, 3.0);
    const leftF = filterOutliers(leftPts, 3.0);
    const rightF = filterOutliers(rightPts, 3.0);

    if (topF.length < 3 || bottomF.length < 3 ||
        leftF.length < 3 || rightF.length < 3) {
        return null;
    }

    // Fit robust lines
    const topLine = fitLineRobust(topF);
    const bottomLine = fitLineRobust(bottomF);
    const leftLine = fitVerticalLine(leftF);
    const rightLine = fitVerticalLine(rightF);

    if (!topLine || !bottomLine || !leftLine || !rightLine) return null;

    // Basic orientation checks
    if (topLine.a * (w / 2) + topLine.b >= bottomLine.a * (w / 2) + bottomLine.b) return null;
    if (leftLine.a * (h / 2) + leftLine.b >= rightLine.a * (h / 2) + rightLine.b) return null;

    // Compute 4 corners
    const tl = lineIntersection(topLine, leftLine);
    const tr = lineIntersection(topLine, rightLine);
    const bl = lineIntersection(bottomLine, leftLine);
    const br = lineIntersection(bottomLine, rightLine);

    if (!tl || !tr || !bl || !br) return null;

    // Relaxed bounds check
    const margin = -15;
    if (tl.x < margin || tl.y < margin ||
        tr.x > w + 15 || tr.y < margin ||
        bl.x < margin || bl.y > h + 15 ||
        br.x > w + 15 || br.y > h + 15) {
        return null;
    }

    // Minimum area check (5% of frame)
    const area = Math.abs(
        (tl.x * tr.y - tr.x * tl.y) + (tr.x * br.y - br.x * tr.y) +
        (br.x * bl.y - bl.x * br.y) + (bl.x * tl.y - tl.x * bl.y)) / 2;
    if (area < (w * h) * 0.05) return null;

    // Scale back to original video size
    const invScale = 1 / scale;
    const result = [
        { x: tl.x * invScale, y: tl.y * invScale },
        { x: tr.x * invScale, y: tr.y * invScale },
        { x: br.x * invScale, y: br.y * invScale },
        { x: bl.x * invScale, y: bl.y * invScale },
    ];

    // Attach quality score (0-1)
    result.quality = quadQualityScore(result, vw, vh);

    return result;
}

function smoothCorners(newCorners, oldCorners, factor = 0.35) {
    if (!oldCorners || !newCorners) return newCorners;
    return newCorners.map((c, i) => ({
        x: c.x * factor + oldCorners[i].x * (1 - factor),
        y: c.y * factor + oldCorners[i].y * (1 - factor),
    }));
}

// ============================
//  Detection Loop — State Machine
// ============================
//  States:  SEARCHING → TRACKING → STABLE → (auto-capture)
//           ↑_____________________________|  (if doc lost, back to SEARCHING)

const DETECT_STATE = Object.freeze({
    SEARCHING: 'searching',
    TRACKING:  'tracking',
    STABLE:    'stable',
});

let detectState = DETECT_STATE.SEARCHING;
let cornerHistory = [];    // [{corners, quality}, ...] sliding window
let stableFrames = 0;
let noDocCount = 0;
let lastDetectTime = 0;
const DETECT_INTERVAL = 100;    // ms (≈10 fps)
const MAX_HISTORY = 12;          // frames of corner history
const STABLE_REQUIRED = 8;       // consecutive stable frames before capture
const NO_DOC_LIMIT = 8;          // frames without doc before resetting

function startDetectionLoop() {
    if (state.animationId) return;
    state.detectionActive = true;
    detectState = DETECT_STATE.SEARCHING;
    cornerHistory = [];
    stableFrames = 0;
    noDocCount = 0;

    function detect() {
        if (!state.detectionActive || video.readyState < 2) {
            state.animationId = requestAnimationFrame(detect);
            return;
        }

        // Throttle to save battery
        const now = performance.now();
        if (now - lastDetectTime < DETECT_INTERVAL && state.lastCorners) {
            state.animationId = requestAnimationFrame(detect);
            return;
        }
        lastDetectTime = now;

        const result = detectDocumentCorners(video);

        if (result) {
            const quality = result.quality || 0.4;
            noDocCount = 0;

            // Smooth corners
            const smoothed = smoothCorners(result, state.lastCorners);
            state.lastCorners = smoothed;
            state.detectedCorners = smoothed;

            // Update corner history
            cornerHistory.push({ corners: smoothed, quality });
            if (cornerHistory.length > MAX_HISTORY) cornerHistory.shift();

            // Calculate stability (0-1) based on corner position variance
            const stability = calcStability(cornerHistory);

            // State machine
            if (quality > 0.25 && stability > 0.55) {
                // Good quality AND stable → STABLE
                if (detectState === DETECT_STATE.STABLE) {
                    stableFrames++;
                } else {
                    detectState = DETECT_STATE.STABLE;
                    stableFrames = 1;
                }
            } else if (quality > 0.15) {
                // Detected but not yet stable → TRACKING
                detectState = DETECT_STATE.TRACKING;
                stableFrames = 0;
            } else {
                // Low quality → SEARCHING
                detectState = DETECT_STATE.SEARCHING;
                stableFrames = 0;
            }

            // Position overlay frame
            positionFrameOverlay(smoothed);

            // Update frame style based on state
            updateFrameStyle(detectState, stableFrames, STABLE_REQUIRED);

            // Auto-capture when ready
            if (state.autoCapture &&
                detectState === DETECT_STATE.STABLE &&
                stableFrames >= STABLE_REQUIRED &&
                !state.capturePending) {
                autoCapture();
            }

        } else {
            // No document detected
            noDocCount++;
            cornerHistory = [];
            stableFrames = 0;
            state.lastCorners = null;
            state.detectedCorners = null;

            if (noDocCount > NO_DOC_LIMIT) {
                detectState = DETECT_STATE.SEARCHING;
                updateFrameStyle(DETECT_STATE.SEARCHING, 0, STABLE_REQUIRED);
                showDefaultFrame();
            }
        }

        state.animationId = requestAnimationFrame(detect);
    }

    state.animationId = requestAnimationFrame(detect);
}

function calcStability(history) {
    if (history.length < 4) return 0;

    const n = history[0].corners.length; // 4
    const m = history.length;
    let totalVar = 0;

    for (let i = 0; i < n; i++) {
        let mx = 0, my = 0;
        for (let j = 0; j < m; j++) {
            mx += history[j].corners[i].x;
            my += history[j].corners[i].y;
        }
        mx /= m;
        my /= m;

        let vx = 0, vy = 0;
        for (let j = 0; j < m; j++) {
            vx += (history[j].corners[i].x - mx) ** 2;
            vy += (history[j].corners[i].y - my) ** 2;
        }
        totalVar += (vx + vy) / m;
    }

    const avgVar = totalVar / (n * 2);
    // avgVar ≈ 0.001 for very stable, ≈ 0.05 for jittery
    return Math.max(0, Math.min(1, 1 - avgVar * 30));
}

function updateFrameStyle(state, stableCount, required) {
    const frame = detectionFrame;
    // Remove all state classes
    frame.classList.remove('state-searching', 'state-tracking', 'state-stable');

    switch (state) {
        case DETECT_STATE.STABLE:
            frame.classList.add('state-stable');
            // Show stable progress indicator in the label
            const pct = Math.min(100, Math.round((stableCount / required) * 100));
            frame.querySelector('.frame-label').textContent =
                `已识别 • ${pct}%`;
            scanLines.classList.remove('active');
            break;
        case DETECT_STATE.TRACKING:
            frame.classList.add('state-tracking');
            frame.querySelector('.frame-label').textContent = '正在跟踪…';
            scanLines.classList.remove('active');
            break;
        default: // SEARCHING
            frame.classList.add('state-searching');
            frame.querySelector('.frame-label').textContent = '正在寻找纸张…';
            break;
    }
}

function stopDetectionLoop() {
    state.detectionActive = false;
    if (state.animationId) {
        cancelAnimationFrame(state.animationId);
        state.animationId = null;
    }
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function positionFrameOverlay(corners) {
    if (!corners) return;

    const vw = video.videoWidth, vh = video.videoHeight;
    const rect = video.getBoundingClientRect();
    const videoAspect = vw / vh;
    const rectAspect = rect.width / rect.height;

    let displayW, displayH, offsetX = 0, offsetY = 0;
    if (videoAspect > rectAspect) {
        displayH = rect.height;
        displayW = displayH * videoAspect;
        offsetX = (rect.width - displayW) / 2;
    } else {
        displayW = rect.width;
        displayH = displayW / videoAspect;
        offsetY = (rect.height - displayH) / 2;
    }

    const cornersNorm = corners.map(c => ({
        x: (c.x / vw) * displayW + offsetX,
        y: (c.y / vh) * displayH + offsetY,
    }));

    updateCornerOverlay(cornersNorm);
    detectionFrame.classList.add('detected');
    detectionFrame.classList.remove('visible');
}

function showDefaultFrame() {
    const rect = video.getBoundingClientRect();
    const margin = rect.width * 0.08;
    const cornersNorm = [
        { x: margin, y: rect.height * 0.22 },
        { x: rect.width - margin, y: rect.height * 0.22 },
        { x: rect.width - margin, y: rect.height * 0.78 },
        { x: margin, y: rect.height * 0.78 },
    ];
    updateCornerOverlay(cornersNorm);
    detectionFrame.classList.remove('detected');
    detectionFrame.classList.add('visible');
    scanLines.classList.add('active');
}

function updateCornerOverlay(corners) {
    if (!corners || corners.length !== 4) return;

    const tl = corners[0], tr = corners[1], br = corners[2], bl = corners[3];
    const frame = detectionFrame;

    const cornerEls = {
        tl: frame.querySelector('.frame-corner.tl'),
        tr: frame.querySelector('.frame-corner.tr'),
        bl: frame.querySelector('.frame-corner.bl'),
        br: frame.querySelector('.frame-corner.br'),
    };

    cornerEls.tl.style.left = (tl.x - 14) + 'px';
    cornerEls.tl.style.top = (tl.y - 14) + 'px';
    cornerEls.tr.style.left = (tr.x - 14) + 'px';
    cornerEls.tr.style.top = (tr.y - 14) + 'px';
    cornerEls.bl.style.left = (bl.x - 14) + 'px';
    cornerEls.bl.style.top = (bl.y - 14) + 'px';
    cornerEls.br.style.left = (br.x - 14) + 'px';
    cornerEls.br.style.top = (br.y - 14) + 'px';

    scanLines.classList.remove('active');
}

// ============================
//  Capture
// ============================

function autoCapture() {
    if (state.capturePending) return;
    // Haptic feedback simulation: brief vibration if available
    if (navigator.vibrate) navigator.vibrate(20);

    state.capturePending = true;
    detectState = DETECT_STATE.SEARCHING;
    stableFrames = 0;

    // Visual feedback - flash the screen
    const flash = document.createElement('div');
    flash.style.cssText = 'position:fixed;inset:0;background:white;z-index:50;opacity:0;animation:captureFlash 0.3s ease';
    flash.style.animation = 'none';
    document.body.appendChild(flash);
    flash.style.opacity = '0.8';
    requestAnimationFrame(() => {
        flash.style.transition = 'opacity 0.2s ease';
        flash.style.opacity = '0';
        setTimeout(() => flash.remove(), 300);
    });

    // Wait slightly before capture to avoid capturing the flash
    setTimeout(() => {
        performCapture();
    }, 100);
}

function performCapture() {
    // Stop detection loop while in adjust/preview
    stopDetectionLoop();

    try {
        // Create a canvas at full resolution
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        const canvas = document.createElement('canvas');
        canvas.width = vw;
        canvas.height = vh;
        const ctx = canvas.getContext('2d');

        // If facing front, mirror the image
        if (state.facingMode === 'user') {
            ctx.translate(vw, 0);
            ctx.scale(-1, 1);
        }
        ctx.drawImage(video, 0, 0, vw, vh);

        const dataUrl = canvas.toDataURL('image/jpeg', 0.92);

        // Save detected corners for the adjust screen (in original image coordinates)
        if (state.detectedCorners) {
            state.adjustCorners = state.detectedCorners.map(c => ({ ...c }));
        } else {
            state.adjustCorners = null;
        }

        // Add page
        state.pages.push({ dataUrl, filter: 'original' });
        state.activePage = state.pages.length - 1;

        // Show the adjust screen with ORIGINAL (unwarped) image
        showAdjustScreen(dataUrl);

    } catch (err) {
        console.error('Capture error:', err);
        showToast('扫描失败，请重试');
    }

    state.capturePending = false;
}

// Manual shutter
$('#btn-shutter').addEventListener('click', () => {
    if (state.capturePending) return;
    if (navigator.vibrate) navigator.vibrate(20);
    state.capturePending = true;
    performCapture();
});

// ============================
//  Perspective Transform
// ============================

function perspectiveWarp(imageData, srcW, srcH, corners, dstW, dstH) {
    // Order corners: TL, TR, BR, BL
    const [tl, tr, br, bl] = corners;

    // Determine output dimensions
    const topWidth = Math.sqrt((tr.x - tl.x) ** 2 + (tr.y - tl.y) ** 2);
    const bottomWidth = Math.sqrt((br.x - bl.x) ** 2 + (br.y - bl.y) ** 2);
    const leftHeight = Math.sqrt((bl.x - tl.x) ** 2 + (bl.y - tl.y) ** 2);
    const rightHeight = Math.sqrt((br.x - tr.x) ** 2 + (br.y - tr.y) ** 2);

    const outW = dstW || Math.round(Math.max(topWidth, bottomWidth));
    const outH = dstH || Math.round(Math.max(leftHeight, rightHeight));

    // Compute matrix mapping destination (output) → source (input)
    // This allows us to find the source pixel for each output pixel
    const src = [0, 0, outW, 0, outW, outH, 0, outH];
    const dst = [tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y];

    // Compute perspective transform matrix
    const A = [];
    for (let i = 0; i < 4; i++) {
        const sx = src[i * 2], sy = src[i * 2 + 1];
        const dx = dst[i * 2], dy = dst[i * 2 + 1];
        A.push([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy]);
        A.push([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy]);
    }

    const b = [];
    for (let i = 0; i < 4; i++) {
        b.push(dst[i * 2]);
        b.push(dst[i * 2 + 1]);
    }

    // Solve 8x8 system using Gaussian elimination
    const h = solveLinearSystem(A, b);
    if (!h) return null;

    // Create output image
    const outCanvas = document.createElement('canvas');
    outCanvas.width = outW;
    outCanvas.height = outH;
    const outCtx = outCanvas.getContext('2d');
    const outData = outCtx.createImageData(outW, outH);

    // Inverse warp: for each output pixel, find source pixel
    for (let y = 0; y < outH; y++) {
        for (let x = 0; x < outW; x++) {
            // Apply forward transform to get source coordinates
            const denom = h[6] * x + h[7] * y + 1;
            if (Math.abs(denom) < 1e-10) continue;
            const srcX = (h[0] * x + h[1] * y + h[2]) / denom;
            const srcY = (h[3] * x + h[4] * y + h[5]) / denom;

            // Bilinear interpolation
            const ix = Math.floor(srcX);
            const iy = Math.floor(srcY);
            const fx = srcX - ix;
            const fy = srcY - iy;

            if (ix < 0 || ix >= srcW - 1 || iy < 0 || iy >= srcH - 1) continue;

            const outIdx = (y * outW + x) * 4;

            for (let c = 0; c < 4; c++) {
                const p00 = imageData.data[(iy * srcW + ix) * 4 + c];
                const p10 = imageData.data[(iy * srcW + (ix + 1)) * 4 + c];
                const p01 = imageData.data[((iy + 1) * srcW + ix) * 4 + c];
                const p11 = imageData.data[((iy + 1) * srcW + (ix + 1)) * 4 + c];

                const val = (1 - fx) * (1 - fy) * p00 +
                            fx * (1 - fy) * p10 +
                            (1 - fx) * fy * p01 +
                            fx * fy * p11;

                outData.data[outIdx + c] = val;
            }
        }
    }

    outCtx.putImageData(outData, 0, 0);
    return outCtx.getImageData(0, 0, outW, outH);
}

function solveLinearSystem(A, b) {
    // Gaussian elimination for 8x8 system
    const n = 8;
    const m = A.map((row, i) => [...row, b[i]]);

    for (let col = 0; col < n; col++) {
        // Find pivot
        let maxRow = col;
        for (let row = col + 1; row < n; row++) {
            if (Math.abs(m[row][col]) > Math.abs(m[maxRow][col])) {
                maxRow = row;
            }
        }

        // Swap
        [m[col], m[maxRow]] = [m[maxRow], m[col]];

        if (Math.abs(m[col][col]) < 1e-12) return null;

        // Eliminate
        for (let row = col + 1; row < n; row++) {
            const factor = m[row][col] / m[col][col];
            for (let j = col; j <= n; j++) {
                m[row][j] -= factor * m[col][j];
            }
        }
    }

    // Back substitution
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
        x[i] = m[i][n];
        for (let j = i + 1; j < n; j++) {
            x[i] -= m[i][j] * x[j];
        }
        x[i] /= m[i][i];
    }

    return x;
}

// ============================
//  Corner Adjustment Screen
// ============================

let adjustState = {
    corners: [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9], // normalized [0..1]
    dragging: -1,
    imgW: 0,
    imgH: 0,
    displayW: 0,
    displayH: 0,
    offsetX: 0,
    offsetY: 0,
    imageData: null,
};

function showAdjustScreen(dataUrl) {
    showScreen('#screen-adjust');

    const img = $('#adjust-image');
    img.onload = () => {
        adjustState.imgW = img.naturalWidth;
        adjustState.imgH = img.naturalHeight;

        // Calculate display dimensions (fit within container)
        const container = $('#adjust-container');
        const maxW = container.clientWidth;
        const maxH = container.clientHeight;

        const imgAspect = adjustState.imgW / adjustState.imgH;
        const containerAspect = maxW / maxH;

        if (imgAspect > containerAspect) {
            adjustState.displayW = maxW;
            adjustState.displayH = maxW / imgAspect;
            adjustState.offsetX = 0;
            adjustState.offsetY = (maxH - adjustState.displayH) / 2;
        } else {
            adjustState.displayH = maxH;
            adjustState.displayW = maxH * imgAspect;
            adjustState.offsetX = (maxW - adjustState.displayW) / 2;
            adjustState.offsetY = 0;
        }

        // Set image wrap dimensions to match display area
        const wrap = $('#adjust-image-wrap');
        wrap.style.width = adjustState.displayW + 'px';
        wrap.style.height = adjustState.displayH + 'px';

        // Image fills the wrap
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.marginLeft = '0';
        img.style.marginTop = '0';

        // Overlay fills the wrap (inset: 0 in CSS)

        // Initialize corners (use detected or default)
        if (state.adjustCorners) {
            const vw = adjustState.imgW, vh = adjustState.imgH;
            adjustState.corners = [
                state.adjustCorners[0].x / vw,
                state.adjustCorners[0].y / vh,
                state.adjustCorners[1].x / vw,
                state.adjustCorners[1].y / vh,
                state.adjustCorners[2].x / vw,
                state.adjustCorners[2].y / vh,
                state.adjustCorners[3].x / vw,
                state.adjustCorners[3].y / vh,
            ];
        } else {
            const margin = 0.08;
            adjustState.corners = [
                margin, margin,
                1 - margin, margin,
                1 - margin, 1 - margin,
                margin, 1 - margin,
            ];
        }

        // Store the full image data for later cropping
        const fullCanvas = document.createElement('canvas');
        fullCanvas.width = adjustState.imgW;
        fullCanvas.height = adjustState.imgH;
        const fctx = fullCanvas.getContext('2d');
        fctx.drawImage(img, 0, 0);
        adjustState.imageData = fctx.getImageData(0, 0, adjustState.imgW, adjustState.imgH);

        // (magnifier now samples from the original image directly,
        // so no display-sized intermediate canvas is needed)

        updateAdjustOverlay();
        updateCornerHandles();
    };
    img.src = dataUrl;
}

function updateAdjustOverlay() {
    const corners = adjustState.corners;
    const svgPolygon = $('#adjust-polygon');
    const pts = [];
    for (let i = 0; i < 4; i++) {
        pts.push(`${corners[i * 2] * 100},${corners[i * 2 + 1] * 100}`);
    }
    svgPolygon.setAttribute('points', pts.join(' '));
    updateCornerHandles();
}

function updateCornerHandles() {
    const corners = adjustState.corners;
    const handleIds = ['corner-tl', 'corner-tr', 'corner-br', 'corner-bl'];

    for (let i = 0; i < 4; i++) {
        const el = $(`#${handleIds[i]}`);
        const x = corners[i * 2] * adjustState.displayW;
        const y = corners[i * 2 + 1] * adjustState.displayH;
        el.style.left = x + 'px';
        el.style.top = y + 'px';
    }
}

function initCornerDrag() {
    const handleIds = ['corner-tl', 'corner-tr', 'corner-br', 'corner-bl'];
    const overlay = document.getElementById('adjust-overlay');

    // Create magnifier element with canvas for zoomed content
    const magnifier = document.createElement('div');
    magnifier.id = 'corner-magnifier';
    magnifier.innerHTML = '<div class="magnifier-ring"><canvas id="magnifier-canvas" width="60" height="60"></canvas></div>';
    overlay.appendChild(magnifier);
    const magCanvas = document.getElementById('magnifier-canvas');
    const magCtx = magCanvas.getContext('2d');

    function updateMagnifier() {
        var idx = adjustState.dragging;
        if (idx < 0) return;
        var img = document.getElementById('adjust-image');
        if (!img || !img.complete) return;
        var normX = adjustState.corners[idx * 2];
        var normY = adjustState.corners[idx * 2 + 1];
        // Sample directly from the original image in natural coordinates,
        // avoiding coordinate mismatch from the intermediate display canvas
        // whose dimensions are rounded (Math.round) vs. unrounded DOM coords.
        var outSize = 60;
        var zoom = 3;
        var displayScaleX = adjustState.displayW / adjustState.imgW;
        var srcSize = (outSize / zoom) / displayScaleX;
        var cx = normX * adjustState.imgW;
        var cy = normY * adjustState.imgH;
        var half = srcSize / 2;
        var sx = Math.max(0, Math.min(adjustState.imgW - srcSize, cx - half));
        var sy = Math.max(0, Math.min(adjustState.imgH - srcSize, cy - half));
        magCtx.clearRect(0, 0, outSize, outSize);
        magCtx.save();
        magCtx.beginPath();
        magCtx.arc(outSize / 2, outSize / 2, outSize / 2 - 1, 0, Math.PI * 2);
        magCtx.clip();
        magCtx.drawImage(img, sx, sy, srcSize, srcSize, 0, 0, outSize, outSize);
        magCtx.restore();
    }

    function positionMagnifier() {
        var idx = adjustState.dragging;
        if (idx < 0) return;
        magnifier.style.left = (adjustState.corners[idx * 2] * adjustState.displayW) + 'px';
        magnifier.style.top = (adjustState.corners[idx * 2 + 1] * adjustState.displayH) + 'px';
    }

    // Calculate a corner's screen position for distance testing
    function getCornerScreen(index) {
        var wrap = $('#adjust-image-wrap');
        var r = wrap.getBoundingClientRect();
        return {
            x: r.left + adjustState.corners[index * 2] * adjustState.displayW,
            y: r.top + adjustState.corners[index * 2 + 1] * adjustState.displayH
        };
    }

    // Find which corner is nearest to the touch point (within 80px threshold)
    function findNearest(cx, cy) {
        var best = -1;
        var bestDist = 300;
        for (var i = 0; i < 4; i++) {
            var p = getCornerScreen(i);
            var d = Math.sqrt((cx - p.x) * (cx - p.x) + (cy - p.y) * (cy - p.y));
            if (d < bestDist) { bestDist = d; best = i; }
        }
        return best;
    }

    // Track drag origin for relative (delta-based) movement
    var dragOriginX = 0, dragOriginY = 0;
    var dragCornerOriginX = 0, dragCornerOriginY = 0;

    function startDrag(index, clientX, clientY) {
        adjustState.dragging = index;
        dragOriginX = clientX;
        dragOriginY = clientY;
        dragCornerOriginX = adjustState.corners[index * 2];
        dragCornerOriginY = adjustState.corners[index * 2 + 1];
        var el = $('#' + handleIds[index]);
        if (el) { el.classList.add('is-dragging'); el.style.opacity = '0'; }
        magnifier.classList.add('active');
        positionMagnifier();
        updateMagnifier();
        // Prevent text selection during drag
        document.body.style.userSelect = 'none';
        document.body.style.webkitUserSelect = 'none';
    }

    function moveDrag(clientX, clientY) {
        if (adjustState.dragging < 0) return;
        var dx = (clientX - dragOriginX) / adjustState.displayW;
        var dy = (clientY - dragOriginY) / adjustState.displayH;
        var margin = 0.05;
        var normX = Math.max(margin, Math.min(1 - margin, dragCornerOriginX + dx));
        var normY = Math.max(margin, Math.min(1 - margin, dragCornerOriginY + dy));
        adjustState.corners[adjustState.dragging * 2] = normX;
        adjustState.corners[adjustState.dragging * 2 + 1] = normY;
        updateAdjustOverlay();
        positionMagnifier();
        updateMagnifier();
    }

    function endDrag() {
        if (adjustState.dragging >= 0) {
            var el = $('#' + handleIds[adjustState.dragging]);
            if (el) { el.classList.remove('is-dragging'); el.style.opacity = '1'; }
        }
        magnifier.classList.remove('active');
        adjustState.dragging = -1;
        document.body.style.userSelect = '';
        document.body.style.webkitUserSelect = '';
    }

    // === Touch handling on overlay (distance-based) ===
    overlay.addEventListener('touchstart', function (e) {
        var t = e.touches[0];
        var idx = findNearest(t.clientX, t.clientY);
        if (idx >= 0) {
            e.preventDefault();
            startDrag(idx, t.clientX, t.clientY);
        }
    }, { passive: false });

    overlay.addEventListener('touchmove', function (e) {
        if (adjustState.dragging < 0) return;
        e.preventDefault();
        var t = e.touches[0];
        moveDrag(t.clientX, t.clientY);
    }, { passive: false });

    overlay.addEventListener('touchend', function (e) {
        if (adjustState.dragging < 0) return;
        e.preventDefault();
        endDrag();
    }, { passive: false });

    overlay.addEventListener('touchcancel', function () {
        if (adjustState.dragging >= 0) endDrag();
    });

    // === Mouse handling ===
    overlay.addEventListener('mousedown', function (e) {
        var idx = findNearest(e.clientX, e.clientY);
        if (idx >= 0) { startDrag(idx, e.clientX, e.clientY); }
    });

    document.addEventListener('mousemove', function (e) {
        moveDrag(e.clientX, e.clientY);
    });

    document.addEventListener('mouseup', function () {
        endDrag();
    });
}

// ============================
//  Image Filters
// ============================

function applyFilterToImage(imageData, filter) {
    const data = imageData.data;
    const len = data.length / 4;

    switch (filter) {
        case 'gray': {
            for (let i = 0; i < len; i++) {
                const idx = i * 4;
                const gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                data[idx] = gray;
                data[idx + 1] = gray;
                data[idx + 2] = gray;
            }
            break;
        }

        case 'bw': {
            // First convert to grayscale
            for (let i = 0; i < len; i++) {
                const idx = i * 4;
                const gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                data[idx] = gray;
                data[idx + 1] = gray;
                data[idx + 2] = gray;
            }
            // Then apply adaptive threshold using local mean
            const w = imageData.width;
            const h = imageData.height;
            const grayValues = new Float32Array(len);
            for (let i = 0; i < len; i++) {
                grayValues[i] = data[i * 4];
            }

            const blockSize = Math.max(8, Math.floor(Math.min(w, h) / 16));
            const halfBlock = Math.floor(blockSize / 2);

            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    let sum = 0, count = 0;
                    for (let dy = -halfBlock; dy <= halfBlock; dy++) {
                        for (let dx = -halfBlock; dx <= halfBlock; dx++) {
                            const nx = x + dx, ny = y + dy;
                            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                                sum += grayValues[ny * w + nx];
                                count++;
                            }
                        }
                    }
                    const localMean = sum / count;
                    const idx = (y * w + x) * 4;
                    const val = grayValues[y * w + x] > localMean - 5 ? 255 : 0;
                    data[idx] = val;
                    data[idx + 1] = val;
                    data[idx + 2] = val;
                }
            }
            break;
        }

        case 'enhance': {
            // Auto-enhance: increase contrast and sharpen
            // First, compute histogram for contrast stretching
            let min = 255, max = 0;
            for (let i = 0; i < len; i++) {
                const idx = i * 4;
                const gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                if (gray < min) min = gray;
                if (gray > max) max = gray;
            }

            const range = max - min;
            if (range > 10) {
                const scale = 255 / range;
                for (let i = 0; i < len; i++) {
                    const idx = i * 4;
                    data[idx] = Math.max(0, Math.min(255, (data[idx] - min) * scale));
                    data[idx + 1] = Math.max(0, Math.min(255, (data[idx + 1] - min) * scale));
                    data[idx + 2] = Math.max(0, Math.min(255, (data[idx + 2] - min) * scale));
                }
            }

            // Simple unsharp mask (sharpen)
            const w = imageData.width;
            const h = imageData.height;
            const orig = new Uint8ClampedArray(data);

            const strength = 0.3;
            for (let y = 1; y < h - 1; y++) {
                for (let x = 1; x < w - 1; x++) {
                    const idx = (y * w + x) * 4;
                    // 3x3 blur
                    for (let c = 0; c < 3; c++) {
                        let blur = 0;
                        for (let dy = -1; dy <= 1; dy++) {
                            for (let dx = -1; dx <= 1; dx++) {
                                blur += orig[((y + dy) * w + (x + dx)) * 4 + c];
                            }
                        }
                        blur /= 9;
                        const sharpened = orig[idx + c] + strength * (orig[idx + c] - blur);
                        data[idx + c] = Math.max(0, Math.min(255, sharpened));
                    }
                }
            }
            break;
        }

        case 'original':
        default:
            // No change
            break;
    }

    return imageData;
}

function renderFilteredPage(pageIndex) {
    const page = state.pages[pageIndex];
    if (!page) return;

    const img = $('#preview-image');
    if (page.filter === 'original') {
        // Show original
        img.src = page.dataUrl;
        return;
    }

    // Apply filter
    const canvas = document.createElement('canvas');
    const tempImg = new Image();
    tempImg.onload = () => {
        canvas.width = tempImg.naturalWidth;
        canvas.height = tempImg.naturalHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(tempImg, 0, 0);
        let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        imageData = applyFilterToImage(imageData, page.filter);
        ctx.putImageData(imageData, 0, 0);
        img.src = canvas.toDataURL('image/jpeg', 0.92);
    };
    tempImg.src = page.dataUrl;
}

// ============================
//  PDF Generation
// ============================

async function generatePDF(pages) {
    if (pages.length === 0) {
        showToast('没有可存储的页面');
        return null;
    }

    try {
        if (!window.jspdf) {
            showToast('PDF 库加载中，请检查网络连接');
            return null;
        }
        const { jsPDF } = window.jspdf;
        const pdf = new jsPDF('p', 'mm', 'a4');
        const pageWidth = pdf.internal.pageSize.getWidth();
        const pageHeight = pdf.internal.pageSize.getHeight();

        for (let i = 0; i < pages.length; i++) {
            if (i > 0) pdf.addPage();

            const page = pages[i];
            const img = await loadImage(page.dataUrl);

            // Calculate image dimensions to fit A4
            const imgAspect = img.width / img.height;
            const pageAspect = pageWidth / pageHeight;

            let imgWidth, imgHeight, x, y;
            if (imgAspect > pageAspect) {
                imgWidth = pageWidth;
                imgHeight = pageWidth / imgAspect;
                x = 0;
                y = (pageHeight - imgHeight) / 2;
            } else {
                imgHeight = pageHeight;
                imgWidth = pageHeight * imgAspect;
                x = (pageWidth - imgWidth) / 2;
                y = 0;
            }

            // Apply filter if needed
            let src = page.dataUrl;
            if (page.filter !== 'original') {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                imageData = applyFilterToImage(imageData, page.filter);
                ctx.putImageData(imageData, 0, 0);
                src = canvas.toDataURL('image/jpeg', 0.92);
            }

            pdf.addImage(src, 'JPEG', x, y, imgWidth, imgHeight);
        }

        const pdfBlob = pdf.output('blob');
        return pdfBlob;
    } catch (err) {
        console.error('PDF generation error:', err);
        showToast('PDF 生成失败');
        return null;
    }
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 5000);
}

// ============================
//  Event Handlers
// ============================

// Flash toggle
$('#btn-flash').addEventListener('click', toggleFlash);

// Close scanner (stop camera)
$('#btn-close').addEventListener('click', () => {
    stopCamera();
    stopDetectionLoop();
    // The prompt shows again, or we just show the prompt
    showCameraPrompt();
});

// Start camera
$('#btn-start-camera').addEventListener('click', async () => {
    $('#btn-start-camera').textContent = '正在启动…';
    $('#btn-start-camera').disabled = true;
    const success = await initCamera();
    if (success) {
        hideCameraPrompt();
        startDetectionLoop();
    } else {
        $('#btn-start-camera').textContent = '无法访问相机';
        showToast('请在系统设置中允许相机访问');
    }
    $('#btn-start-camera').disabled = false;
});

// Gallery
$('#btn-gallery').addEventListener('click', () => {
    $('#file-input').click();
});

$('#file-input').addEventListener('change', (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    for (const file of files) {
        const reader = new FileReader();
        reader.onload = (ev) => {
            const dataUrl = ev.target.result;
            state.pages.push({ dataUrl, filter: 'original' });
            state.activePage = state.pages.length - 1;
            showAdjustScreen(dataUrl);
        };
        reader.readAsDataURL(file);
    }

    // Reset input so same file can be selected again
    e.target.value = '';
});

// Adjust screen buttons
$('#btn-adjust-retake').addEventListener('click', () => {
    // Remove last page and go back to scanner
    if (state.pages.length > 0) {
        state.pages.pop();
        state.activePage = state.pages.length - 1;
    }
    showScreen('#screen-scanner');
    state.detectedCorners = null;
    state.adjustCorners = null;
    startDetectionLoop();
});

$('#btn-adjust-done').addEventListener('click', () => {
    // Apply perspective crop with current (user-adjusted) corners
    if (adjustState.imageData) {
        const corners = [];
        for (let i = 0; i < 4; i++) {
            corners.push({
                x: adjustState.corners[i * 2] * adjustState.imgW,
                y: adjustState.corners[i * 2 + 1] * adjustState.imgH,
            });
        }

        const cropped = perspectiveWarp(
            adjustState.imageData,
            adjustState.imgW,
            adjustState.imgH,
            corners,
            null, null
        );

        if (cropped) {
            const canvas = document.createElement('canvas');
            canvas.width = cropped.width;
            canvas.height = cropped.height;
            const ctx = canvas.getContext('2d');
            ctx.putImageData(cropped, 0, 0);
            const newDataUrl = canvas.toDataURL('image/jpeg', 0.92);

            // Update the current page with the warped image
            state.pages[state.pages.length - 1] = {
                ...state.pages[state.pages.length - 1],
                dataUrl: newDataUrl,
            };
        }
    }

    // Go to preview
    showPreviewScreen();
});

// Preview screen
function showPreviewScreen() {
    showScreen('#screen-preview');
    updatePageThumbnails();
    renderFilteredPage(state.activePage);
    updateFilterSelection();
}

// Preview buttons
$('#btn-preview-back').addEventListener('click', () => {
    stopCamera();
    showCameraPrompt();
});

$('#btn-preview-save').addEventListener('click', async () => {
    const btn = $('#btn-preview-save');
    btn.textContent = '生成中…';
    btn.disabled = true;

    const pdfBlob = await generatePDF(state.pages);
    if (pdfBlob) {
        downloadBlob(pdfBlob, `扫描文稿_${formatDate()}.pdf`);
        showToast('PDF 已保存');
    }

    btn.textContent = '存储';
    btn.disabled = false;
});

function formatDate() {
    const d = new Date();
    return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`;
}

// Filter selection
$$('.filter-option').forEach(el => {
    el.addEventListener('click', () => {
        const filter = el.dataset.filter;
        const pageIdx = state.activePage;
        if (pageIdx < 0 || pageIdx >= state.pages.length) return;

        state.pages[pageIdx].filter = filter;
        renderFilteredPage(pageIdx);
        updateFilterSelection();
    });
});

function updateFilterSelection() {
    const page = state.pages[state.activePage];
    if (!page) return;

    $$('.filter-option').forEach(el => {
        el.classList.toggle('active', el.dataset.filter === page.filter);
    });
}

// Share button
$('#btn-share').addEventListener('click', async () => {
    const page = state.pages[state.activePage];
    if (!page) return;

    try {
        const response = await fetch(page.dataUrl);
        const blob = await response.blob();

        if (navigator.share) {
            await navigator.share({
                title: '扫描文稿',
                files: [new File([blob], `扫描_${formatDate()}.jpg`, { type: 'image/jpeg' })],
            });
        } else {
            downloadBlob(blob, `扫描_${formatDate()}.jpg`);
        }
    } catch (err) {
        console.error('Share error:', err);
        showToast('分享失败');
    }
});

// Auto-capture toggle
$('#btn-auto').addEventListener('click', () => {
    state.autoCapture = !state.autoCapture;
    $('#btn-auto').classList.toggle('active', state.autoCapture);
    if (state.autoCapture) {
        stableFrames = 0;
        showToast('自动扫描已开启');
    } else {
        showToast('自动扫描已关闭');
    }
});

// Add page from preview
$('#btn-add-page').addEventListener('click', () => {
    showScreen('#screen-scanner');
    startDetectionLoop();
});

// ============================
//  Page Thumbnails
// ============================

function updatePageThumbnails() {
    const scroll = $('#pages-scroll');
    scroll.innerHTML = '';

    state.pages.forEach((page, idx) => {
        const thumb = document.createElement('div');
        thumb.className = 'page-thumb' + (idx === state.activePage ? ' active' : '');

        const img = document.createElement('img');
        img.src = page.dataUrl;
        img.alt = `第 ${idx + 1} 页`;
        thumb.appendChild(img);

        const del = document.createElement('button');
        del.className = 'page-delete';
        del.innerHTML = '×';
        del.addEventListener('click', (e) => {
            e.stopPropagation();
            state.pages.splice(idx, 1);
            if (state.activePage >= state.pages.length) {
                state.activePage = state.pages.length - 1;
            }
            if (state.pages.length === 0) {
                showScreen('#screen-scanner');
                startDetectionLoop();
                return;
            }
            updatePageThumbnails();
            renderFilteredPage(state.activePage);
            updateFilterSelection();
        });
        thumb.appendChild(del);

        thumb.addEventListener('click', () => {
            state.activePage = idx;
            updatePageThumbnails();
            renderFilteredPage(state.activePage);
            updateFilterSelection();
        });

        scroll.appendChild(thumb);
    });
}

// ============================
//  Camera Prompt
// ============================

function showCameraPrompt() {
    cameraPrompt.classList.add('visible');
    $('#btn-start-camera').textContent = '允许访问相机';
    if (state.pages.length > 0) {
        // We have pages, also show the preview button or...
        // Actually, just let them start camera again
    }
}

function hideCameraPrompt() {
    cameraPrompt.classList.remove('visible');
}

// ============================
//  Styles injection for animations
// ============================

// Add capture flash animation
const styleSheet = document.createElement('style');
styleSheet.textContent = `
    @keyframes captureFlash {
        0% { opacity: 0.8; }
        100% { opacity: 0; }
    }
`;
document.head.appendChild(styleSheet);

// ============================
//  Initialize
// ============================

function init() {
    showCameraPrompt();
    initCornerDrag();

    // If there's no camera, still allow image upload
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        document.querySelector('.camera-prompt-icon').innerHTML = `
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <polyline points="21 15 16 10 5 21"/>
            </svg>
        `;
        document.querySelector('#camera-prompt h2').textContent = '选择图片扫描';
        document.querySelector('#camera-prompt p').textContent = '从相册中选择文稿图片';
        $('#btn-start-camera').textContent = '选择图片';
        $('#btn-start-camera').addEventListener('click', () => {
            $('#file-input').click();
        });
    }

    // Check if we can detect orientation and adjust
    if (screen && screen.orientation) {
        screen.orientation.addEventListener('change', () => {
            // Recalculate layout on orientation change
            if (state.mode === 'adjust' && adjustState.imageData) {
                // The container will reflow, corners may need recalculation
            }
        });
    }
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
