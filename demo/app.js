/*
 * app.js — the demo: a body bends, and the network predicts its potential
 * phi_x for every pose, live, in this browser.
 *
 * Everything is computed here, every frame, with nothing cached between frames:
 * the bend, the canonical frame the network was trained in, the vertex normals,
 * the farthest-point sampling and the network itself. The page downloads one
 * rest mesh per body (assets/shapes.bin, from experiments/webdemo_shapes.py)
 * and the weights. The time it reports covers all of it.
 */
import * as THREE from 'three';
import { canonicalize, vertexNormals, meanZero } from './nsp.js';

// The colours below ARE the data, so no working-space conversion may sit
// between the numbers here, the pixels on the mesh and the CSS legend bar.
THREE.ColorManagement.enabled = false;

const ASSETS = 'assets/';

/* Potential: diverging blue -> green -> red, green exactly at zero. */
const PHI_MAP = [
  [0.071, 0.192, 0.514],
  [0.125, 0.404, 0.769],
  [0.176, 0.616, 0.749],
  [0.294, 0.745, 0.451],
  [0.796, 0.769, 0.239],
  [0.898, 0.478, 0.145],
  [0.671, 0.094, 0.125],
];
const LUT_N = 512;
function ramp(t) {
  t = Math.min(1, Math.max(0, t)) * (PHI_MAP.length - 1);
  const i = Math.min(PHI_MAP.length - 2, Math.floor(t)), f = t - i;
  const a = PHI_MAP[i], b = PHI_MAP[i + 1];
  return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
}
const LUT = new Float32Array(3 * LUT_N);
for (let i = 0; i < LUT_N; i++) LUT.set(ramp(i / (LUT_N - 1)), 3 * i);
const CSS_RAMP = `linear-gradient(to right, ${Array.from({ length: 17 }, (_, i) =>
  `rgb(${ramp(i / 16).map((v) => Math.round(255 * v)).join(',')}) ${(100 * i / 16).toFixed(1)}%`).join(', ')})`;

/* ------------------------------------------------------------------ assets */
async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

async function getBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.arrayBuffer();
}

/** fetch with a byte-progress callback, so the page can say how far it got. */
async function fetchProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body) return res.arrayBuffer();
  const chunks = [];
  let got = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    onBytes(got, total);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  return buf.buffer;
}

/* ------------------------------------------------------------ deformation */
/** Barr bend, as experiments/webdemo_build.bend: every vertex turns about
 *  `axis` by k times its `up` coordinate, measured from the mean over vertices.
 *  A proper rotation per vertex, so the winding and the outward normals survive. */
function bend(rest, k, up, axis, out) {
  const n = rest.length / 3;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += rest[3 * i + up];
  mean /= n;
  const [p, q] = [0, 1, 2].filter((a) => a !== axis);
  for (let i = 0; i < n; i++) {
    const t = k * (rest[3 * i + up] - mean), c = Math.cos(t), s = Math.sin(t);
    const x = rest[3 * i + p], y = rest[3 * i + q];
    out[3 * i + p] = c * x - s * y;
    out[3 * i + q] = s * x + c * y;
    out[3 * i + axis] = rest[3 * i + axis];
  }
  return out;
}

/* -------------------------------------------------------------------- view */
/* One orbit state, eased towards its targets every drawn frame. */
const orbit = { az: 0.34, pol: 1.30, zoom: 1, azT: 0.34, polT: 1.30, zoomT: 1 };
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/* Embedded in a page, a wheel or a vertical swipe over the demo belongs to that
 * page, or nobody can scroll past it: zoom then needs Ctrl or Cmd (a trackpad
 * pinch arrives as a wheel with ctrlKey), and a vertical swipe scrolls. The
 * page's demo-window.js posts {fullscreen} when it promotes this frame to a full
 * screen overlay, where the canvas takes every gesture again. */
let embedded = window.parent !== window;
addEventListener('message', (e) => {
  if (e.source !== window.parent || typeof e.data?.fullscreen !== 'boolean') return;
  embedded = !e.data.fullscreen;
  document.documentElement.classList.toggle('fullscreen', e.data.fullscreen);
  for (const c of document.querySelectorAll('canvas')) c.style.touchAction = embedded ? 'pan-y' : 'none';
});

/* Lights live in view space: the shading stays put while the body turns, so a
 * moving highlight never gets mistaken for a moving field. */
const VERT = `
attribute vec3 color;
varying vec3 vN; varying vec3 vV; varying vec3 vC;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = normalize(-mv.xyz);
  vC = color;
  gl_Position = projectionMatrix * mv;
}`;
const FRAG = `
varying vec3 vN; varying vec3 vV; varying vec3 vC;
void main() {
  vec3 N = normalize(vN);
  if (!gl_FrontFacing) N = -N;
  vec3 V = normalize(vV);
  vec3 L1 = normalize(vec3(0.32, 0.52, 0.86));
  vec3 L2 = normalize(vec3(-0.70, -0.26, -0.52));
  float lam = 0.70 + 0.28 * max(dot(N, L1), 0.0) + 0.09 * max(dot(N, L2), 0.0);
  float fres = pow(1.0 - max(dot(N, V), 0.0), 3.0);
  float spec = pow(max(dot(reflect(-L1, N), V), 0.0), 28.0);
  vec3 col = vC * lam + vec3(0.10, 0.10, 0.13) * fres + 0.09 * spec;
  gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}`;

class View {
  constructor(canvas, n, nf) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.05, 100);

    const g = new THREE.BufferGeometry();
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(3 * nf), 1));
    for (const name of ['position', 'normal', 'color']) {
      g.setAttribute(name, new THREE.BufferAttribute(new Float32Array(3 * n), 3));
    }
    this.geom = g;
    const mesh = new THREE.Mesh(g, new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG }));
    // The mesh changes every frame and always lies inside the unit sphere; let
    // three skip the bounding-volume bookkeeping rather than get it wrong.
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    this.baseDist = 4;
    this._w = this._h = 0;
    attachOrbit(canvas);
  }

  /** @returns true if the canvas changed size (so the framing must be redone). */
  resize() {
    const w = this.canvas.clientWidth | 0, h = this.canvas.clientHeight | 0;
    if (!w || !h || (w === this._w && h === this._h)) return false;
    this._w = w; this._h = h;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    return true;
  }

  /**
   * Exact framing for the current orbit direction, against `cloud` (a fixed
   * sample of the vertices of the whole bend). A unit bounding sphere would
   * waste about a third of the canvas; the silhouette of the sample is a bound
   * and stays still while the body bends.
   */
  fit(cloud) {
    const tanV = Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2);
    const tanH = tanV * this.camera.aspect;
    const sp = Math.sin(orbit.pol), cp = Math.cos(orbit.pol);
    const sa = Math.sin(orbit.az), ca = Math.cos(orbit.az);
    const ux = sp * sa, uy = cp, uz = sp * ca;      // origin -> camera
    const rx = ca, rz = -sa;                        // screen right (ry is 0)
    const vx = -cp * sa, vy = sp, vz = -cp * ca;    // screen up = u x right
    let d = 0;
    for (let i = 0; i < cloud.length; i += 3) {
      const px = cloud[i], py = cloud[i + 1], pz = cloud[i + 2];
      const s = px * ux + py * uy + pz * uz;
      const a = Math.abs(px * rx + pz * rz) / tanH;
      const b = Math.abs(px * vx + py * vy + pz * vz) / tanV;
      d = Math.max(d, s + Math.max(a, b));
    }
    this.baseDist = d * 1.04;   // the vertices the cloud skipped
  }

  setFaces(F) {
    this.geom.index.array.set(F);
    this.geom.index.needsUpdate = true;
  }

  setGeometry(V, N) {
    this.geom.attributes.position.array.set(V);
    this.geom.attributes.normal.array.set(N);
    this.geom.attributes.position.needsUpdate = true;
    this.geom.attributes.normal.needsUpdate = true;
  }

  /** Colour by phi on a symmetric scale; max 0 means no prediction yet (grey). */
  paint(phi, max) {
    const c = this.geom.attributes.color.array;
    for (let k = 0; k < phi.length; k++) {
      if (!max) { c[3 * k] = 0.74; c[3 * k + 1] = 0.75; c[3 * k + 2] = 0.77; continue; }
      const j = 3 * ((clamp(0.5 * (phi[k] / max + 1), 0, 1) * (LUT_N - 1)) | 0);
      c[3 * k] = LUT[j]; c[3 * k + 1] = LUT[j + 1]; c[3 * k + 2] = LUT[j + 2];
    }
    this.geom.attributes.color.needsUpdate = true;
  }

  draw() {
    const d = this.baseDist * orbit.zoom, sp = Math.sin(orbit.pol);
    this.camera.position.set(d * sp * Math.sin(orbit.az), d * Math.cos(orbit.pol),
                             d * sp * Math.cos(orbit.az));
    this.camera.lookAt(0, 0, 0);
    this.renderer.render(this.scene, this.camera);
  }
}

/* Only the primary pointer orbits: with two fingers down, taking each move
 * against the other finger's position would add their separation to the angle. */
function attachOrbit(el) {
  let last = null;
  el.style.touchAction = embedded ? 'pan-y' : 'none';
  el.style.cursor = 'grab';
  el.addEventListener('pointerdown', (e) => {
    if (!e.isPrimary) return;
    last = [e.clientX, e.clientY]; el.style.cursor = 'grabbing';
    el.setPointerCapture(e.pointerId);
  });
  el.addEventListener('pointermove', (e) => {
    if (!last || !e.isPrimary) return;
    orbit.azT -= (e.clientX - last[0]) * 0.0072;
    orbit.polT = clamp(orbit.polT - (e.clientY - last[1]) * 0.0072, 0.12, Math.PI - 0.12);
    last = [e.clientX, e.clientY];
  });
  const end = (e) => {
    if (!e.isPrimary) return;
    last = null; el.style.cursor = 'grab';
    try { el.releasePointerCapture(e.pointerId); } catch { /* already gone */ }
  };
  el.addEventListener('pointerup', end);
  el.addEventListener('pointercancel', end);
  el.addEventListener('wheel', (e) => {
    if (embedded && !e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    orbit.zoomT = clamp(orbit.zoomT * Math.exp(e.deltaY * 0.0012), 0.5, 2.6);
  }, { passive: false });
}

/* -------------------------------------------------------------------- main */
const el = (id) => document.getElementById(id);

/* The network lives in a worker, so its forward pass never blocks the page: the
 * body and the camera keep moving at display rate while a pose is computing.
 * Positions and normals are copied over (the main thread still needs them to
 * draw); the potential comes back as a transferred buffer. One request is
 * outstanding at a time, which is what the inference loop wants. */
class Net {
  constructor() {
    this.seq = 0;
    this.pending = new Map();
    this.backend = '';
    this.threads = 1;
    this.ready = new Promise((resolve, reject) => { this._ok = resolve; this._fail = reject; });
    this.ready.catch(() => {});              // surfaced through the load await
    this.worker = new Worker(new URL('nsp-worker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'ready') {
        this.backend = m.backend; this.threads = m.threads; this._ok(this);
      } else if (m.type === 'result') {
        const p = this.pending.get(m.id); this.pending.delete(m.id); p.resolve(m.phi);
      } else if (m.type === 'error') {
        if (m.id != null && this.pending.has(m.id)) {
          const p = this.pending.get(m.id); this.pending.delete(m.id); p.reject(new Error(m.error));
        } else {
          this._fail(new Error(m.error));
        }
      }
    };
    this.worker.onerror = (e) => this._fail(new Error(e.message || 'worker failed'));
  }

  /** Hand the graph to the worker and start it compiling. Transfers the bytes. */
  init(onnx, nPoints) {
    this.worker.postMessage({ type: 'init', onnx, nPoints }, [onnx]);
    return this.ready;
  }

  predict(positions, normals) {
    const id = this.seq++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: 'predict', id, positions, normals });
    });
  }
}

(async function main() {
  const setStatus = (t, p) => {
    el('statusText').textContent = t;
    if (p != null) el('progBar').style.width = `${Math.round(p)}%`;
  };
  el('bar').style.background = CSS_RAMP;

  /* ---- the bodies first: the page has something to show long before the
     network has finished downloading and compiling ---- */
  setStatus('loading…', 2);
  const meta = await getJSON(ASSETS + 'shapes.json');
  const raw = await getBytes(ASSETS + 'shapes.bin');
  const n = meta.n_vertices, nf = meta.n_faces, B = meta.bend, K = B.frames;
  const bodies = new Map(meta.bodies.map((b) => [b.name, {
    azimuth: b.view_azimuth,
    rest: new Float32Array(raw, b.offset, 3 * n),
    faces: new Uint16Array(raw, b.offset + 12 * n, 3 * nf),
  }]));

  const view = new View(el('view'), n, nf);
  const V = new Float32Array(3 * n), N = new Float32Array(3 * n), phi = new Float64Array(n);
  let body, fitCloud, phiMax = 0, frame = 0, playing = true, dirty = true, redraw = true;
  let lastFitAz = 1e9, lastFitPol = 1e9;   // far from any real angle: fit on the next frame

  /** This frame's pose of the current body, in the canonical frame, into V and N. */
  function pose(i) {
    bend(body.rest, B.peak_radians_per_unit_height * Math.sin(2 * Math.PI * i / K),
         B.up_axis, B.bend_axis, V);
    canonicalize(V, body.faces);
    vertexNormals(V, body.faces, N);
  }

  function select(name) {
    body = bodies.get(name);
    view.setFaces(body.faces);
    // Frame the camera against every 3rd vertex of every 4th pose.
    const pts = [];
    for (let k = 0; k < K; k += 4) {
      pose(k);
      for (let i = 0; i < n; i += 3) pts.push(V[3 * i], V[3 * i + 1], V[3 * i + 2]);
    }
    fitCloud = Float32Array.from(pts);
    lastFitAz = 1e9;
    orbit.azT = body.azimuth;
    frame = 0; phiMax = 0; dirty = true;
    pose(0);
    view.setGeometry(V, N);
    view.paint(phi, 0);
    redraw = true;
    for (const b of document.querySelectorAll('[data-body]')) {
      b.setAttribute('aria-pressed', String(b.dataset.body === name));
    }
  }
  select('bunny');

  // Nothing is computed or drawn while the demo is out of view. Embedded, this
  // frame shares the page's main thread, and single-threaded WASM would block
  // the page the visitor is reading. In an iframe the observer's implicit root
  // is the top-level viewport, so scrolling the page away counts.
  let onScreen = true;
  new IntersectionObserver(([e]) => {
    onScreen = e.isIntersecting;
    if (onScreen) redraw = true;
  }).observe(el('view'));

  // Draw only what changed: an unchanged scene is not redrawn at 60 Hz.
  (function render() {
    requestAnimationFrame(render);
    if (!onScreen) return;
    orbit.az += (orbit.azT - orbit.az) * 0.22;
    orbit.pol += (orbit.polT - orbit.pol) * 0.22;
    orbit.zoom += (orbit.zoomT - orbit.zoom) * 0.22;
    // Re-fit only when the view actually moved: the fit is a pass over ~10k
    // points, cheap but not free at 60 Hz for no reason.
    const moved = Math.abs(orbit.az - lastFitAz) + Math.abs(orbit.pol - lastFitPol) > 1e-4;
    const zooming = Math.abs(orbit.zoomT - orbit.zoom) > 1e-4;
    const resized = view.resize();
    if (resized || moved) view.fit(fitCloud);
    if (resized || moved || zooming || redraw) view.draw();
    redraw = false;
    if (moved) { lastFitAz = orbit.az; lastFitPol = orbit.pol; }
  })();

  // Until the network is up, the body bends uncoloured.
  const preview = setInterval(() => {
    if (!onScreen) return;
    frame = (frame + 1) % K;
    pose(frame);
    view.setGeometry(V, N);
    redraw = true;
  }, 55);

  /* ---- the network ---- */
  const net = new Net();               // starts the worker and its ORT import now
  const onnx = await fetchProgress(ASSETS + 'nsp_phix.onnx',
    (got, total) => setStatus('loading the network…', total ? 5 + 85 * got / total : 40));
  setStatus('compiling…', 92);
  await net.init(onnx, n);
  const backend = net.backend === 'webgpu' ? 'WebGPU' : `WASM, ${net.threads} thread${net.threads > 1 ? 's' : ''}`;

  const recent = [];
  async function computeFrame(i) {
    const b = body, t0 = performance.now();
    pose(i);
    const out = await net.predict(V, N);
    if (b !== body) return;     // the body changed while the network ran
    meanZero(out, phi);
    const ms = performance.now() - t0;
    if (!phiMax) {               // one colour scale per body, from its rest pose
      phiMax = Math.ceil(100 * phi.reduce((m, v) => Math.max(m, Math.abs(v)), 0)) / 100;
      el('lo').textContent = `−${phiMax.toFixed(2)}`;
      el('hi').textContent = `+${phiMax.toFixed(2)}`;
    }
    view.setGeometry(V, N);
    view.paint(phi, phiMax);
    redraw = true;
    recent.push(ms);
    if (recent.length > 20) recent.shift();
    const med = [...recent].sort((a, b) => a - b)[Math.floor(recent.length / 2)];
    el('time').textContent = `${med.toFixed(0)} ms per pose · ${backend}`;
  }

  setStatus('warming up…', 97);
  clearInterval(preview);
  frame = 0;
  await computeFrame(frame);        // the first run compiles kernels; keep it off the readout
  recent.length = 0;
  dirty = false;
  el('status').hidden = true;

  // The forward pass is in a worker now, so this loop never blocks rendering or
  // the orbit: the camera eases at display rate even mid-inference. Still hand
  // the thread back between poses (a promise chain alone only yields
  // microtasks), so the orbit and the buttons stay responsive.
  const yieldFrame = () => new Promise((r) => requestAnimationFrame(r));
  (async function inferenceLoop() {
    for (;;) {
      if (onScreen && dirty) {            // a new body: its rest pose first
        dirty = false;
        await computeFrame(frame);
        await yieldFrame();
      } else if (onScreen && playing) {
        frame = (frame + 1) % K;
        await computeFrame(frame);
        await yieldFrame();
      } else {
        await new Promise((r) => setTimeout(r, 40));
      }
    }
  })();

  /* ---- controls, enabled only now that they can act ---- */
  for (const b of document.querySelectorAll('[data-body]')) {
    b.addEventListener('click', () => select(b.dataset.body));
    b.disabled = false;
  }
  el('play').addEventListener('click', (e) => {
    playing = !playing;
    e.currentTarget.textContent = playing ? 'Pause' : 'Play';
  });
  el('play').disabled = false;
})().catch((e) => {
  el('status').hidden = false;
  el('statusText').textContent = `failed: ${e}`;
  console.error(e);
});
