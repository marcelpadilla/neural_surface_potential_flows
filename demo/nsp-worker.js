/*
 * nsp-worker.js — the network, off the main thread.
 *
 * The same pipeline as nsp.js (canonical-frame input, JS farthest-point
 * sampling, the ONNX graph), but in a worker so a forward pass never blocks the
 * page: the body and the camera keep moving at display rate while a pose is
 * computing. Before this, single-threaded WASM ran the pass on the main thread
 * and the orbit froze for its whole length; that is also why the embedded page
 * used to stall the site around it.
 *
 * A module worker: ORT is the same pinned CDN build the page used to load, and
 * nsp.js is imported unchanged. ORT's own `env.wasm.proxy` is not used — it
 * fails with the runtime loaded from a CDN and poisons the backend registry.
 */

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.webgpu.min.mjs';

// nsp.js reads ORT from the global scope (the page used to define it there).
globalThis.ort = ort;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
ort.env.wasm.numThreads = self.crossOriginIsolated
  ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1)) : 1;

import { NSPNet } from './nsp.js';

let net = null;

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'init') {
    try {
      net = await NSPNet.load(new Uint8Array(m.onnx), m.nPoints);
      postMessage({ type: 'ready', backend: net.backend, threads: ort.env.wasm.numThreads });
    } catch (err) {
      postMessage({ type: 'error', error: String((err && err.message) || err) });
    }
  } else if (m.type === 'predict') {
    try {
      const phi = await net.predict(m.positions, m.normals);
      postMessage({ type: 'result', id: m.id, phi }, [phi.buffer]);
    } catch (err) {
      postMessage({ type: 'error', id: m.id, error: String((err && err.message) || err) });
    }
  }
};
