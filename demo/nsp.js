/*
 * nsp.js — the method, in the browser.
 *
 * One forward pass of the neural surface-potential network on one mesh:
 *   1. the input: the canonical frame the network was trained in and the
 *      vertex normals, computed exactly as the training pipeline does
 *   2. farthest point sampling at three levels (here, in JS)
 *   3. the ONNX graph (ball query, k-NN interpolation, the MLPs)
 *
 * FPS lives outside the graph on purpose. It is a sequential loop with no
 * weights; traced into ONNX it unrolls to tens of thousands of nodes and costs
 * about half the runtime. Exporting it out keeps the graph to a few hundred
 * nodes. The algorithm below
 * is models/pointnet2.py::_farthest_point_sample_loop with deterministic=True
 * (same seed, first maximum wins). It computes distances in double precision
 * where PyTorch uses single, so near-ties can break differently: on the 60
 * frames of the demo's first bunny two level-1 samples swapped places on 2
 * frames (same set, same deeper levels), and the field errors matched the
 * PyTorch reference to within 1e-5. experiments/webdemo_validate.py checks its
 * numpy mirror against PyTorch index for index.
 */

/**
 * The canonical frame, in place: area centroid at the origin, then a uniform
 * scale that puts the farthest vertex on the unit sphere
 * (utils/mesh_repair.normalize_to_unit_bounding_sphere).
 */
export function canonicalize(V, F) {
  let cx = 0, cy = 0, cz = 0, tot = 0;
  for (let t = 0; t < F.length; t += 3) {
    const a = 3 * F[t], b = 3 * F[t + 1], c = 3 * F[t + 2];
    const ux = V[b] - V[a], uy = V[b + 1] - V[a + 1], uz = V[b + 2] - V[a + 2];
    const vx = V[c] - V[a], vy = V[c + 1] - V[a + 1], vz = V[c + 2] - V[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const area = 0.5 * Math.sqrt(nx * nx + ny * ny + nz * nz);
    cx += area * (V[a] + V[b] + V[c]) / 3;
    cy += area * (V[a + 1] + V[b + 1] + V[c + 1]) / 3;
    cz += area * (V[a + 2] + V[b + 2] + V[c + 2]) / 3;
    tot += area;
  }
  cx /= tot; cy /= tot; cz /= tot;
  let r2 = 0;
  for (let i = 0; i < V.length; i += 3) {
    V[i] -= cx; V[i + 1] -= cy; V[i + 2] -= cz;
    r2 = Math.max(r2, V[i] * V[i] + V[i + 1] * V[i + 1] + V[i + 2] * V[i + 2]);
  }
  const s = 1 / Math.sqrt(r2);
  for (let i = 0; i < V.length; i++) V[i] *= s;
  return V;
}

/**
 * Unit vertex normals, each face normal weighted by the face's corner angle at
 * the vertex: trimesh's Trimesh.vertex_normals, which the training data used.
 */
export function vertexNormals(V, F, out) {
  out.fill(0);
  const corner = (i, j, k) => {      // angle at i in triangle (i, j, k)
    const ux = V[j] - V[i], uy = V[j + 1] - V[i + 1], uz = V[j + 2] - V[i + 2];
    const vx = V[k] - V[i], vy = V[k + 1] - V[i + 1], vz = V[k + 2] - V[i + 2];
    const d = (ux * vx + uy * vy + uz * vz)
      / Math.sqrt((ux * ux + uy * uy + uz * uz) * (vx * vx + vy * vy + vz * vz));
    return Math.acos(Math.min(1, Math.max(-1, d)));
  };
  for (let t = 0; t < F.length; t += 3) {
    const a = 3 * F[t], b = 3 * F[t + 1], c = 3 * F[t + 2];
    const ux = V[b] - V[a], uy = V[b + 1] - V[a + 1], uz = V[b + 2] - V[a + 2];
    const vx = V[c] - V[a], vy = V[c + 1] - V[a + 1], vz = V[c + 2] - V[a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (!(len > 0)) continue;
    nx /= len; ny /= len; nz /= len;
    for (const [p, w] of [[a, corner(a, b, c)], [b, corner(b, c, a)], [c, corner(c, a, b)]]) {
      out[p] += w * nx; out[p + 1] += w * ny; out[p + 2] += w * nz;
    }
  }
  for (let i = 0; i < out.length; i += 3) {
    const len = Math.hypot(out[i], out[i + 1], out[i + 2]) || 1;
    out[i] /= len; out[i + 1] /= len; out[i + 2] /= len;
  }
  return out;
}

/** Farthest point sampling. xyz is a flat Float32Array of 3*n coordinates. */
export function fpsIndices(xyz, n, npoint) {
  npoint = Math.min(npoint, n);
  // Seed = the vertex farthest from the vertex mean. A geometric, permutation
  // stable choice: a random seed would make the whole network stochastic
  // (~1% rel-L2 between two runs on the same input).
  let mx = 0, my = 0, mz = 0;
  for (let i = 0; i < n; i++) { mx += xyz[3 * i]; my += xyz[3 * i + 1]; mz += xyz[3 * i + 2]; }
  mx /= n; my /= n; mz /= n;
  let seed = 0, seedD = -1;
  for (let i = 0; i < n; i++) {
    const dx = xyz[3 * i] - mx, dy = xyz[3 * i + 1] - my, dz = xyz[3 * i + 2] - mz;
    const d = dx * dx + dy * dy + dz * dz;
    if (d > seedD) { seedD = d; seed = i; }
  }

  const dist = new Float64Array(n).fill(1e10);
  const out = new Int32Array(npoint);
  let far = seed;
  for (let s = 0; s < npoint; s++) {
    out[s] = far;
    const fx = xyz[3 * far], fy = xyz[3 * far + 1], fz = xyz[3 * far + 2];
    let best = -1, bestI = 0;
    for (let j = 0; j < n; j++) {           // fused min-update + argmax
      const dx = xyz[3 * j] - fx, dy = xyz[3 * j + 1] - fy, dz = xyz[3 * j + 2] - fz;
      const d = dx * dx + dy * dy + dz * dz;
      if (d < dist[j]) dist[j] = d;
      if (dist[j] > best) { best = dist[j]; bestI = j; }   // strict >: first max wins, as argmax does
    }
    far = bestI;
  }
  return out;
}

/** Gather rows of a flat (n, 3) array by index. */
function gather3(src, idx) {
  const out = new Float32Array(idx.length * 3);
  for (let i = 0; i < idx.length; i++) {
    const j = 3 * idx[i];
    out[3 * i] = src[j]; out[3 * i + 1] = src[j + 1]; out[3 * i + 2] = src[j + 2];
  }
  return out;
}

const toI64 = (a) => BigInt64Array.from(a, (v) => BigInt(v));

export class NSPNet {
  constructor(session, backend, nPoints) {
    this.session = session;
    this.backend = backend;
    this.n = nPoints;
    // The npoint policy of SurfacePotentialNet.forward: N/4, N/16, N/64.
    this.np1 = Math.max(3, Math.floor(nPoints / 4));
    this.np2 = Math.max(3, Math.floor(nPoints / 16));
    this.np3 = Math.max(3, Math.floor(nPoints / 64));
    this.feat = new Float32Array(nPoints * 6);
  }

  /**
   * Load the graph, preferring WebGPU and falling back to WASM.
   *
   * ``source`` is a URL or the model bytes; the page passes bytes, because it
   * downloads the graph itself to drive a progress bar. ORT logs at error level
   * only: its warning that the int64 index ops run on the CPU is expected here,
   * and it would reach the console as two red errors on every WebGPU load.
   *
   * Not attempted: ORT's worker proxy (``ort.env.wasm.proxy``), which would keep
   * the forward pass off the main thread. With the runtime loaded from a CDN it
   * fails with "worker not ready", and a failed attempt poisons the backend
   * registry so the plain WASM fallback fails too. Revisit only with the ORT
   * assets served same-origin.
   */
  static async load(source, nPoints) {
    let lastErr;
    for (const ep of navigator.gpu ? ['webgpu', 'wasm'] : ['wasm']) {
      try {
        const session = await ort.InferenceSession.create(source, {
          executionProviders: [ep], graphOptimizationLevel: 'all', logSeverityLevel: 3,
        });
        return new NSPNet(session, ep, nPoints);
      } catch (e) { lastErr = e; console.warn(`[nsp] ${ep} backend unavailable:`, e); }
    }
    throw lastErr;
  }

  /**
   * Predict phi_x on one mesh.
   * positions/normals: flat Float32Arrays of 3*n, in the canonical frame
   * (area centroid at the origin, unit bounding sphere) — the frame the
   * network was trained in.
   */
  async predict(positions, normals) {
    const n = this.n;
    const f1 = fpsIndices(positions, n, this.np1);
    const xyz1 = gather3(positions, f1);
    const f2 = fpsIndices(xyz1, this.np1, this.np2);
    const xyz2 = gather3(xyz1, f2);
    const f3 = fpsIndices(xyz2, this.np2, this.np3);

    const feat = this.feat;
    for (let i = 0; i < n; i++) {
      feat[6 * i] = positions[3 * i];
      feat[6 * i + 1] = positions[3 * i + 1];
      feat[6 * i + 2] = positions[3 * i + 2];
      feat[6 * i + 3] = normals[3 * i];
      feat[6 * i + 4] = normals[3 * i + 1];
      feat[6 * i + 5] = normals[3 * i + 2];
    }

    const out = await this.session.run({
      points: new ort.Tensor('float32', feat, [1, n, 6]),
      fps1: new ort.Tensor('int64', toI64(f1), [1, this.np1]),
      fps2: new ort.Tensor('int64', toI64(f2), [1, this.np2]),
      fps3: new ort.Tensor('int64', toI64(f3), [1, this.np3]),
    });
    return out.phi.data;
  }
}

/* The potential is defined up to a constant; the page shows it in the mean-zero
 * gauge of code/eval/metrics.py, so the colour scale is symmetric about zero. */
export function meanZero(a, out) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m += a[i];
  m /= a.length;
  for (let i = 0; i < a.length; i++) out[i] = a[i] - m;
  return out;
}
