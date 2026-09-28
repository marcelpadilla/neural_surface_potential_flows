# Neural Surface Potential for Potential Flows

<p align="center">
  <a href="https://marcelpadilla.com/Neural_Surface_Potential/">
    <img src="https://img.shields.io/badge/-Show%20Project%20Page-930000?style=for-the-badge" alt="Show Project Page">
  </a>
</p>

Supplementary material for the preprint **Neural Surface Potential for Potential Flows**. NSP is a feed-forward network that predicts the six Kirchhoff surface potentials of exterior potential flow from a closed triangle mesh, without per-mesh precomputation.

The network runs live in the browser on a body that bends:

```bash
python demo/serve.py        # http://127.0.0.1:8000/
```

`demo/assets/nsp_phix.onnx` is exported with dynamic shapes, so the network runs at any vertex count. It is the single-mode &phi;<sub>x</sub> model; the paper's six-mode network is assembled from it by the same pipeline. `demo/nsp.js` builds the input the graph expects — the canonical frame, vertex normals and farthest-point sampling — from a mesh you supply, if you want to run your own.

Marcel Padilla, ETH Zürich. Code is MIT; the meshes keep their own terms. 2026.
