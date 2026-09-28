/*
 * coi-serviceworker.js — cross-origin isolation on a host that will not send
 * headers (GitHub Pages).
 *
 * Multi-threaded WASM needs SharedArrayBuffer, SharedArrayBuffer needs the
 * document to be cross-origin isolated, and that needs COOP/COEP response
 * headers the host does not send. A service worker sits in front of every
 * request and adds them. The first visit registers the worker and reloads once,
 * before any of the ~8 MB payload is fetched, so the cost is one extra HTML
 * round trip.
 *
 * After the reload `self.crossOriginIsolated` is true and index.html asks
 * onnxruntime for real threads. Nothing else on the page depends on this: if
 * registration fails, or the visitor lands with `?nocoi=1`, the page runs
 * single-threaded exactly as before. WebGPU never needed it.
 *
 * Only a top-level page can be isolated this way. A frame is cross-origin
 * isolated only if the page embedding it is as well, so inside an iframe the
 * worker does not register: it could only cost a reload. The website embeds the
 * demo in its project page and therefore ships without this file.
 *
 * Same idea as Guido Zuidhof's coi-serviceworker (MIT), reduced to what this
 * page uses and given a reload guard.
 *
 * This one file is both halves: loaded as a classic script it registers, loaded
 * as a worker it serves.
 */
if (typeof window === 'undefined') {
  /* ------------------------------------------------------- worker half */
  self.addEventListener('install', () => self.skipWaiting());
  self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
  self.addEventListener('fetch', (event) => {
    const req = event.request;
    // Cache-probe requests must be passed through untouched or they throw.
    if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
    event.respondWith(fetch(req).then((res) => {
      if (res.status === 0) return res;            // opaque: no headers to set
      const h = new Headers(res.headers);
      h.set('Cross-Origin-Embedder-Policy', 'require-corp');
      h.set('Cross-Origin-Resource-Policy', 'cross-origin');
      h.set('Cross-Origin-Opener-Policy', 'same-origin');
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
    }));
  });
} else {
  /* --------------------------------------------------------- page half */
  (function register() {
    if (self.crossOriginIsolated) return;                 // already isolated
    if (window.top !== window) return;                    // a frame cannot be
    if (!self.isSecureContext || !('serviceWorker' in navigator)) return;
    if (new URL(location.href).searchParams.has('nocoi')) return;
    // A worker that installs but never isolates would otherwise reload forever.
    if (sessionStorage.getItem('coi-reloaded') === '1') return;
    const src = document.currentScript && document.currentScript.src;
    if (!src) return;
    const reload = () => {
      sessionStorage.setItem('coi-reloaded', '1');
      location.reload();
    };
    // Reload once the worker controls this page (clients.claim() on activate),
    // so the reloaded navigation is the one it serves with the headers.
    navigator.serviceWorker.addEventListener('controllerchange', reload, { once: true });
    navigator.serviceWorker.register(src).then((reg) => {
      if (reg.active && !navigator.serviceWorker.controller) reload();
    }, (e) => console.warn('[coi] not isolated, staying single-threaded:', e));
  })();
}
