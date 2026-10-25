// 수다목장 오프라인 캐시: 그림·코드는 한 번 받으면 기기에 저장 (화면 HTML은 항상 새로 받아 업데이트 반영)
const CACHE = 'sudafarm-v1'
self.addEventListener('install', () => self.skipWaiting())
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()))
self.addEventListener('fetch', (e) => {
  const req = e.request
  if (req.method !== 'GET' || new URL(req.url).origin !== location.origin) return
  if (req.mode === 'navigate') {
    e.respondWith(fetch(req).catch(() => caches.match(req)))
    return
  }
  e.respondWith(caches.open(CACHE).then(async (c) => {
    const hit = await c.match(req)
    if (hit) return hit
    const res = await fetch(req)
    if (res.ok) c.put(req, res.clone())
    return res
  }))
})
