// 오프라인 캐시는 껐다 (아이폰 사파리에서 페이지가 반복해서 멈추는 원인으로 의심, 2026-10-25).
// 예전에 설치된 기기에서도 스스로 지우고 손을 뗀다 — 요청은 가로채지 않는다.
self.addEventListener('install', () => self.skipWaiting())
self.addEventListener('activate', (e) => e.waitUntil((async () => {
  for (const k of await caches.keys()) await caches.delete(k)
  await self.registration.unregister()
})()))
