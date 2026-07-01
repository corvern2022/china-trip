# 연태(옌타이) 여행 앱 — 작업 인수인계 문서

Claude Code로 넘어가서 이어서 작업하기 위한 요약입니다. 이 문서와 `yantai-day-planner.html`을 같은 폴더에 넣고 시작하세요.

## 한 줄 요약
연태(烟台) 4박5일 여행을 위한 **단일 HTML 파일** 웹앱. 검색 → ⭐저장 → 동선(일정) 담기 구조. Supabase로 2인 실시간 공유. 토스(Toss) 스타일 모바일 UI로 전면 재설계 완료.

## 지금 할 일 (이번 요청)
이 폴더를 GitHub 저장소로 만들고 GitHub Pages로 배포. 파일명은 `index.html`로 변경해서 올릴 것 (그래야 저장소 루트 주소로 바로 열림).

```
git init
mv yantai-day-planner.html index.html
git add index.html
git commit -m "연태 여행 앱 초기 배포"
gh repo create yantai-map --public --source=. --push
gh api repos/{owner}/yantai-map/pages -X POST -f source[branch]=main -f source[path]=/ 2>/dev/null || echo "Pages는 웹 UI(Settings→Pages)에서 수동 설정 필요할 수 있음"
```
(gh CLI 인증이 안 되어 있으면 `gh auth login`으로 브라우저 인증 먼저 진행)

배포 후 `https://{github아이디}.github.io/yantai-map/` 형태의 주소가 생김. **이 주소는 이후 안 바뀜** — Netlify Drop과 달리 파일을 다시 업로드해도 같은 주소 유지됨.

## 아키텍처 개요

**단일 HTML 파일** (`yantai-day-planner.html`, 약 590줄). 외부 의존성은 CDN만 사용:
- Leaflet 1.9.4 (지도)
- Pretendard 폰트
- 지도 타일: AutoNavi(高德) 기본 + GeoQ(智图) 대체 — 좌표계는 GCJ-02 (Google Places 좌표 그대로 사용 가능, WGS84 변환 불필요)

**화면 구조 (토스식, 하단 탭바 4개)**
- `#view_map` — 지도 전체화면. 검색바(상단 플로팅) + 장소 카드(하단 플로팅) + FAB(➕ 장소추가, 우하단)
- `#view_saved` — 저장한 장소 리스트 (⭐favorites 배열)
- `#view_route` — 일차별 동선. 상단 일차 칩(1~5, +날 추가 최대 12일), 순서 있는 정류장 리스트(▲▼ 재정렬, ✕ 삭제)
- `#view_browse` — 카테고리별 전체 장소 탐색, 한자 복사 기능

동선은 "지도에서 선 긋기" 방식이 **아니라** "장소를 일정에 담으면(➕ 일정에 담기 → 일차 선택 시트) 그 날 리스트에 순서대로 쌓이는" 방식. 지도에는 활성 일차(activeDay)의 경로가 폴리라인+번호마커로 자동 그려짐.

## 데이터
`places[]` (관광/맛집/카페/로컬시장/쇼핑/마사지, 약 68곳) + `hotels[]`(숙소, 16곳)이 HTML 안에 하드코딩되어 있음. 각 항목: `{id, cat, name, han(한자/영문), sub, lat, lng, note, alias?}`. 사용자가 앱에서 직접 추가한 장소는 `userPlaces` (localStorage)에 별도 저장되고 `rebuildDB()`가 합쳐서 `DB[]`/`byId{}`를 만듦.

기본 즐겨찾기 시드(`DEFAULT_FAVS`)가 최초 1회 자동 저장됨 (localStorage 플래그 `yt_seed`).

## 저장 방식
- 로컬: `localStorage` (`yt_favs`, `yt_routes`, `yt_userPlaces`, `yt_userCats`, `yt_numDays`, `yt_sync`, `yt_seed`)
- 공유: Supabase REST(PostgREST) 테이블 `yt_rooms(room text primary key, data jsonb, updated_at bigint)`. 앱이 4초 간격 폴링 + upsert. `SYNC={url, key, room}` 형태로 URL 쿼리파라미터(`?su=&sk=&room=`)에 인코딩해서 공유 링크로 전달 → 링크 받은 사람은 입력 없이 자동 연결.
- Firebase는 **의도적으로 배제**함 — 중국 본토에서 구글 인프라 접속이 불안정하기 때문. Supabase가 대안으로 선택됨.

## 사용자의 실제 Supabase 설정 (라이브)
- Project: `corvern2022's Project` (region: Seoul, ap-northeast-2)
- URL: `[REDACTED — 별도 비공개 채널로 전달]`
- 공유 코드(room): `[REDACTED — 별도 비공개 채널로 전달]`
- anon key는 **legacy `eyJ...` 형식**을 사용 중 (Supabase 새 형식 `sb_publishable_...`는 앱이 아직 지원 안 함 — 필요하면 헤더 방식은 동일하니 쉽게 확장 가능)
- SQL 셋업은 이미 실행 완료된 것으로 보임(공유 링크가 정상 작동한 것 확인됨)

## 이전 배포 이력
- Netlify Drop으로 배포했었음: `https://clinquant-faloodeh-57e7ea.netlify.app/`
- **문제**: Netlify Drop은 새로 드롭할 때마다 주소가 바뀜 → 공유 링크가 깨짐. 이게 GitHub Pages로 옮기려는 이유.
- GitHub Pages로 배포 완료 후, 앱의 "👥 공유" 시트에서 "🔗 링크 복사"를 다시 눌러 **새 주소 기준 공유 링크**를 재발급해서 상대방에게 다시 보내야 함. Supabase room 데이터 자체는 유지되므로 내용은 안 날아감.

## UI/UX 변경 이력 (중요도순)
1. 최초: Leaflet 지도 + 사이드패널 구조 (데스크톱 위주) — 모바일에서 매우 불편하다는 피드백
2. 상단 툴바를 동선 탭 안으로 이동, 바텀시트 시도 — 여전히 불편하다는 피드백 ("어디가 문제인지 말하기도 어려울 정도")
3. **전면 재설계**: 토스(Toss) 스타일 — 떠다니는 패널/시트 제거, 하단 탭바 4개로 전체화면 뷰 전환 방식. 지금 이 구조.
4. 추가 QA 2라운드 완료:
   - 선택한 핀이 하단 카드에 가려지는 문제 → 지도를 프로젝션 좌표로 오프셋(`pt.y+=120`)해서 핀이 카드 위쪽에 오도록 보정
   - 저장(⭐) 토글 시 지도가 불필요하게 다시 flyTo 되던 문제 → `paintCard()`로 카드 내용만 갱신하는 함수 분리, `showCard()`는 최초 선택시에만 지도 이동
   - 검색 결과 선택 시 키보드 자동 닫힘 처리 추가
   - 카드가 떠 있을 때 FAB(➕) 버튼 숨김 처리 (겹침 방지)
   - 검색어에 따옴표 포함 시 HTML 깨지는 버그 수정
   - "일정에 담기" 완료 시 동선 탭으로 자동 전환 + "N일차 M번째로 담았어요" 토스트로 확인 제공
   - 동선이 비어있을 때 "장소 둘러보러 가기" CTA 버튼 추가

## 알려진 미해결/향후 고려사항
- Supabase 새 형식 API 키(`sb_publishable_...`) 미지원 (legacy `eyJ...` anon key만 검증됨) — 필요시 헤더 방식이 거의 동일하므로 쉽게 추가 가능
- 실시간 동기화는 "동시 편집 시 나중에 저장한 쪽이 이김(last-write-wins)" 방식, 진짜 CRDT/충돌해결 아님 — 2인 사용 기준으로는 충분하다고 판단해서 그대로 둠
- 중국 본토에서 Supabase 접속 안정성은 100% 보장 아님 (Firebase보다는 낫다는 정도) — 백업으로 파일 export/import 기능이 공유 시트 안에 있음
- 화면 회전 시 지도 리사이즈 처리는 되어 있으나 실제 기기 테스트는 못 해본 상태
- 사용자 정의 카테고리(userCats) 기능은 코드에 있지만 UI에서 추가하는 진입점이 아직 없음 (장소 추가 시트의 카테고리 select는 기존 7종 + userCats 키만 나열, 새 카테고리를 만드는 버튼은 없음)

## 검증 방법 (수정할 때마다)
파일 안에 `<script>` 블록이 3개 있음. 수정 후 아래로 문법 체크:
```bash
node -e "
const fs=require('fs');const h=fs.readFileSync('index.html','utf8');
const blocks=[...h.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]);
blocks.forEach((b,i)=>{try{new Function(b);}catch(e){console.log('BLOCK',i,e.message);}});
console.log('checked', blocks.length, 'blocks');
"
```

## 사용자 커뮤니케이션 규칙 (중요)
- 항상 자연스러운 한국어, **em dash(—) 사용 금지**
- AI스러운 말투 지양, 담백하고 실용적인 톤
- 영어 메시지/문서를 작성할 일이 생기면 한국어 버전도 같이 제공
