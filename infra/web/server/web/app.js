// 본디 — 유물 자료. 화면 로직. 빌드 없음, API 는 /api/*, 정적 파일은 /files/* /preview/*.
import { createViewer } from './viewer.js';

const $ = (s) => document.querySelector(s);
const state = { media: '3d', q: '', source: '', variant: '', restore: '', offset: 0, limit: 60, total: 0, card: null, viewer: null, viewer2: null };
// media 가 'fav' 이면 2D·3D 가리지 않고 즐겨찾기한 카드만 보인다 (팀 공용, 서버 favorites 표)
// 즐겨찾기는 VR 로 볼 수 있는 유물이 맨 앞에 온다 (시연 흐름: 즐겨찾기로 보여준 뒤 VR 로 넘어간다)

// 복원 종류 — 코드 → 짧은 표시. 배지가 길면 카드가 두 줄로 밀린다
const RT_KO = { shape: '형태', color: '색', img2mesh: '2D→3D', symmetry: '회전대칭', pointr: '점군 완성' };

// ───────── 즐겨찾기 ─────────
async function toggleStar(assetId, on) {
  const r = await api('/api/favorites/' + assetId.split('/').map(encodeURIComponent).join('/'), { method: on ? 'PUT' : 'DELETE' });
  $('#nfav').textContent = r.total;
  return r.starred === 1;
}
function starBtn(c) {
  const b = document.createElement('button');
  b.className = 'star' + (c.starred ? ' on' : ''); b.textContent = c.starred ? '★' : '☆'; b.title = c.starred ? '즐겨찾기 해제' : '즐겨찾기 (팀 공용)';
  b.onclick = async (e) => {
    e.stopPropagation();                                   // 카드 클릭(상세 열기)으로 번지지 않게
    const on = await toggleStar(c.asset_id, !c.starred); c.starred = on ? 1 : 0;
    b.classList.toggle('on', on); b.textContent = on ? '★' : '☆';
    if (state.media === 'fav' && !on) b.closest('.card')?.remove();   // 즐겨찾기 모드에서 해제하면 목록에서 바로 빠진다
  };
  return b;
}
const fmtMB = (b) => b >= 1e9 ? (b / 1e9).toFixed(2) + ' GB' : b >= 1e6 ? (b / 1e6).toFixed(1) + ' MB' : Math.max(1, Math.round(b / 1e3)) + ' KB';  // MTL 처럼 작은 파일이 '0.0 MB' 로 보이지 않게
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const api = async (p, opt) => { const r = await fetch(p, opt); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.status); return r.json(); };

// ───────── 목록 ─────────
function badge(c) {
  const b = [`<span class="b media${c.media_type}">${c.media_type.toUpperCase()}</span>`, `<span class="b ${c.variant === 'source' ? '' : 'restored'}">${esc(c.variant_ko)}</span>`];
  // 복원본은 '무엇을 복원했는지' 를 따로 붙인다 — '복원본(팀 AI)' 만으로는 알 수 없다는 피드백
  (c.restore_types || []).forEach((t) => b.push(`<span class="b rt ${t}" title="복원 종류">${esc(RT_KO[t] || t)}</span>`));
  if (c.vr) b.push(`<span class="b vr" title="VR 에서 볼 수 있는 유물">VR</span>`);
  if (c.license_status !== 'ok') b.push(`<span class="b ${c.license_status}">${c.license_status === 'restricted' ? '이용 제한' : '라이선스 미확인'}</span>`);
  if (c.designation) b.push(`<span class="b">${esc(c.designation.replace('National Treasure', '국보 ').replace(/^Treasure/, '보물 '))}</span>`);
  return b.join('');
}
function cardEl(c) {
  const el = document.createElement('article'); el.className = 'card' + (c.vr ? ' isvr' : ''); el.dataset.id = c.asset_id;
  // 회선이 약한 곳(굴)에서도 목록이 빨리 뜨게:
  //   · loading=lazy  — 화면에 들어올 때만 받는다
  //   · decoding=async — 디코딩이 스크롤을 붙잡지 않게
  //   · 자리 확보(.thumb 가 1:1) + 받는 동안 옅은 자리표시 — 이미지가 늦어도 레이아웃이 튀지 않는다
  //   · onload 로 자리표시를 걷는다
  const th = c.thumb_url
    ? `<img loading="lazy" decoding="async" src="${c.thumb_url}" alt="" onload="this.parentNode.classList.add('ready')">`
    : `<span class="ph">${c.media_type === '3d' ? '◈' : '▣'}</span>`;
  el.innerHTML = `<div class="thumb ${c.media_type === '2d' ? 'photo' : ''}${c.thumb_url ? ' load' : ''}">${th}</div>
    <div class="cbody"><div class="ctitle">${esc(c.name_ko)}</div>
    <div class="csub">${esc([c.period, c.material, c.museum].filter(Boolean).join(' · '))}</div>
    <div class="badges">${badge(c)}<span class="b">${c.file_count}개 · ${fmtMB(c.total_bytes)}</span></div></div>`;
  el.prepend(starBtn(c));
  el.onclick = () => { location.hash = '#card/' + encodeURIComponent(c.asset_id); };
  return el;
}
// 탭을 누르면 핸들러가 한 번, 이어서 hashchange 의 route() 가 또 한 번 불렀다.
// 둘 다 비운 뒤 각자 응답을 붙이는 바람에 같은 카드가 두 벌씩 쌓였다(즐겨찾기 2건인데 4장).
// 순번을 매겨 마지막 요청만 그린다. 비우는 것도 응답이 온 뒤로 미뤄 화면이 깜빡이지 않게 한다.
// 한 줄에 몇 칸 들어가는지 — 60장씩 받으면 마지막 줄이 4칸만 차서 오른쪽이 휑했다.
// 칸 수의 배수로 받아 줄이 늘 꽉 차게 한다.
function gridCols() {
  const t = getComputedStyle($('#grid')).gridTemplateColumns.split(' ').filter(Boolean).length;
  return Math.max(1, t);
}
let listSeq = 0;
async function loadList(append = false) {
  const seq = ++listSeq;
  const cols = gridCols();
  state.limit = cols * Math.max(2, Math.round(60 / cols));   // 60 언저리를 칸 수의 배수로
  if (!append) state.offset = 0;
  const p = new URLSearchParams({ limit: state.limit, offset: state.offset });
  if (state.media === 'fav') { p.set('starred', 'true'); p.set('sort', 'title'); }  // 즐겨찾기: 이름순 — 찾던 것을 이름으로 짚는다
  else p.set('media', state.media);
  if (state.q) p.set('q', state.q); if (state.source) p.set('source_org', state.source); if (state.variant) p.set('variant', state.variant);
  if (state.restore) p.set('restore_type', state.restore);
  const r = await api('/api/cards?' + p);
  if (seq !== listSeq) return;                        // 더 새 요청이 있다 — 이 응답은 버린다
  const grid = $('#grid');
  if (!append) grid.replaceChildren();
  grid.querySelector('.moreslot')?.remove();          // 앞서 끼워 둔 '더 보기' 는 새 카드 뒤로 다시 옮긴다
  state.total = r.total;
  const vrN = r.items.filter((c) => c.vr).length;
  // 즐겨찾기는 이름순이라 VR 이 앞에 모이지 않는다 — 구분선은 두지 않고 개수만 알린다
  r.items.forEach((c) => grid.appendChild(cardEl(c)));
  state.offset += r.items.length;
  const label = state.media === 'fav' ? '★ 즐겨찾기' : state.media.toUpperCase() + ' 자료';
  $('#listCount').textContent = `${label} ${r.total.toLocaleString()}건` + (state.q ? ` — '${state.q}' 검색` : '')
    + (state.media === 'fav' ? ' · 이름순' + (vrN ? ` · VR ${vrN}건` : '') : '');
  if (state.media === 'fav' && r.total === 0 && !append) grid.innerHTML = '<div class="fav-empty">아직 즐겨찾기한 자료가 없습니다. 카드의 ☆ 를 누르면 팀 전체에 ★ 로 표시됩니다.</div>';
  placeMore(grid);
}

// '더 보기' 는 목록 맨 끝이 아니라 **두 줄 남기고** 그 자리에 둔다 (피드백 2026-09-17).
// 그리드 안에 한 칸을 통째로 차지하는 줄로 끼워 넣어, 스크롤을 끝까지 내리지 않아도 눌린다.
// 남은 카드가 없으면 감춘다. 아래 #more 는 화면이 좁아 두 줄이 안 남을 때의 대비책으로 남겨 둔다.
function placeMore(grid) {
  grid.querySelector('.moreslot')?.remove();
  const rest = state.total - state.offset;
  $('#more').hidden = true;
  if (rest <= 0) return;
  const cards = [...grid.querySelectorAll('.card')];
  if (!cards.length) return;
  const slot = document.createElement('div');
  slot.className = 'moreslot';
  slot.innerHTML = `<button class="more">더 보기 <small>남은 ${rest.toLocaleString()}건</small></button>`;
  // 누른 뒤 **새로 불러온 첫 카드**를 화면 위로 올린다.
  // 안 하면 스크롤이 그대로라 방금 부른 60장이 화면 아래에 쌓이고, 버튼은 새 목록의 두 줄 위 —
  // 즉 4,000px 쯤 아래로 사라진다 (1600×950 · 6열에서 실측). 두 번째 클릭부터는
  // "끝까지 내려가서 누르지 않게" 라는 애초 취지가 반대로 뒤집힌다.
  slot.querySelector('button').addEventListener('click', async (e) => {
    const shown = state.offset;                       // 지금까지 그려진 카드 수 = 새로 올 첫 카드의 번호
    e.currentTarget.disabled = true;                  // 연타로 한 쪽 응답이 버려지는 것도 막는다
    await loadList(true);
    const first = grid.querySelectorAll('.card')[shown];
    if (first) scrollTo({ top: first.getBoundingClientRect().top + scrollY - 12, behavior: 'smooth' });
  });
  // 줄은 **실제로 놓인 위치**로 센다. gridTemplateColumns 를 읽어 열 수를 구하면 목록을 막 채운 직후에는
  // 아직 한 줄짜리로 잡혀(=cols 1) '두 줄 위' 가 '두 칸 위' 가 됐다.
  const tops = [...new Set(cards.map((c) => c.offsetTop))].sort((a, b) => a - b);
  const targetTop = tops[Math.max(0, tops.length - 2)];        // 마지막에서 두 번째 줄
  const before = cards.find((c) => c.offsetTop === targetTop);
  if (before && tops.length > 2) grid.insertBefore(slot, before);
  else grid.appendChild(slot);                                  // 두 줄 이하면 그냥 끝에
}
async function loadStats() {
  const s = await api('/api/stats');
  $('#n3d').textContent = s.by_media['3d']?.cards ?? 0; $('#n2d').textContent = s.by_media['2d']?.cards ?? 0;
  $('#nfav').textContent = s.favorites ?? 0;
  $('#mediaNav').querySelector('.fav').title =
    `팀이 즐겨찾기한 것만 (2D·3D 모두) · 이름순 · VR 로 볼 수 있는 유물 ${s.vr_starred ?? 0}건`;
  $('#fbN').textContent = s.feedback_open || '';            // 안 끝난 피드백이 있으면 버튼에 숫자
  const tb = Object.values(s.by_media).reduce((a, m) => a + (m.bytes || 0), 0);
  $('#stats').textContent = `유물 ${s.artifacts}건 · 카드 ${Object.values(s.by_media).reduce((a, m) => a + m.cards, 0)}장 · ${fmtMB(tb)}` + (s.last_scan ? ` · 마지막 스캔 ${fbDay(s.last_scan.finished_at)} ${fbTime(s.last_scan.finished_at)}` : '');   // 서버는 UTC — 보는 사람 시각으로
}

// ───────── 상세 ─────────
function disposeViewers() { state.viewer?.dispose(); state.viewer2?.dispose(); state.viewer = state.viewer2 = null; $('#pv2').hidden = true; }
let pvSeq = 0;                                  // 늦게 도착한 옛 카드의 로드 결과를 버리기 위한 번호표
async function showPreview(file, card) {
  const my = ++pvSeq;
  const pv = $('#pv'); pv.replaceChildren();
  // replaceChildren 로 캔버스를 방금 지웠다. 뷰어를 그대로 두면 화면에 없는 캔버스에 계속 그린다
  // — 파일 목록에서 '미리보기' 를 누를 때 앞 카드의 그림이 남아 보이던 원인.
  if (state.viewer) { state.viewer.dispose(); state.viewer = null; }
  if (state.viewer2) { state.viewer2.dispose(); state.viewer2 = null; }
  $('#pv2').hidden = true;
  if (card.media_type === '2d' || (file && file.kind === 'photo') || (file && file.kind === 'image')) {
    const img = document.createElement('img'); img.src = file ? file.url : card.files.find((f) => f.previewable)?.url; pv.appendChild(img);
    $('#pvLabel').textContent = file ? file.label_ko : '사진'; $('#pvHint').textContent = file ? `${file.width}×${file.height} · ${fmtMB(file.size)} · 클릭한 파일을 그대로 표시` : '';
    return;
  }
  const url = card.preview_url;
  if (!url) {                                  // 아직 GLB 가 없다 — 서버에서 만들 수 있으니 버튼을 준다
    pv.innerHTML = '<span class="msg">3D 미리보기가 아직 없습니다<br><button id="pvMake" class="ghost">지금 만들기</button></span>';
    $('#pvMake')?.addEventListener('click', async () => {
      try { await api('/api/preview/rebuild?asset=' + encodeURIComponent(card.asset_id), { method: 'POST' }); waitPreview(card.asset_id); }
      catch (e) { pv.innerHTML = `<span class="msg">${esc(e.message)}</span>`; }
    });
    return;
  }
  if (!state.viewer) state.viewer = createViewer(pv);
  const prog = addProgressUI(pv);
  try {
    const info = await state.viewer.load(url, prog.on);
    if (my !== pvSeq) { prog.done(); return; } // 그 사이 다른 카드로 옮겨 갔다
    prog.done();
    addBrightUI(pv, state.viewer);
    $('#pvLabel').textContent = `${card.variant_ko} — 경량 미리보기`;
    $('#pvHint').textContent = (info.points ? `점 ${info.points.toLocaleString()}개` : `삼각형 ${info.triangles.toLocaleString()}개`) + ' · 드래그 회전, 휠 확대, 우클릭 이동 · 원본은 아래 파일에서 다운로드';
  } catch (e) { prog.done(); pv.innerHTML = `<span class="msg">미리보기 로드 실패: ${esc(e.message || e)}</span>`; }
}
// ───────── 3D 로딩 막대 ─────────
// GLB 는 6~30MB 라 회선이 느리면 몇 초씩 걸린다. 글자만 있으면 멈춘 것처럼 보인다.
function addProgressUI(pane) {
  pane.querySelector('.loadbar')?.remove();
  const el = document.createElement('div');
  el.className = 'loadbar';
  el.innerHTML = '<div class="lb-track"><i></i></div><span>불러오는 중…</span>';
  pane.appendChild(el);
  const bar = el.querySelector('i'), txt = el.querySelector('span');
  let t0 = performance.now();
  return {
    on(loaded, total) {
      if (total) {
        const pct = Math.min(100, Math.round((loaded / total) * 100));
        bar.style.width = pct + '%';
        txt.textContent = `불러오는 중… ${pct}% · ${fmtMB(loaded)} / ${fmtMB(total)}`;
      } else {                                  // 총량을 모를 때(압축 전송 등)는 받은 양만
        el.classList.add('indet');
        txt.textContent = `불러오는 중… ${fmtMB(loaded)}`;
      }
    },
    done() {
      const ms = Math.round(performance.now() - t0);
      el.remove();
      return ms;
    },
  };
}

// ───────── 즐겨찾기 선행 로딩 ─────────
// 즐겨찾기는 시연에서 반드시 여는 것들이다. 홈에서 미리 받아 두면 누를 때 바로 뜬다
// (/preview/ 는 7일 캐시라 한 번 받아 두면 다음부터 네트워크를 안 탄다).
const PREFETCH_MAX_FILES = 12, PREFETCH_MAX_BYTES = 150 * 1024 * 1024;
let prefetched = false;
async function prefetchFavorites() {
  if (prefetched) return;
  prefetched = true;
  const c = navigator.connection;
  if (c && (c.saveData || /^(slow-)?2g$/.test(c.effectiveType || ''))) return;   // 데이터 절약·느린 회선이면 건너뛴다
  let items;
  try { items = (await api('/api/cards?starred=true&sort=title&limit=200')).items || []; }
  catch { return; }
  const urls = items.filter((x) => x.media_type === '3d' && x.preview_url).map((x) => x.preview_url);
  let bytes = 0, n = 0;
  for (const u of urls.slice(0, PREFETCH_MAX_FILES)) {
    if (state.card) break;                     // 사용자가 뭔가 열었으면 대역폭을 양보한다
    try {
      const r = await fetch(u, { cache: 'force-cache', priority: 'low' });
      const b = await r.blob();                // 캐시에 들어가려면 본문을 끝까지 읽어야 한다
      bytes += b.size; n += 1;
      if (bytes > PREFETCH_MAX_BYTES) break;
    } catch { /* 하나 실패해도 계속 */ }
  }
  if (n) console.info(`즐겨찾기 3D ${n}개 미리 받음 (${fmtMB(bytes)})`);
}

// ───────── 밝기 조절 ─────────
// 유물마다 원래 색이 크게 달라서(청동기는 거의 검다) 고정값 하나로는 부족하다.
// 화면마다 슬라이더를 하나씩 얹는다 — 비교할 때 원본과 복원본을 따로 올릴 수 있어야 한다.
const BRIGHT_KEY = 'c201.bright';
const BRIGHT_MIN = 0.4, BRIGHT_MAX = 5;      // 상한 2.4 로는 금관처럼 어두운 유물이 안 보였다 (평균 RGB 36,24,12)
const brightDefault = () => { const v = parseFloat(localStorage.getItem(BRIGHT_KEY)); return v >= BRIGHT_MIN && v <= BRIGHT_MAX ? v : 1; };
function addBrightUI(pane, viewer) {
  pane.querySelector('.bright')?.remove();
  const k0 = brightDefault();
  viewer.setBrightness(k0);
  const el = document.createElement('label');
  el.className = 'bright';
  el.title = '어두운 유물을 들여다볼 때 올려 쓰세요. 1.0 이 목록 썸네일과 같은 밝기입니다 (최대 ' + BRIGHT_MAX.toFixed(1) + '배)';
  el.innerHTML = `밝기 <input type="range" min="${BRIGHT_MIN}" max="${BRIGHT_MAX}" step="0.1" value="${k0}"><b>${k0.toFixed(1)}</b><button class="rst" type="button" title="1.0 으로">⟲</button>`;
  const range = el.querySelector('input'), num = el.querySelector('b');
  const put = (k) => { range.value = k; num.textContent = (+k).toFixed(1); viewer.setBrightness(+k); localStorage.setItem(BRIGHT_KEY, k); };
  // 휠로도 조절 — 슬라이더 폭이 84px 인데 범위가 넓어져 손으로 끌면 0.1 단위를 집기 어렵다 (방향키는 range 가 기본 지원)
  el.addEventListener('wheel', (e) => { e.preventDefault();
    put(Math.min(BRIGHT_MAX, Math.max(BRIGHT_MIN, Math.round((+range.value + (e.deltaY < 0 ? 0.1 : -0.1)) * 10) / 10))); },
    { passive: false });
  range.addEventListener('input', () => put(range.value));
  el.querySelector('.rst').addEventListener('click', (e) => { e.preventDefault(); put(1); });
  el.addEventListener('click', (e) => e.preventDefault());     // label 클릭이 슬라이더를 건드리지 않게
  pane.appendChild(el);
}

async function compare(card) {
  const cands = card.siblings.filter((s) => s.media_type === '3d' && s.preview_url);
  // 원본 카드에서는 복원본을, 복원본 카드에서는 원본을 상대로 고른다
  const other = (card.variant === 'source'
    ? cands.find((s) => s.variant !== 'source')
    : cands.find((s) => s.variant === 'source')) || cands[0];
  if (!other) return;
  // 어느 카드에서 눌렀든 **원본이 늘 왼쪽**이다. 복원본 카드에서 눌렀으면 좌우를 바꿔 싣는다
  // (안 그러면 복원본에서 누를 때만 좌우가 뒤집혀 비교가 헷갈린다)
  const swap = card.variant !== 'source' && other.variant === 'source';
  const left = swap ? other : card, right = swap ? card : other;
  const pv = $('#pv'), pv2 = $('#pv2');
  pv2.hidden = false;                       // .previews 가 flex 라 보이는 순간 좌우 반씩 나눠 갖는다 (ResizeObserver 가 캔버스를 맞춤)
  if (!state.viewer2) state.viewer2 = createViewer(pv2);
  if (swap) await state.viewer.load(left.preview_url);   // 왼쪽에 이미 복원본이 떠 있으니 원본으로 갈아 싣는다
  await state.viewer2.load(right.preview_url);
  pv.querySelector('.lbl')?.remove(); pv2.querySelector('.lbl')?.remove();
  pv.insertAdjacentHTML('beforeend', `<span class="lbl">${esc(left.variant_ko)}</span>`);
  pv2.insertAdjacentHTML('beforeend', `<span class="lbl">${esc(right.variant_ko)}</span>`);
  addBrightUI(pv, state.viewer); addBrightUI(pv2, state.viewer2);   // 좌우 각각 독립
  // 카메라 동기화 (AS-AT-05)
  state.viewer.onChange(() => state.viewer2.setPose(state.viewer.getPose()));
  state.viewer2.onChange(() => state.viewer.setPose(state.viewer2.getPose()));
  state.viewer2.setPose(state.viewer.getPose());
  $('#pvHint').textContent = '두 화면의 카메라가 함께 움직입니다. 왼쪽 ' + left.variant_ko + ' · 오른쪽 ' + right.variant_ko;
}
async function openCard(id) {
  disposeViewers();
  const c = await api('/api/cards/' + id.split('/').map(encodeURIComponent).join('/'));
  state.card = c;
  $('#list').hidden = true; $('#detail').hidden = false; window.scrollTo(0, 0);
  $('#dTitle').textContent = c.title_ko; $('#dBadges').innerHTML = badge(c);
  // 별 하나만 떠 있으면 누르는 것인지 모른다 — 글자를 붙인다
  const ds = $('#dStar'), starLabel = (on) => (on ? '★' : '☆') + ' 즐겨찾기';
  ds.classList.toggle('on', !!c.starred); ds.textContent = starLabel(!!c.starred);
  ds.onclick = async () => { const on = await toggleStar(c.asset_id, !c.starred); c.starred = on ? 1 : 0; ds.classList.toggle('on', on); ds.textContent = starLabel(on); };
  $('#fileCount').textContent = `${c.file_count}개 · ${fmtMB(c.total_bytes)}`;
  $('#tabDownN').textContent = c.file_count;
  showPane('info');                                              // 카드를 열면 늘 '유물 정보' 부터
  // 폴더 단위 ZIP — OBJ 세트처럼 여러 파일이 함께 있어야 쓰는 것들
  const zl = $('#zips'); zl.replaceChildren();
  try {
    const g = await api('/api/groups?asset=' + encodeURIComponent(c.asset_id));
    const order = ['digital_obj', 'scan_ply', 'print_stl', 'pointcloud_las', 'photo_2d', 'records', 'all'];
    g.groups.sort((a, b) => order.indexOf(a.group) - order.indexOf(b.group)).forEach((x) => {
      if (x.group === 'all' && g.groups.length === 2) return;            // 폴더가 하나면 '전체' 는 중복
      const li = document.createElement('li');
      li.innerHTML = `<a href="${x.zip_url}" class="${x.group === 'digital_obj' ? 'primary' : ''}"><span class="zl">${esc(x.label_ko)}${x.kinds.length > 1 ? `<small>${x.kinds.map(k => k.toUpperCase()).join(' · ')}</small>` : ''}</span><span class="zs">${x.count}개 · ${fmtMB(x.bytes)}</span><span class="zb">ZIP 받기</span></a>`;
      zl.appendChild(li);
    });
  } catch (e) { zl.innerHTML = `<li class="hint">폴더 ZIP 정보를 못 읽었습니다: ${esc(e.message)}</li>`; }
  const ul = $('#files'); ul.replaceChildren();
  c.files.forEach((f, i) => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="lab"><b>${esc(f.label_ko)}</b><small title="${esc(f.filename)}">${esc(f.filename)}</small></span><span class="sz">${fmtMB(f.size)}</span>` +
      (f.previewable ? `<button class="ghost pvb">미리보기</button>` : '') + `<a class="dl" href="${f.url}" download>다운로드</a>`;
    li.querySelector('.pvb')?.addEventListener('click', () => { ul.querySelectorAll('li').forEach((x) => x.classList.remove('on')); li.classList.add('on'); showPreview(f, c); });
    ul.appendChild(li);
  });
  const a = c.artifact, dl = $('#dInfo'); dl.replaceChildren();
  const rows = [['한글명', a.name_ko], ['다른 명칭', a.alt_name], ['영문명', a.name_en], ['지정', a.designation], ['시대', a.period], ['재질', a.material], ['분류', a.category], ['크기', a.size], ['출토지', a.excavation_site], ['소장품번호', a.accession_ko || a.accession], ['소장처', a.museum], ['이용조건', a.license_note || a.license_status]];
  rows.filter(([, v]) => v).forEach(([k, v]) => dl.insertAdjacentHTML('beforeend', `<dt>${k}</dt><dd>${esc(v)}</dd>`));
  if (a.detail_url) dl.insertAdjacentHTML('beforeend', `<dt>원 출처</dt><dd><a href="${a.detail_url}" target="_blank" rel="noopener">상세 페이지 ↗</a></dd>`);
  $('#dDesc').innerHTML = esc(a.description_ko) + `<small>설명 출처: ${a.description_source === 'official' ? '국립중앙박물관' : a.description_source === 'emuseum_official' ? 'e뮤지엄' : a.description_source.startsWith('derived') ? '팀 작성 (공식 설명 없음)' : a.description_source}</small>`;
  const sb = $('#siblings'); sb.replaceChildren();
  c.siblings.forEach((s) => sb.insertAdjacentHTML('beforeend', `<li><a href="#card/${encodeURIComponent(s.asset_id)}">${s.media_type.toUpperCase()} · ${esc(s.variant_ko)}${s.media_type === '2d' ? ` (${s.file_count}장)` : ''}</a></li>`));
  $('#sibBar').hidden = !c.siblings.length;                       // 없으면 줄째로 감춘다
  $('#cmpBtn').hidden = !(c.media_type === '3d' && c.siblings.some((s) => s.media_type === '3d' && s.preview_url));
  // 복원본 등록은 "원본 3D 카드"에서만 보인다. 이미 복원본이 있으면 덮어쓰기 안내로 바뀐다
  const hasRestored = c.siblings.some((s) => s.media_type === '3d' && s.variant.startsWith('restored'));
  $('#restoredBox').hidden = !(c.media_type === '3d' && c.variant === 'source' && ['museum', 'team'].includes(c.source_org));
  $('#upOverwrite').closest('label').hidden = !hasRestored;
  $('#tabEdit').textContent = $('#restoredBox').hidden ? '메모' : '복원본 등록 · 메모';
  $('#upHint').textContent = hasRestored
    ? '이미 복원본이 있습니다 — 바꾸려면 덮어쓰기를 체크하세요'
    : '파일만 올리면 유물 정보는 원본에서 가져옵니다';
  upState.files = []; renderPicked();
  $('#upLabel').value = ''; $('#upMethod').value = ''; $('#upOwner').value = ''; $('#upNote').value = ''; $('#upOverwrite').checked = false;
  // 복원본 카드에서는 복원본만, 원본 카드에서는 유물 통째로 지운다
  const delAll = !c.variant.startsWith('restored');
  $('#delWhat').textContent = delAll
    ? `유물 '${a.name_ko}' 의 카드 ${1 + c.siblings.length}장과 파일 전부가 목록에서 사라집니다.`
    : `이 복원본만 사라집니다. 원본과 사진은 그대로 남습니다.`;
  $('#delBtn').textContent = delAll ? '이 유물 삭제' : '이 복원본만 삭제';
  $('#delBtn').onclick = async () => {
    const what = delAll ? 'all' : 'restored';
    const only = delAll ? '' : `&variant=${encodeURIComponent(c.variant)}`;   // 여러 개 중 이것만
    const ask = (delAll ? '유물 전체' : '복원본') + '를 휴지통으로 옮깁니다.\n되돌리려면 서버에서 폴더를 도로 옮겨야 합니다.\n\n확인하려면 유물 ID 를 그대로 적어 주세요:\n' + c.artifact_id;
    if (prompt(ask) !== c.artifact_id) {
      $('#delMsg').textContent = '취소했습니다'; return;
    }
    $('#delBtn').disabled = true; $('#delMsg').textContent = '옮기는 중…';
    try {
      const r = await api(`/api/artifacts/${encodeURIComponent(c.artifact_id)}?confirm=${encodeURIComponent(c.artifact_id)}&what=${what}${only}`, { method: 'DELETE' });
      await loadStats();
      alert('휴지통으로 옮겼습니다.\n\n' + r.moved.join('\n') + '\n→ ' + r.trash);
      location.hash = delAll ? '' : '#card/' + encodeURIComponent(c.artifact_id + '/3d/source');
      if (!delAll) openCard(c.artifact_id + '/3d/source');
      else loadList();
    } catch (e) { $('#delMsg').textContent = '실패: ' + e.message; }
    finally { $('#delBtn').disabled = false; }
  };
  $('#delMsg').textContent = '';
  $('#eTags').value = a.tags || ''; $('#eNote').value = a.note || ''; $('#eMethod').value = c.method || ''; $('#eOwner').value = c.owner || ''; $('#saveMsg').textContent = '';
  // 복원 종류 — 복원본 3D 카드에서만 고칠 수 있다 (원본은 복원한 것이 없다)
  $('#rtBox').hidden = !(c.media_type === '3d' && c.variant !== 'source');
  const have = new Set(c.restore_types || []);
  $('#rtPick').querySelectorAll('input').forEach((x) => { x.checked = have.has(x.value); });
  $('#rtMsg').textContent = '';
  await showPreview(c.media_type === '2d' ? c.files.find((f) => f.previewable) : null, c);
}

// ───────── 오른쪽 패널 탭 ─────────
function showPane(name) {
  $('#infoTabs').querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.pane === name));
  document.querySelectorAll('.tabpane').forEach((p) => { p.hidden = p.dataset.pane !== name; });
  $('.info').scrollTop = 0;
}
$('#infoTabs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) showPane(b.dataset.pane); });

// ───────── 복원본 등록 ─────────
// 사용자는 파일만 고른다. 유물명·시대·재질·출처는 서버가 원본 _meta.json 에서 상속한다.
const UP_OK = ['.obj', '.mtl', '.ply', '.stl', '.glb', '.jpg', '.jpeg', '.png', '.json', '.txt', '.md', '.npy', '.las'];
const upState = { files: [] };
const extOf = (n) => { const i = n.lastIndexOf('.'); return i < 0 ? '' : n.slice(i).toLowerCase(); };
// 서버의 _dest_dir 과 같은 규칙 — 올리기 전에 어디로 갈지 미리 보여준다
function destOf(name) {
  const e = extOf(name), l = name.toLowerCase();
  if (e === '.obj' || e === '.mtl' || e === '.glb') return 'digital_obj';
  if (e === '.ply') return 'scan_ply';
  if (e === '.stl') return 'print_stl';
  if (e === '.las') return 'pointcloud_las';
  if (['.json', '.txt', '.md', '.npy'].includes(e)) return 'records';
  if (['.jpg', '.jpeg', '.png'].includes(e)) return /compare|damage|overlay|report|diag/.test(l) ? 'records' : 'digital_obj';
  return 'records';
}
const GLTF_WARN = (n) => n.toLowerCase().endsWith('.gltf');   // 분리형 glTF 는 .bin·텍스처 참조가 깨진다
function renderPicked() {
  const ul = $('#pickedList'); ul.replaceChildren();
  let ok = 0;
  upState.files.forEach((f) => {
    const good = UP_OK.includes(extOf(f.name));
    if (good) ok++;
    const li = document.createElement('li');
    if (!good) li.className = 'bad';
    li.innerHTML = `<span class="n">${esc(f.name)}</span><span class="d">${good ? destOf(f.name) + ' · ' + fmtMB(f.size) : (GLTF_WARN(f.name) ? 'GLB 로 내보내 주세요' : '지원하지 않는 형식')}</span>` +
      `<button class="x" title="빼기">×</button>`;
    const idx = upState.files.indexOf(f);
    li.querySelector('.x').addEventListener('click', () => { upState.files.splice(idx, 1); renderPicked(); });
    ul.appendChild(li);
  });
  const total = upState.files.filter((f) => UP_OK.includes(extOf(f.name))).reduce((a, f) => a + f.size, 0);
  $('#upBtn').disabled = ok === 0;
  $('#upMsg').textContent = ok ? `${ok}개 · ${fmtMB(total)} 등록 준비됨` : '';
}
// 폴더째 넣으면 .DS_Store 같은 것이 딸려 온다 — 조용히 버린다
const JUNK = (n) => n.startsWith('.') || n === 'Thumbs.db' || n === 'desktop.ini';
function addFiles(list) {
  const have = new Set(upState.files.map((f) => f.name + ':' + f.size));   // 같은 파일을 두 번 떨어뜨려도 한 번만
  [...list].forEach((f) => {
    if (JUNK(f.name) || have.has(f.name + ':' + f.size)) return;
    have.add(f.name + ':' + f.size); upState.files.push(f);
  });
  renderPicked();
}
$('#upFiles').addEventListener('change', (e) => { addFiles(e.target.files); e.target.value = ''; });
$('#upDir').addEventListener('change', (e) => { addFiles(e.target.files); e.target.value = ''; });   // 폴더 선택 — 하위까지 전부 들어온다

// 폴더를 끌어다 놓으면 dataTransfer.files 에는 폴더 한 칸만 잡힌다.
// 항목(entry)을 재귀로 훑어 안에 든 파일만 모은다. 경로는 버린다 — 어디로 갈지는 서버가 확장자로 다시 정한다.
async function collectEntries(entries) {
  const out = [];
  const walk = async (en) => {
    if (en.isFile) out.push(await new Promise((res, rej) => en.file(res, rej)));
    else if (en.isDirectory) {
      const rd = en.createReader();
      for (;;) {                                    // readEntries 는 한 번에 100개씩만 준다
        const batch = await new Promise((res, rej) => rd.readEntries(res, rej));
        if (!batch.length) break;
        for (const e of batch) await walk(e);
      }
    }
  };
  for (const en of entries) await walk(en);
  return out;
}
const dz = $('#dropZone');
['dragenter', 'dragover'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add('on'); }));
['dragleave', 'drop'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove('on'); }));
dz.addEventListener('drop', (e) => {
  // dataTransfer 는 await 뒤에 비워지므로 지금 당장 꺼내 둔다
  const entries = [...(e.dataTransfer.items || [])].map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null)).filter(Boolean);
  const plain = [...e.dataTransfer.files];
  if (!entries.length) return addFiles(plain);
  $('#upMsg').textContent = '폴더를 읽는 중…';
  collectEntries(entries).then(addFiles).catch(() => addFiles(plain));
});

$('#upBtn').addEventListener('click', async () => {
  const c = state.card; if (!c) return;
  const good = upState.files.filter((f) => UP_OK.includes(extOf(f.name)));
  if (!good.length) return;
  const fd = new FormData();
  good.forEach((f) => fd.append('files', f, f.name));
  if ($('#upLabel').value) fd.append('label', $('#upLabel').value);
  if ($('#upMethod').value) fd.append('method', $('#upMethod').value);
  if ($('#upOwner').value) fd.append('owner', $('#upOwner').value);
  if ($('#upNote').value) fd.append('note', $('#upNote').value);
  fd.append('overwrite', $('#upOverwrite').checked ? 'true' : 'false');
  $('#upBtn').disabled = true; $('#upMsg').textContent = '올리는 중… 큰 파일은 몇 분 걸립니다';
  try {
    const r = await fetch(`/api/artifacts/${encodeURIComponent(c.artifact_id)}/restored`, { method: 'POST', body: fd });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || r.status);
    $('#upMsg').textContent = `등록 완료 — 파일 ${d.files}개 · ${fmtMB(d.bytes)}` + (d.reference_fixed?.length ? ` · 참조 ${d.reference_fixed.length}건 수정` : '');
    if (d.warning) alert(d.warning);
    upState.files = []; renderPicked();
    location.hash = '#card/' + encodeURIComponent(d.asset_id);   // 새로 생긴 복원본 카드로 이동
    loadStats();
    waitPreview(d.asset_id, 60, d.preview_url);                  // GLB 면 즉시, 아니면 만들어지는 대로 켠다
  } catch (e) {
    $('#upMsg').textContent = '실패: ' + e.message;
    $('#upBtn').disabled = false;
  }
});

// 업로드 직후 서버가 Blender 로 GLB 를 만든다(보통 10~30초). 다 되면 뷰어를 자동으로 켠다.
async function waitPreview(assetId, tries = 60, readyUrl = null) {
  const pv = $('#pv');
  if (readyUrl && state.card && state.card.asset_id === assetId) {
    // GLB 를 올린 경우: 변환을 기다리지 않고 올린 파일을 그대로 먼저 보여준다
    state.card.preview_url = readyUrl; state.card.preview_kind = 'glb';
    await showPreview(null, state.card);
    $('#pvHint').textContent += ' · 올린 GLB 그대로 · 경량 판본을 만드는 중';
  }
  for (let i = 0; i < tries; i++) {
    if (!location.hash.includes(encodeURIComponent(assetId))) return;   // 사용자가 다른 화면으로 갔으면 중단
    let s;
    try { s = await api('/api/preview-status?asset=' + encodeURIComponent(assetId)); } catch { return; }
    if (s.state === 'done' && s.preview_url) {
      if (state.card && state.card.asset_id === assetId) {
        state.card.preview_url = s.preview_url; state.card.preview_kind = 'glb';
        await showPreview(null, state.card);
        $('#cmpBtn').hidden = !state.card.siblings.some((x) => x.media_type === '3d' && x.preview_url);
      }
      return;
    }
    if (s.state === 'failed') {
      pv.innerHTML = `<span class="msg">미리보기 생성 실패: ${esc(s.error || '')}<br><button id="pvRetry" class="ghost">다시 시도</button></span>`;
      $('#pvRetry')?.addEventListener('click', async () => {
        await api('/api/preview/rebuild?asset=' + encodeURIComponent(assetId), { method: 'POST' });
        waitPreview(assetId);
      });
      return;
    }
    if (i === 0 && !readyUrl) pv.innerHTML = '<span class="msg">3D 미리보기를 만드는 중… (보통 10~30초)</span>';
    await new Promise((r) => setTimeout(r, 3000));
  }
}

// ───────── 이벤트 ─────────
$('#mediaNav').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; state.media = b.dataset.media; $('#mediaNav').querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b)); location.hash = ''; loadList(); });
let t; $('#q').addEventListener('input', () => { clearTimeout(t); t = setTimeout(() => { state.q = $('#q').value.trim(); loadList(); }, 250); });
$('#fSource').addEventListener('change', (e) => { state.source = e.target.value; loadList(); });
$('#fVariant').addEventListener('change', (e) => { state.variant = e.target.value; loadList(); });
$('#fRestore').addEventListener('change', (e) => { state.restore = e.target.value; loadList(); });
$('#more').addEventListener('click', () => loadList(true));
$('#back').addEventListener('click', () => { location.hash = ''; });
$('#pvReset').addEventListener('click', () => { state.viewer?.reset(); state.viewer2?.reset(); });
$('#cmpBtn').addEventListener('click', () => compare(state.card));
$('#save').addEventListener('click', async () => {
  const c = state.card; $('#saveMsg').textContent = '저장 중…';
  try {
    await api('/api/artifacts/' + encodeURIComponent(c.artifact_id), { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ tags: $('#eTags').value || null, note: $('#eNote').value || null }) });
    await api('/api/cards/' + c.asset_id.split('/').map(encodeURIComponent).join('/'), { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ method: $('#eMethod').value || null, owner: $('#eOwner').value || null }) });
    $('#saveMsg').textContent = '저장됨 ✓';
  } catch (e) { $('#saveMsg').textContent = '실패: ' + e.message; }
});
$('#rtSave').addEventListener('click', async () => {
  const c = state.card; if (!c) return;
  const picked = [...$('#rtPick').querySelectorAll('input:checked')].map((x) => x.value);
  $('#rtMsg').textContent = '저장 중…';
  try {
    const r = await api('/api/cards/' + c.asset_id.split('/').map(encodeURIComponent).join('/'),
      { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ restore_types: picked.join(',') || null }) });
    c.restore_types = r.restore_types; c.restore_types_ko = r.restore_types_ko;
    $('#dBadges').innerHTML = badge(c);                 // 제목 옆 배지도 바로 갱신
    $('#rtMsg').textContent = '저장됨 ✓';
  } catch (e) { $('#rtMsg').textContent = '실패: ' + e.message; }
});
$('#rescan').addEventListener('click', async () => { if (!confirm('files/ 를 다시 스캔해 DB 를 갱신합니다. 진행할까요?')) return; $('#rescan').disabled = true; try { const r = await api('/api/rescan', { method: 'POST' }); alert(`재스캔 완료 (${r.took_ms} ms)\n유물 ${r.scan.artifacts} · 카드 ${r.scan.assets} · 파일 ${r.scan.files}`); loadStats(); loadList(); } catch (e) { alert('실패: ' + e.message); } finally { $('#rescan').disabled = false; } });
// ───────── 새 유물 등록 ─────────
// 3D 칸과 2D 칸을 나눠 받는다. 사용자가 넣은 칸이 곧 분류라 서버가 짐작할 일이 없다.
const nState = { f3: [], f2: [] };
const nDest = (name, bucket) => {
  const e = extOf(name);
  if (bucket === '2d') return /compare|damage|overlay|report|diag/.test(name.toLowerCase()) ? 'records' : 'photo_2d';
  return destOf(name);
};
function nRender() {
  for (const [key, ul, bucket] of [['f3', '#nList3', '3d'], ['f2', '#nList2', '2d']]) {
    const el = $(ul); el.replaceChildren();
    nState[key].forEach((f, i) => {
      const badExt = !UP_OK.includes(extOf(f.name));
      const badHere = bucket === '2d' && MODEL_EXT.includes(extOf(f.name));
      const li = document.createElement('li');
      if (badExt || badHere) li.className = 'bad';
      li.innerHTML = `<span class="n">${esc(f.name)}</span><span class="d">${
        badHere ? '3D 칸에 넣어 주세요' : badExt ? '지원하지 않는 형식' : nDest(f.name, bucket) + ' · ' + fmtMB(f.size)}</span>` +
        `<button class="x" title="빼기">×</button>`;
      li.querySelector('.x').addEventListener('click', () => { nState[key].splice(i, 1); nRender(); });
      el.appendChild(li);
    });
  }
  const ok3 = nState.f3.filter((f) => UP_OK.includes(extOf(f.name)));
  const ok2 = nState.f2.filter((f) => UP_OK.includes(extOf(f.name)) && !MODEL_EXT.includes(extOf(f.name)));
  const bytes = [...ok3, ...ok2].reduce((a, f) => a + f.size, 0);
  const bad2 = nState.f2.some((f) => MODEL_EXT.includes(extOf(f.name)));
  $('#nFileMsg').textContent = (ok3.length || ok2.length)
    ? `3D ${ok3.length}개 · 사진 ${ok2.length}장 · ${fmtMB(bytes)}` + (bad2 ? ' · 사진 칸의 3D 파일은 빼 주세요' : '')
    : '';
  $('#nSubmit').disabled = !(ok3.length + ok2.length) || !$('#nName').value.trim() || bad2;
}
const MODEL_EXT = ['.obj', '.ply', '.stl', '.glb', '.las'];
const nAdd = (key, list) => {
  const have = new Set(nState[key].map((f) => f.name + ':' + f.size));
  [...list].forEach((f) => { if (!JUNK(f.name) && !have.has(f.name + ':' + f.size)) { have.add(f.name + ':' + f.size); nState[key].push(f); } });
  nRender();
};
$('#nFiles3').addEventListener('change', (e) => { nAdd('f3', e.target.files); e.target.value = ''; });
$('#nDir3').addEventListener('change', (e) => { nAdd('f3', e.target.files); e.target.value = ''; });
$('#nFiles2').addEventListener('change', (e) => { nAdd('f2', e.target.files); e.target.value = ''; });
$('#nName').addEventListener('input', nRender);
for (const [id, key] of [['#nDrop3', 'f3'], ['#nDrop2', 'f2']]) {
  const z = $(id);
  ['dragenter', 'dragover'].forEach((t) => z.addEventListener(t, (e) => { e.preventDefault(); z.classList.add('on'); }));
  ['dragleave', 'drop'].forEach((t) => z.addEventListener(t, (e) => { e.preventDefault(); z.classList.remove('on'); }));
  z.addEventListener('drop', (e) => {
    const entries = [...(e.dataTransfer.items || [])].map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null)).filter(Boolean);
    const plain = [...e.dataTransfer.files];
    if (!entries.length) return nAdd(key, plain);
    collectEntries(entries).then((fs) => nAdd(key, fs)).catch(() => nAdd(key, plain));
  });
}
$('#newBtn').addEventListener('click', () => { location.hash = '#new'; });
$('#newBack').addEventListener('click', () => { location.hash = ''; });
$('#nSubmit').addEventListener('click', async () => {
  const fd = new FormData();
  nState.f3.filter((f) => UP_OK.includes(extOf(f.name))).forEach((f) => fd.append('files_3d', f, f.name));
  nState.f2.filter((f) => UP_OK.includes(extOf(f.name)) && !MODEL_EXT.includes(extOf(f.name))).forEach((f) => fd.append('files_2d', f, f.name));
  fd.append('name_ko', $('#nName').value.trim());
  for (const [k, id] of [['description_ko', '#nDesc'], ['accession', '#nAcc'], ['designation', '#nDesig'],
                         ['period', '#nPeriod'], ['material', '#nMaterial'], ['museum', '#nMuseum'],
                         ['size', '#nSize'], ['owner', '#nOwner'], ['note', '#nNote']]) {
    if ($(id).value.trim()) fd.append(k, $(id).value.trim());
  }
  $('#nSubmit').disabled = true; $('#nMsg').textContent = '올리는 중… 큰 파일은 몇 분 걸립니다';
  try {
    const r = await api('/api/artifacts', { method: 'POST', body: fd });
    await loadStats();
    if (r.warning) alert(r.warning);
    nState.f3 = []; nState.f2 = [];
    ['#nName', '#nDesc', '#nAcc', '#nDesig', '#nPeriod', '#nMaterial', '#nMuseum', '#nSize', '#nOwner', '#nNote'].forEach((i) => ($(i).value = ''));
    nRender(); $('#nMsg').textContent = '';
    location.hash = '#card/' + encodeURIComponent(r.asset_3d || r.asset_2d);   // 만들어진 카드로 바로 간다
  } catch (e) {
    $('#nMsg').textContent = '실패: ' + e.message; $('#nSubmit').disabled = false;
  }
});

// ───────── 피드백 — 채팅방 ─────────
// 자유롭게 적고 아래로 쌓인다. 상태 알약이 그 자체로 선택기다(미확인·확인·대기·진행·완료).
const FB_ST = { new: '미확인', seen: '확인', hold: '대기', doing: '진행', done: '완료' };
const ME_KEY = 'c201.author';
const fbState = { filter: '' };
let fbSig = null, fbTimer = null;        // 마지막으로 그린 내용의 지문 · 자동 갱신 타이머
const fbMe = () => (localStorage.getItem(ME_KEY) || '').trim();
// 서버는 UTC 로 준다(끝에 Z). 문자열을 그대로 자르면 한국보다 9시간 이른 시각이 떴다 — 보는 사람 시각으로 바꾼다.
const fbAt = (t) => { const d = new Date(t); return isNaN(d) ? null : d; };
const p2 = (n) => String(n).padStart(2, '0');
const fbDay = (t) => { const d = fbAt(t); return d ? `${d.getFullYear()}-${p2(d.getMonth() + 1)}-${p2(d.getDate())}` : (t || '').slice(0, 10); };
const fbTime = (t) => { const d = fbAt(t); return d ? `${p2(d.getHours())}:${p2(d.getMinutes())}` : (t || '').slice(11, 16); };
// 이름을 색 6개 중 하나로 (같은 사람은 늘 같은 색)
const fbHue = (name) => [...name].reduce((a, c) => (a * 31 + c.charCodeAt(0)) % 6, 7);

// force=true 면 무조건 다시 그리고 맨 아래로 내린다 (내가 글을 남겼을 때)
async function loadFeedback(force) {
  const r = await api('/api/feedback' + (fbState.filter ? '?status=' + fbState.filter : ''));
  // 바뀐 게 없으면 다시 그리지 않는다 — 4초마다 깜빡이고, 열어 둔 상태 드롭다운이 닫히고,
  // 읽던 스크롤이 튀는 것을 막기 위해.
  const sig = r.total + '|' + r.open + '|' + fbState.filter + '|' +
    r.items.map((f) => [f.id, f.status, f.body, f.author, f.updated_at].join('')).join('');
  if (!force && sig === fbSig) return;
  fbSig = sig;
  const room = $('#fbList');                 // 스크롤하는 쪽은 글 목록이다 (방은 토끼를 붙들고 있다)
  // 맨 아래를 보고 있었으면 새 글을 따라 내려간다. 위를 읽고 있었으면 그 자리를 지킨다 (채팅 정석)
  const wasAtBottom = force || room.scrollHeight - room.scrollTop - room.clientHeight < 40;
  const keep = room.scrollTop;
  $('#fbCounts').innerHTML = `<b>${r.total}</b><span>전체</span>` +
    (r.open ? `<b class="open">${r.open}</b><span>안 끝난 것</span>` : '<span>다 처리됨 🎉</span>');
  $('#fbN').textContent = r.open || '';
  const ul = $('#fbList'); ul.replaceChildren();
  if (!r.items.length) {
    ul.innerHTML = '<li class="empty"><svg class="rab huge"><use href="#ic-rabbit"/></svg>아직 남긴 피드백이 없습니다<small>아래에 적으면 여기에 쌓입니다</small></li>';
    return;
  }
  let day = '', prev = null;
  r.items.forEach((f) => {
    const d = fbDay(f.created_at);
    if (d !== day) {                                  // 날짜가 바뀌면 구분선 (채팅방 관례)
      day = d; prev = null;
      const li = document.createElement('li');
      li.className = 'fbday'; li.innerHTML = `<span>${esc(d)}</span>`;
      ul.appendChild(li);
    }
    // 같은 사람이 5분 안에 이어 쓴 글은 아바타·이름을 접는다
    const cont = !!prev && prev.author === f.author && fbTime(f.created_at) === fbTime(prev.created_at);
    ul.appendChild(fbEl(f, cont));
    prev = f;
  });
  room.scrollTop = wasAtBottom ? room.scrollHeight : keep;
}

// ───────── 자동 갱신 ─────────
// 남이 쓴 글이 바로 보이려면 주기적으로 다시 읽어야 한다(WebSocket 없이 4초 폴링).
// 피드백 화면에 있을 때만, 탭이 앞에 있을 때만 돈다.
function fbPoll(on) {
  if (fbTimer) { clearInterval(fbTimer); fbTimer = null; }
  if (!on) return;
  fbTimer = setInterval(() => { if (!document.hidden) loadFeedback().catch(() => {}); }, 4000);
}
document.addEventListener('visibilitychange', () => {      // 탭으로 돌아오면 바로 한 번
  if (!document.hidden && fbTimer) loadFeedback().catch(() => {});
});

function fbEl(f, cont) {
  // 내가 쓴 것과 남이 쓴 것을 좌우로 나누지 않는다 — 팀 게시판이라 한 줄기로 읽는 게 낫다
  const who = f.author || '익명';
  const li = document.createElement('li');
  li.className = 'fb ' + f.status + (cont ? ' cont' : '');
  li.innerHTML = `
    <span class="av c${fbHue(who)}">${f.author ? esc(who.slice(0, 1)) : '<svg class="rab"><use href="#ic-rabbit"/></svg>'}</span>
    <span class="col">
      <span class="who">${esc(who)}${f.updated_at ? ' · 수정됨' : ''}</span>
      <span class="bubble"><select class="st ${f.status}" title="처리 상태">${
        Object.entries(FB_ST).map(([k, v]) => `<option value="${k}"${k === f.status ? ' selected' : ''}>${v}</option>`).join('')
      }</select>${esc(f.body)}</span>
    </span>
    <span class="acts"><span class="at">${esc(fbTime(f.created_at))}</span><button class="del" title="지우기">×</button></span>`;
  const sel = li.querySelector('select');
  sel.addEventListener('change', async () => {
    try { await api('/api/feedback/' + f.id, { method: 'PATCH', headers: { 'Content-Type': 'application/json' },
                                               body: JSON.stringify({ status: sel.value }) });
      loadFeedback(true); loadStats();
    } catch (e) { alert('실패: ' + e.message); }
  });
  li.querySelector('.del').addEventListener('click', async () => {
    if (!confirm('이 피드백을 지웁니다.')) return;
    try { await api('/api/feedback/' + f.id, { method: 'DELETE' }); loadFeedback(true); loadStats(); }
    catch (e) { alert('실패: ' + e.message); }
  });
  return li;
}

function fbSetFilter(f) {
  fbState.filter = f;
  $('#fbFilter').querySelectorAll('button').forEach((x) => x.classList.toggle('on', x.dataset.f === f));
}
$('#fbFilter').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  fbSetFilter(b.dataset.f);
  loadFeedback(true);
});
const fbBody = $('#fbBody');
fbBody.addEventListener('input', () => {
  $('#fbSend').disabled = !fbBody.value.trim();
  fbBody.style.height = 'auto'; fbBody.style.height = Math.min(fbBody.scrollHeight, 140) + 'px';   // 줄 수에 따라 자란다
});
// 채팅답게 Enter 로 보낸다. 줄바꿈은 Ctrl+Enter (Shift+Enter 도 받는다 — 손에 익은 사람이 있다)
fbBody.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter' || e.isComposing) return;          // 한글 조합 중의 Enter 는 글자 확정이다
  if (e.ctrlKey || e.metaKey || e.shiftKey) {              // 줄바꿈 — 커서 자리에 끼워 넣는다
    e.preventDefault();
    const s = fbBody.selectionStart, t = fbBody.selectionEnd;
    fbBody.value = fbBody.value.slice(0, s) + '\n' + fbBody.value.slice(t);
    fbBody.selectionStart = fbBody.selectionEnd = s + 1;
    fbBody.dispatchEvent(new Event('input', { bubbles: true }));   // 높이도 같이 자라게
    return;
  }
  e.preventDefault();
  if (fbBody.value.trim()) $('#fbSend').click();
});
$('#fbSend').addEventListener('click', async () => {
  $('#fbSend').disabled = true; $('#fbMsg').textContent = '보내는 중…';
  try {
    await api('/api/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ body: fbBody.value, author: $('#fbAuthor').value }) });   // 상태는 '미확인' 으로 시작 — 보내진 말풍선 앞에서 바꾼다
    localStorage.setItem(ME_KEY, $('#fbAuthor').value.trim());   // 다음에도 이름이 채워지게
    fbBody.value = ''; fbBody.style.height = 'auto'; $('#fbMsg').textContent = '';
    // 필터가 '완료' 인 채로 남기면 새 글은 '미확인' 이라 화면에 안 나왔다
    // ("새로고침해야 뜬다" 의 원인 — 새로고침하면 필터가 초기화되니까). 필터를 푼다.
    fbSetFilter('');
    $('#fbSend').disabled = true;
    await loadFeedback(true); loadStats();
  } catch (e) { $('#fbMsg').textContent = '실패: ' + e.message; $('#fbSend').disabled = false; }
});
$('#fbBtn').addEventListener('click', () => { location.hash = '#feedback'; });
$('#fbBack').addEventListener('click', () => { location.hash = ''; });

function route() {
  const m = location.hash.match(/^#card\/(.+)$/);
  $('#newArt').hidden = true; $('#fbPage').hidden = true;
  if (m) { openCard(decodeURIComponent(m[1])); return; }
  disposeViewers(); $('#detail').hidden = true;
  if (location.hash === '#new') { $('#list').hidden = true; $('#newArt').hidden = false; window.scrollTo(0, 0); return; }
  $('.foot').hidden = false;
  if (location.hash === '#feedback') {
    $('#list').hidden = true; $('#fbPage').hidden = false; window.scrollTo(0, 0);
    $('.foot').hidden = true;                                   // 피드백 화면은 하단바 없이 방을 끝까지 쓴다
    $('#fbAuthor').value = $('#fbAuthor').value || fbMe();
    loadFeedback(true); fbPoll(true); return;                   // 남이 쓴 글도 바로 보이게 자동 갱신
  }
  fbPoll(false);
  $('#list').hidden = false;
  if (!$('#grid').children.length) loadList();
}
window.addEventListener('hashchange', route);
loadStats(); route();
// 화면이 다 뜨고 한가할 때 즐겨찾기 3D 를 미리 받아 둔다 (첫 화면 로딩과 경쟁하지 않게)
(window.requestIdleCallback || ((f) => setTimeout(f, 1500)))(() => prefetchFavorites(), { timeout: 5000 });

// ───────── 새 버전 감지 ─────────
// 화면 파일이 배포로 바뀌면 열어 둔 탭은 옛 화면을 계속 쓴다(새 기능이 안 보인다).
// /api/health 의 asset_version 을 1분마다 확인해 바뀌면 알려 준다.
let myVer = null;
async function checkVersion() {
  try {
    const h = await (await fetch('/api/health', { cache: 'no-store' })).json();
    if (myVer === null) { myVer = h.asset_version; return; }
    if (h.asset_version && h.asset_version !== myVer) $('#newVer').hidden = false;
  } catch { /* 서버가 잠깐 없을 수도 있다 — 조용히 넘긴다 */ }
}
$('#verReload').addEventListener('click', () => location.reload());
checkVersion();
setInterval(checkVersion, 60000);
