// three.js GLB 뷰어 — 메시(OBJ→GLB)와 점군(LAS→GLB, POINTS) 둘 다.
// GLB 는 make_preview.py 가 최대 변 2 m(점군은 반경 1 m) 로 정규화해 뒀으므로 카메라는 고정 거리에서 시작한다.
import * as THREE from 'three';
import { OrbitControls } from './vendor/jsm/controls/OrbitControls.js';
import { GLTFLoader } from './vendor/jsm/loaders/GLTFLoader.js';

export function createViewer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.replaceChildren(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;

  // 밝기는 썸네일(blender_preview.py 의 Workbench STUDIO + Standard +0.5EV)에 맞춘 값이다.
  // 같은 유물을 목록 썸네일과 이 뷰어에서 보면 색과 밝기가 같아야 한다 — 셋을 재서 맞췄다:
  //   나전침 썸네일 19 / 뷰어 26 · 죽순주전자 71 / 58 · 백자병 129 / 127
  // 전방위광(Ambient)을 크게 주면 어두운 유물이 통째로 들뜬다. 방향광 위주로 간다.
  // 화면의 밝기 슬라이더는 이 값들에 배율을 곱한다 (base 를 들고 있어야 되돌릴 수 있다)
  const lights = [];
  const addLight = (l) => { lights.push({ l, base: l.intensity }); scene.add(l); return l; };
  addLight(new THREE.HemisphereLight(0xffffff, 0x9a978e, 0.2));
  addLight(new THREE.DirectionalLight(0xffffff, 1.2)).position.set(2.5, 4, 3);    // key
  addLight(new THREE.DirectionalLight(0xffffff, 0.2)).position.set(-3, 1, -2);    // fill
  addLight(new THREE.DirectionalLight(0xffffff, 0.15)).position.set(0, -2, -3);   // rim

  const loader = new GLTFLoader();
  let model = null, raf = 0, onChange = null, syncing = false;

  function resize() {
    const w = container.clientWidth || 1, h = container.clientHeight || 1;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize); ro.observe(container); resize();

  function fit() {
    if (!model) return;
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3()), center = box.getCenter(new THREE.Vector3());
    const r = Math.max(size.x, size.y, size.z) * 0.5 || 1;
    const d = r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.15;
    camera.position.set(center.x + d * 0.62, center.y + d * 0.42, center.z + d * 0.66);
    controls.target.copy(center); camera.near = d / 100; camera.far = d * 20; camera.updateProjectionMatrix(); controls.update();
  }

  function loop() {
    raf = requestAnimationFrame(loop);
    controls.update();
    renderer.render(scene, camera);
  }
  controls.addEventListener('change', () => { if (onChange && !syncing) onChange(camera, controls); });

  return {
    // onProgress(loaded, total) — GLB 는 nginx 가 Content-Length 를 주므로 총량을 안다.
    // 큰 모델(수십 MB)에서 "불러오는 중…" 만 떠 있으면 멈춘 것처럼 보인다.
    load(url, onProgress) {
      return new Promise((res, rej) => {
        if (model) { scene.remove(model); model = null; }
        loader.load(url, (g) => {
          model = g.scene;
          let pts = 0, tris = 0;
          model.traverse((o) => {
            if (o.isPoints) { pts += o.geometry.attributes.position.count; o.material = new THREE.PointsMaterial({ size: 0.012, vertexColors: !!o.geometry.attributes.color, color: o.geometry.attributes.color ? 0xffffff : 0xb9b6ad }); }
            if (o.isMesh) {
              tris += (o.geometry.index ? o.geometry.index.count : o.geometry.attributes.position.count) / 3;
              // 스캔 모델은 금속이 아니다. MTL 에서 넘어온 낮은 거칠기 때문에 반사가 강해 어둡고 번들거린다
              for (const m of [o.material].flat().filter(Boolean)) {
                m.side = THREE.DoubleSide;
                if (m.isMeshStandardMaterial) { m.metalness = 0; m.roughness = 0.95; }
              }
            }
          });
          scene.add(model); fit(); if (!raf) loop();
          res({ points: pts, triangles: Math.round(tris) });
        }, (e) => { if (onProgress) onProgress(e.loaded || 0, e.lengthComputable ? e.total : 0); }, rej);
      });
    },
    reset: fit,
    // 밝기 배율. 1 이 썸네일과 맞춘 기본값 — 어두운 청동기를 들여다볼 때 올려 쓴다
    setBrightness(k) { for (const { l, base } of lights) l.intensity = base * k; },
    getPose() { return { p: camera.position.toArray(), t: controls.target.toArray() }; },
    setPose(pose) { syncing = true; camera.position.fromArray(pose.p); controls.target.fromArray(pose.t); controls.update(); syncing = false; },
    onChange(fn) { onChange = fn; },
    dispose() { cancelAnimationFrame(raf); raf = 0; ro.disconnect(); controls.dispose(); renderer.dispose(); container.replaceChildren(); },
  };
}
