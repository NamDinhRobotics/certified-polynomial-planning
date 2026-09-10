import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = id => document.getElementById(id);
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const heroVideo = document.querySelector('.hero-visual video');
const heroPause = document.createElement('button');
heroPause.className = 'hero-pause';
heroPause.type = 'button';
heroPause.textContent = 'Pause film Ⅱ';
heroPause.setAttribute('aria-label', 'Pause background film');
heroVideo.after(heroPause);
function syncHeroButton() {
  heroPause.textContent = heroVideo.paused ? 'Play film ▶' : 'Pause film Ⅱ';
  heroPause.setAttribute('aria-label', heroVideo.paused ? 'Play background film' : 'Pause background film');
}
heroPause.addEventListener('click', () => heroVideo.paused ? heroVideo.play().catch(syncHeroButton) : heroVideo.pause());
heroVideo.addEventListener('play', syncHeroButton);
heroVideo.addEventListener('pause', syncHeroButton);
if (reducedMotion.matches) heroVideo.pause();
reducedMotion.addEventListener('change', event => { if (event.matches) heroVideo.pause(); });
syncHeroButton();

document.querySelectorAll('[data-chapter]').forEach(button => {
  button.addEventListener('click', () => {
    const film = $('research-film');
    const seek = () => { film.currentTime = Number(button.dataset.chapter); film.play().catch(() => {}); };
    if (film.readyState) seek();
    else { film.addEventListener('loadedmetadata', seek, { once: true }); film.load(); }
    film.scrollIntoView({ behavior: reducedMotion.matches ? 'instant' : 'smooth', block: 'center' });
  });
});
$('copy-command').addEventListener('click', async () => {
  const button = $('copy-command');
  try {
    await navigator.clipboard.writeText($('verify-command').textContent);
    button.textContent = 'Copied ✓';
  } catch {
    const range = document.createRange();
    range.selectNodeContents($('verify-command'));
    getSelection().removeAllRanges();
    getSelection().addRange(range);
    button.textContent = 'Selected — copy';
  }
  setTimeout(() => { button.textContent = 'Copy ↗'; }, 2500);
});

let data, historicalData, current, stage = 3, time = 0, playing = false;
let renderer, scene, camera, orbit, recordGroup, robot, trackedLine;
let viewReady = false, inViewport = true, dirty = true, lastFrame = 0;
const layers = {};
const campaigns = new Map();
const objectivePaths = { original_certificate: 'Original certificate', current_root_last_attempt: 'Normalized dual repair', independent_same_SDP_objective_witness: 'Same-SDP witness fallback' };
const groupNames = { easy: 'easy', cluttered: 'cluttered', on_axis_recovery_target: 'on-axis recovery target' };
const stageNames = ['SDP PROJECTION / BEFORE RECOVERY', 'RECOVERY / PHYSICAL CURVE', 'VERIFICATION / STORED EXACT CHECKS', 'TRACKING / RECORDED TELEMETRY'];

function showFallback() {
  viewReady = false;
  playing = false;
  $('viewer-status').hidden = true;
  $('viewer-fallback').hidden = false;
  $('viewer-wrap').classList.add('fallback-mode');
  document.querySelector('.layer-controls').hidden = true;
  $('play-button').disabled = true;
  $('restart-button').disabled = true;
  $('timeline').disabled = true;
  $('flight-hud').hidden = true;
}

function init3D() {
  const mount = $('scene-canvas');
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'low-power' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  mount.append(renderer.domElement);
  renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute('aria-label', '3D scene. Use arrow keys to pan; use the camera buttons for preset views.');
  renderer.domElement.addEventListener('webglcontextlost', event => { event.preventDefault(); showFallback(); });
  scene = new THREE.Scene();
  scene.background = new THREE.Color('#111c30');
  scene.fog = new THREE.Fog('#111c30', 14, 35);
  camera = new THREE.PerspectiveCamera(40, 1, .05, 80);
  camera.up.set(0, 0, 1);
  orbit = new OrbitControls(camera, renderer.domElement);
  orbit.listenToKeyEvents(renderer.domElement);
  orbit.enableDamping = false;
  orbit.minDistance = 3.5;
  orbit.maxDistance = 22;
  orbit.maxPolarAngle = Math.PI * .94;
  orbit.addEventListener('change', () => { dirty = true; });
  scene.add(new THREE.HemisphereLight(0xd9fff0, 0x263537, 2.4));
  const key = new THREE.DirectionalLight(0xecfff6, 3.2);
  key.position.set(-3, -4, 8);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x58cbb9, 2.1);
  rim.position.set(4, 5, 3);
  scene.add(rim);
  const grid = new THREE.GridHelper(30, 60, 0x3d587b, 0x233954);
  grid.rotation.x = Math.PI / 2;
  grid.material.transparent = true;
  grid.material.opacity = .45;
  scene.add(grid);
  recordGroup = new THREE.Group();
  scene.add(recordGroup);
  const resize = () => {
    const { width, height } = mount.getBoundingClientRect();
    if (!width || !height) return;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    dirty = true;
  };
  new ResizeObserver(resize).observe(mount);
  new IntersectionObserver(entries => { inViewport = entries[0].isIntersecting; dirty = true; }, { rootMargin: '100px' }).observe(mount);
  resize();
  viewReady = true;
  setCamera('perspective');
  $('viewer-status').hidden = true;
  requestAnimationFrame(frame);
}

function setCamera(name) {
  if (!viewReady) return;
  orbit.target.set(0, 0, 1.25);
  camera.up.set(0, 0, 1);
  if (name === 'top') { camera.position.set(0, -.001, 12); }
  else if (name === 'side') { camera.position.set(0, -12, 1.8); }
  else { camera.position.set(7, -9, 6.5); }
  orbit.update();
  document.querySelectorAll('[data-view]').forEach(button => {
    const active = button.dataset.view === name;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  dirty = true;
}

function material(color, options = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness: .5, metalness: .35, ...options });
}
function line(points, color, dashed = false, opacity = 1) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(...p)));
  const lineMaterial = dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: .1, gapSize: .065, transparent: true, opacity })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  const object = new THREE.Line(geometry, lineMaterial);
  if (dashed) object.computeLineDistances();
  return object;
}
function referenceCurve(points) {
  const group = new THREE.Group();
  if (points.length < 2) return group;
  // Linear interpolation preserves the sampled polyline used by the viewer.
  class SampledCurve extends THREE.Curve {
    getPoint(t, target = new THREE.Vector3()) {
      const offset = Math.min(points.length - 1, t * (points.length - 1));
      const i = Math.min(points.length - 2, Math.floor(offset));
      return target.fromArray(points[i]).lerp(new THREE.Vector3(...points[i + 1]), offset - i);
    }
  }
  const curve = new SampledCurve();
  const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, points.length - 1, .013, 6, false), material(0x65d8c7, { emissive: 0x65d8c7, emissiveIntensity: .7, roughness: .3 }));
  group.add(tube);
  return group;
}
function positionLabel(text, position, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 80;
  const ctx = canvas.getContext('2d');
  ctx.font = '600 29px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillStyle = color;
  ctx.fillText(text, 128, 46);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false }));
  sprite.position.set(position[0], position[1], position[2] + .38);
  sprite.scale.set(.95, .3, 1);
  return sprite;
}
function marker(position, label, color) {
  const group = new THREE.Group();
  const dot = new THREE.Mesh(new THREE.SphereGeometry(.055, 16, 12), material(color, { emissive: color, emissiveIntensity: .7 }));
  dot.position.set(...position);
  group.add(dot, positionLabel(label, position, '#' + new THREE.Color(color).getHexString()));
  const guide = line([[position[0], position[1], 0], position], color, true, .22);
  group.add(guide);
  return group;
}
function makeRobot() {
  const group = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(.17, .11, .055), material(0xe8f6ed));
  group.add(body);
  for (const x of [-.1, .1]) for (const y of [-.1, .1]) {
    group.add(line([[0, 0, 0], [x, y, 0]], 0xd0ddd6));
    const rotor = new THREE.Mesh(new THREE.TorusGeometry(.062, .008, 5, 24), material(0xb3f2bf));
    rotor.position.set(x, y, .01);
    group.add(rotor);
  }
  const envelope = new THREE.Mesh(new THREE.SphereGeometry(.23, 16, 10), new THREE.MeshBasicMaterial({ color: 0xb3f2bf, wireframe: true, transparent: true, opacity: .1, depthWrite: false }));
  group.add(envelope);
  return group;
}
function rebuildScene() {
  if (!viewReady) return;
  recordGroup.traverse(object => {
    object.geometry?.dispose();
    const materials = object.material ? (Array.isArray(object.material) ? object.material : [object.material]) : [];
    materials.forEach(m => { m.map?.dispose(); m.dispose(); });
  });
  recordGroup.clear();
  layers.envelope = new THREE.Group();
  for (const obstacle of current.obstacles) {
    const sphere = new THREE.Mesh(new THREE.IcosahedronGeometry(obstacle.radius, 3), material(0x556e70, { flatShading: true, roughness: .45 }));
    sphere.position.set(...obstacle.center);
    recordGroup.add(sphere);
    const shell = new THREE.Mesh(new THREE.SphereGeometry(obstacle.planningRadius, 32, 20), material(0x9ebdb6, { transparent: true, opacity: .035, depthWrite: false }));
    shell.position.copy(sphere.position);
    layers.envelope.add(shell);
    for (let plane = 0; plane < 3; plane++) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(obstacle.planningRadius, .005, 4, 80), new THREE.MeshBasicMaterial({ color: 0x75998d, transparent: true, opacity: .37 }));
      ring.position.copy(sphere.position);
      if (plane === 1) ring.rotation.x = Math.PI / 2;
      if (plane === 2) ring.rotation.y = Math.PI / 2;
      layers.envelope.add(ring);
    }
  }
  recordGroup.add(layers.envelope, marker(current.start, 'START', 0xb3f2bf), marker(current.goal, 'GOAL', 0xb3f2bf));
  layers.projected = line(current.projected, 0xe7a277, true, .85);
  layers.recovered = referenceCurve(current.recovered);
  trackedLine = line(current.samples.map(s => s.slice(1, 4)), 0xf6fff6, false, .95);
  layers.tracked = new THREE.Group();
  robot = makeRobot();
  layers.tracked.add(trackedLine, robot);
  recordGroup.add(layers.projected, layers.recovered, layers.tracked);
  updateLayers();
  dirty = true;
}

function updateLayers() {
  if (!current) return;
  for (const key of ['projected', 'recovered', 'tracked', 'envelope']) {
    if (layers[key]) layers[key].visible = $('layer-' + key).checked;
  }
  $('flight-hud').hidden = !viewReady || !current.samples.length || stage !== 3 || !$('layer-tracked').checked;
  dirty = true;
}
function setStage(value) {
  stage = value;
  document.querySelectorAll('[data-stage]').forEach(button => {
    const active = Number(button.dataset.stage) === stage;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  $('layer-projected').checked = stage < 2 || stage === 3;
  $('layer-recovered').checked = stage > 0;
  $('layer-tracked').checked = stage === 3;
  $('layer-envelope').checked = stage > 0;
  if (stage !== 3) setPlaying(false);
  updateDescription();
  updateLayers();
  syncPlayback();
}
function updateDescription() {
  if (!current) return;
  const failed = current.status === 'ROOT_FAILURE';
  const physicalOnly = current.status === 'PHYSICAL_ONLY';
  const descriptions = [
    `The projected curve has stored status ${current.projectedStatus?.replaceAll('_', ' ').toLowerCase() || 'unavailable'}. A lifted SDP candidate is not itself a collision-free physical path.`,
    'The recovered polynomial reference is shown in teal. Its geometry comes from the exact stored coefficients, sampled for display.',
    physicalOnly ? 'The reference passed physical checks. Its root SDP objective interval did not pass the exact gap gate in this historical configuration.' : 'The physical reference passes exact checks; a separate rational lower–upper interval certifies the root SDP objective. This does not certify global optimality of the recovered physical trajectory.',
    'The reference and recorded execution are separate curves. Tracking error and body clearance below come from the simulation log.'
  ];
  $('stage-label').textContent = failed ? 'ROOT FAILURE / NO RECOVERY OR EXECUTION' : stageNames[stage];
  $('stage-description').textContent = failed ? 'The conic-arm first repetition failed at the root solve. No projected curve, recovered reference, or robot telemetry is available for this attempt.' : descriptions[stage];
  const badge = $('run-badge');
  badge.textContent = failed ? 'ROOT SOLVE FAILED' : stage === 0 ? 'PROJECTED CANDIDATE' : physicalOnly ? 'PHYSICAL CHECKS PASSED' : 'VERIFIED REFERENCE';
  badge.classList.toggle('warning', failed || stage === 0 || physicalOnly);
  $('failure-message').hidden = !failed || !viewReady;
}
function setPlaying(value) {
  playing = Boolean(value && current?.samples.length && stage === 3 && viewReady);
  $('play-button').textContent = playing ? 'Ⅱ' : '▶';
  $('play-button').setAttribute('aria-label', playing ? 'Pause recorded flight' : 'Play recorded flight');
}
function syncPlayback() {
  const unavailable = !current?.samples.length || !viewReady || stage !== 3;
  for (const id of ['play-button', 'restart-button', 'timeline', 'play-speed']) $(id).disabled = unavailable;
}
function selectScene(id) {
  current = data.scenes.find(s => s.id === Number(id));
  if (!current) return;
  setPlaying(false);
  time = 0;
  $('scene-select').value = String(current.id);
  $('scene-caption').textContent = `Scene ${current.id} · ${groupNames[current.group] || current.group}`;
  document.querySelectorAll('[data-scene]').forEach(button => {
    const active = Number(button.dataset.scene) === current.id;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  const duration = current.samples.at(-1)?.[0] || 0;
  $('timeline').max = duration || 1;
  $('timeline').value = 0;
  $('total-time').textContent = `${duration.toFixed(2)} s`;
  $('family-label').textContent = `Degree ${current.family.d} · ${current.family.N} segments · C${'⁰¹²³⁴⁵⁶⁷⁸⁹'[current.family.k]} continuity`;
  $('record-label').textContent = `${data.campaignId || 'v3_20260909'} / ${current.runId}`;
  $('objective-label').textContent = current.objectivePath ? `${objectivePaths[current.objectivePath]} · gap ${current.objectiveGap.toExponential(3)}` : `SDP objective gate: ${current.completeOk ? 'passed' : 'not certified'}`;
  $('root-label').textContent = current.rootStatus ? `Original numerical root: ${current.rootStatus.replaceAll('_', ' ').toLowerCase()}` : '';
  const gates = [['physical-gate', current.physicalOk], ['tracking-gate', current.metrics?.success], ['objective-gate', current.completeOk]];
  for (const [id, passed] of gates) {
    $(id).textContent = passed ? 'Passed' : id === 'tracking-gate' && !current.samples.length ? 'Not executed' : 'Not certified';
    $(id).classList.toggle('unavailable', !passed);
  }
  $('telemetry-grid').hidden = !current.samples.length;
  if (current.metrics) {
    $('max-error-label').textContent = `MAX ${(current.metrics.maxError * 100).toFixed(2)} cm`;
    $('min-clearance-label').textContent = `MIN ${(current.metrics.minClearance * 100).toFixed(2)} cm`;
    drawTelemetry($('error-chart'), 8, '#65d8c7');
    drawTelemetry($('clearance-chart'), 9, '#b3f2bf');
  }
  rebuildScene();
  updateDescription();
  syncPlayback();
  updateTime(0);
}
function sampleAt(t) {
  const samples = current.samples;
  if (t >= samples.at(-1)[0]) return { a: samples.at(-1), b: samples.at(-1), mix: 0, index: samples.length - 1 };
  let low = 0, high = samples.length - 1;
  while (low + 1 < high) {
    const mid = (low + high) >> 1;
    if (samples[mid][0] <= t) low = mid; else high = mid;
  }
  const a = samples[low], b = samples[high];
  const mix = Math.max(0, Math.min(1, (t - a[0]) / (b[0] - a[0] || 1)));
  return { a, b, mix, index: low };
}
function updateTime(t) {
  time = t;
  $('play-time').textContent = `${time.toFixed(2)} s`;
  $('timeline').value = time;
  if (!current?.samples.length) return;
  const { a, b, mix, index } = sampleAt(time);
  const lerp = i => a[i] + mix * (b[i] - a[i]);
  // HUD reports the preceding recorded sample; geometry interpolates for smooth playback.
  $('live-error').textContent = (a[8] * 100).toFixed(2);
  $('live-clearance').textContent = (a[9] * 100).toFixed(2);
  $('live-speed').textContent = a[10].toFixed(2);
  if (viewReady && robot) {
    robot.position.set(lerp(1), lerp(2), lerp(3));
    robot.quaternion.set(a[5], a[6], a[7], a[4]);
    robot.quaternion.slerp(new THREE.Quaternion(b[5], b[6], b[7], b[4]), mix);
    trackedLine.geometry.setDrawRange(0, Math.min(current.samples.length, index + 2));
  }
  const x = 62 + (time / current.samples.at(-1)[0]) * 520;
  document.querySelectorAll('.chart-cursor').forEach(cursor => { cursor.setAttribute('x1', x); cursor.setAttribute('x2', x); });
  dirty = true;
}
function frame(now) {
  const delta = Math.min((now - lastFrame) / 1000, .05);
  lastFrame = now;
  if (viewReady && inViewport && !document.hidden) {
    if (playing) {
      const end = current.samples.at(-1)[0];
      updateTime(Math.min(end, time + delta * Number($('play-speed').value)));
      if (time >= end) setPlaying(false);
    }
    if (dirty) { renderer.render(scene, camera); dirty = false; }
  }
  requestAnimationFrame(frame);
}

function drawTelemetry(svg, column, color) {
  const samples = current.samples;
  const max = Math.max(...samples.map(s => s[column] * 100));
  const ceiling = Math.max(1, Math.ceil(max * 1.15));
  const end = samples.at(-1)[0];
  const fontSize = Math.min(30, 13 * 600 / Math.max(260, svg.getBoundingClientRect().width));
  const path = samples.map((s, i) => `${i ? 'L' : 'M'}${(62 + s[0] / end * 520).toFixed(2)},${(124 - s[column] * 100 / ceiling * 86).toFixed(2)}`).join(' ');
  let markup = '';
  for (const fraction of [0, .5, 1]) {
    const y = 124 - fraction * 86;
    markup += `<line x1="62" x2="582" y1="${y}" y2="${y}" stroke="#30425d"/><text x="52" y="${y + 4}" text-anchor="end" fill="#a7b9d2" font-size="${fontSize}">${(ceiling * fraction).toFixed(1).replace('.0', '')}</text>`;
  }
  markup += `<path d="${path}" fill="none" stroke="${color}" stroke-width="2"/><line class="chart-cursor" x1="62" x2="62" y1="32" y2="127" stroke="#f3f6ff" stroke-opacity=".65" stroke-dasharray="3 3"/><text x="62" y="151" fill="#a7b9d2" font-size="${fontSize}">0 s</text><text x="582" y="151" text-anchor="end" fill="#a7b9d2" font-size="${fontSize}">${end.toFixed(1)} s</text><text x="62" y="24" fill="#a7b9d2" font-size="${fontSize}">cm</text>`;
  svg.innerHTML = markup;
}
function drawEnergy() {
  const ratios = historicalData.summary.energyRatios;
  const min = .89, max = 1.23, count = 34;
  const bins = Array(count).fill(0);
  ratios.forEach(ratio => { bins[Math.max(0, Math.min(count - 1, Math.floor((ratio - min) / (max - min) * count)))]++; });
  const peak = Math.max(...bins);
  const fontSize = Math.min(26, 13 * 600 / Math.max(300, $('energy-chart').getBoundingClientRect().width));
  let markup = `<text x="15" y="22" fill="#61728d" font-size="${fontSize}">${peak} pairs</text>`;
  bins.forEach((n, i) => {
    const height = n / peak * 90;
    const x = 18 + i / count * 562;
    markup += `<rect x="${x}" y="${119 - height}" width="14" height="${height}" rx="1" fill="${n === peak ? '#2f60c9' : '#91b2ec'}"><title>${(min + i * .01).toFixed(2)}–${(min + (i + 1) * .01).toFixed(2)}: ${n} pairs</title></rect>`;
  });
  const center = 18 + (1 - min) / (max - min) * 562;
  markup += `<line x1="18" x2="580" y1="120" y2="120" stroke="#d5dfef"/><line x1="${center}" x2="${center}" y1="29" y2="123" stroke="#2f60c9" stroke-dasharray="3 3"/><text x="580" y="22" text-anchor="end" fill="#2f60c9" font-size="${fontSize}">median 1.000</text>`;
  for (const value of [.9, 1, 1.1, 1.2]) markup += `<text x="${18 + (value - min) / (max - min) * 562}" y="145" text-anchor="middle" fill="#61728d" font-size="${fontSize}">${value.toFixed(2)}</text>`;
  $('energy-chart').innerHTML = markup;
}

document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => setCamera(button.dataset.view)));
document.querySelectorAll('[data-stage]').forEach(button => button.addEventListener('click', () => setStage(Number(button.dataset.stage))));
document.querySelectorAll('[data-jump-stage]').forEach(button => button.addEventListener('click', () => setStage(Number(button.dataset.jumpStage))));
document.querySelectorAll('.layer-controls input').forEach(input => input.addEventListener('change', updateLayers));
$('scene-select').addEventListener('change', event => selectScene(event.target.value));
$('play-button').addEventListener('click', () => {
  if (time >= current.samples.at(-1)[0]) updateTime(0);
  setPlaying(!playing);
});
$('restart-button').addEventListener('click', () => { setPlaying(false); updateTime(0); });
$('timeline').addEventListener('input', event => { setPlaying(false); updateTime(Number(event.target.value)); });
syncPlayback();

function populateScenes() {
  $('scene-select').replaceChildren();
  $('scene-grid').replaceChildren();
  for (const entry of data.scenes) {
    const outcome = entry.status === 'ROOT_FAILURE' ? 'root failure' : 'executed';
    const option = new Option(`${entry.id} · ${groupNames[entry.group]} · ${outcome}`, entry.id);
    $('scene-select').add(option);
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.scene = entry.id;
    button.textContent = String(entry.id).slice(-2);
    button.className = entry.status === 'ROOT_FAILURE' ? 'failed' : '';
    button.setAttribute('aria-label', `Scene ${entry.id}, ${groupNames[entry.group]}, ${outcome}`);
    button.title = `Scene ${entry.id} · ${groupNames[entry.group]} · ${outcome}`;
    button.addEventListener('click', () => selectScene(entry.id));
    $('scene-grid').append(button);
  }
}
function selectCampaign(id) {
  const sceneId = current?.id || 41015;
  data = campaigns.get(id);
  if (!data) return;
  $('campaign-context').textContent = id === 'v3_20260909' ? 'Historical configuration · unsuccessful attempts retained' : 'Same 20 scene IDs · post-hoc validation';
  populateScenes();
  selectScene(sceneId);
}
$('campaign-select').addEventListener('change', event => selectCampaign(event.target.value));
try {
  const datasets = await Promise.all(['./assets/evidence.json', './assets/evidence-objective.json'].map(async path => {
    const response = await fetch(path);
    if (!response.ok) throw new Error('Evidence could not be loaded');
    const evidence = await response.json();
    if (!Array.isArray(evidence.scenes) || evidence.scenes.length !== 20) throw new Error('Incomplete scene data');
    return evidence;
  }));
  historicalData = datasets[0];
  campaigns.set('v3_20260909', historicalData);
  campaigns.set('objective_repair_20260909', datasets[1]);
  data = datasets[1];
  drawEnergy();
  try { init3D(); } catch { showFallback(); }
  selectCampaign($('campaign-select').value);
  new ResizeObserver(() => {
    if (!current) return;
    if (current.samples.length) {
      drawTelemetry($('error-chart'), 8, '#65d8c7');
      drawTelemetry($('clearance-chart'), 9, '#b3f2bf');
      updateTime(time);
    }
    drawEnergy();
  }).observe($('explorer'));
} catch {
  showFallback();
  $('scene-caption').textContent = 'Historical video · scene 41013';
  $('stage-description').textContent = 'Interactive data could not be loaded. The historical film and static results remain available.';
  $('telemetry-grid').hidden = true;
  $('scene-select').disabled = true;
  $('campaign-select').disabled = true;
}
