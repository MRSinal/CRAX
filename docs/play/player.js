// In-browser keyboard player for CRAX, the web counterpart of crax/play/play.py.
//
// MuJoCo (WebAssembly) steps the physics, crax-env.js runs the task logic and
// three.js draws the scene from MuJoCo's geom and camera poses.
import * as THREE from 'three';
import { loadEnv, MUJOCO_VERSION } from './crax-env.js';

const MUJOCO_URL = `https://cdn.jsdelivr.net/npm/@mujoco/mujoco@${MUJOCO_VERSION}/mujoco.js`;
const ENV_DIR = new URL('./envs', import.meta.url).href;
const CAMERAS = ['track', 'fixedfar', 'vision'];
const MAX_STEPS = 1000;

// mjtGeom
const PLANE = 0, SPHERE = 2, CAPSULE = 3, ELLIPSOID = 4, CYLINDER = 5, BOX = 6;
const TEXROLE_RGB = 1, NTEXROLE = 10;

function geometryFor(type, size) {
  switch (type) {
    case PLANE: {
      const sx = size[0] > 0 ? size[0] : 50, sy = size[1] > 0 ? size[1] : 50;
      return new THREE.PlaneGeometry(2 * sx, 2 * sy);
    }
    case SPHERE: return new THREE.SphereGeometry(size[0], 32, 16);
    case ELLIPSOID: return new THREE.SphereGeometry(1, 32, 16).scale(size[0], size[1], size[2]);
    case CAPSULE: return new THREE.CapsuleGeometry(size[0], 2 * size[1], 8, 24).rotateX(Math.PI / 2);
    case CYLINDER: return new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 48).rotateX(Math.PI / 2);
    case BOX: return new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
    default: return null;
  }
}

function textureFor(model, texId, repeat) {
  const w = model.tex_width[texId], h = model.tex_height[texId];
  const nc = model.tex_nchannel[texId], adr = Number(model.tex_adr[texId]);
  const src = model.tex_data;
  const rgba = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    for (let c = 0; c < 3; c++) rgba[4 * i + c] = src[adr + nc * i + Math.min(c, nc - 1)];
    rgba[4 * i + 3] = nc === 4 ? src[adr + 4 * i + 3] : 255;
  }
  const tex = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(repeat[0], repeat[1]);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  tex.needsUpdate = true;
  return tex;
}

class SceneView {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0d12);
    this.camera = new THREE.PerspectiveCamera(45, 4 / 3, 0.02, 100);
    this.camera.matrixAutoUpdate = false;
    // MuJoCo renders these scenes with a camera headlight only; add a soft key
    // light with shadows so depth reads better in the browser.
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    this.headlight = new THREE.DirectionalLight(0xffffff, 1.2);
    this.scene.add(this.headlight, this.headlight.target);
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(2, -3, 8);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -5, right: 5, top: 5, bottom: -5, near: 0.5, far: 20 });
    sun.shadow.bias = -0.0005;
    this.scene.add(sun);
    this.meshes = [];
  }

  setModel(model) {
    for (const m of this.meshes) {
      this.scene.remove(m);
      m.geometry.dispose();
      m.material.map?.dispose();
      m.material.dispose();
    }
    this.meshes = [];
    this.model = model;
    const rgbaAll = model.geom_rgba, sizeAll = model.geom_size;
    for (let g = 0; g < model.ngeom; g++) {
      const type = model.geom_type[g];
      const size = [sizeAll[3 * g], sizeAll[3 * g + 1], sizeAll[3 * g + 2]];
      const geom = geometryFor(type, size);
      if (!geom) { this.meshes.push(null); continue; }
      let rgba = [rgbaAll[4 * g], rgbaAll[4 * g + 1], rgbaAll[4 * g + 2], rgbaAll[4 * g + 3]];
      let map = null;
      const mat = model.geom_matid[g];
      if (mat >= 0) {
        const isDefault = rgba[0] === 0.5 && rgba[1] === 0.5 && rgba[2] === 0.5 && rgba[3] === 1;
        if (isDefault) rgba = Array.from(model.mat_rgba.slice(4 * mat, 4 * mat + 4));
        const tex = model.mat_texid[NTEXROLE * mat + TEXROLE_RGB];
        if (tex >= 0) {
          // mat_texuniform is a bool array, which the WASM bindings cannot expose;
          // the exported floors all use texuniform="false" (repeat over the plane).
          map = textureFor(model, tex, [model.mat_texrepeat[2 * mat], model.mat_texrepeat[2 * mat + 1]]);
        }
      }
      const transparent = rgba[3] < 1;
      const material = new THREE.MeshLambertMaterial({
        color: new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace),
        map, transparent, opacity: rgba[3], depthWrite: !transparent,
      });
      const mesh = new THREE.Mesh(geom, material);
      mesh.matrixAutoUpdate = false;
      mesh.receiveShadow = true;
      mesh.castShadow = type !== PLANE && !transparent && size[2] !== 0.001;
      if (transparent) mesh.renderOrder = 1;
      this.scene.add(mesh);
      this.meshes.push(mesh);
    }
  }

  sync(data, camId) {
    const xpos = data.geom_xpos, xmat = data.geom_xmat;
    for (let g = 0; g < this.meshes.length; g++) {
      const mesh = this.meshes[g];
      if (!mesh) continue;
      const r = 9 * g, p = 3 * g;
      mesh.matrix.set(
        xmat[r], xmat[r + 1], xmat[r + 2], xpos[p],
        xmat[r + 3], xmat[r + 4], xmat[r + 5], xpos[p + 1],
        xmat[r + 6], xmat[r + 7], xmat[r + 8], xpos[p + 2],
        0, 0, 0, 1);
      mesh.matrixWorldNeedsUpdate = true;
    }
    // MuJoCo cameras look down -Z with +Y up, same convention as three.js.
    const cp = data.cam_xpos, cm = data.cam_xmat, r = 9 * camId, p = 3 * camId;
    this.camera.matrix.set(
      cm[r], cm[r + 1], cm[r + 2], cp[p],
      cm[r + 3], cm[r + 4], cm[r + 5], cp[p + 1],
      cm[r + 6], cm[r + 7], cm[r + 8], cp[p + 2],
      0, 0, 0, 1);
    this.camera.matrixWorldNeedsUpdate = true;
    const fovy = this.model.cam_fovy[camId];
    if (this.camera.fov !== fovy) { this.camera.fov = fovy; this.camera.updateProjectionMatrix(); }
    this.headlight.position.set(cp[p], cp[p + 1], cp[p + 2]);
    this.headlight.target.position.set(cp[p] - cm[r + 2], cp[p + 1] - cm[r + 5], cp[p + 2] - cm[r + 8]);
    this.headlight.target.updateMatrixWorld();
  }

  resize(w, h) {
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render() { this.renderer.render(this.scene, this.camera); }
}

// Keyboard / touch state -> point-agent action [thrust, turn], as in play.py.
class Controls {
  constructor(root) {
    this.keys = new Set();
    this.touch = { thrust: 0, turn: 0 };
    root.addEventListener('keydown', (e) => {
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) e.preventDefault();
      this.keys.add(e.code);
    });
    root.addEventListener('keyup', (e) => this.keys.delete(e.code));
    root.addEventListener('blur', () => this.keys.clear());
  }

  axis(pos, neg) {
    return (pos.some((k) => this.keys.has(k)) ? 1 : 0) - (neg.some((k) => this.keys.has(k)) ? 1 : 0);
  }

  action() {
    const thrust = this.axis(['KeyW', 'ArrowUp'], ['KeyS', 'ArrowDown']) || this.touch.thrust;
    const turn = this.axis(['KeyA', 'ArrowLeft'], ['KeyD', 'ArrowRight']) || this.touch.turn;
    return [thrust, turn];
  }
}

export class Player {
  constructor(root) {
    this.root = root;
    this.$ = (sel) => root.querySelector(sel);
    this.canvas = this.$('canvas');
    this.envName = root.dataset.env || 'safe_goal_point';
    this.level = Number(root.dataset.level || 1);
    this.camIdx = 0;
    this.paused = false;
    this.cache = new Map();
    this.bindUi();
  }

  bindUi() {
    this.$('[data-action=start]').addEventListener('click', () => this.start());
    this.root.querySelectorAll('[data-env-choice]').forEach((b) => b.addEventListener('click', () => {
      this.envName = b.dataset.envChoice;
      this.switchEnv();
    }));
    this.root.querySelectorAll('[data-level-choice]').forEach((b) => b.addEventListener('click', () => {
      this.level = Number(b.dataset.levelChoice);
      this.switchEnv();
    }));
    this.$('[data-action=camera]').addEventListener('click', () => this.cycleCamera());
    this.$('[data-action=reset]').addEventListener('click', () => this.newEpisode());
    this.$('[data-action=pause]').addEventListener('click', () => this.togglePause());
    this.stage = this.$('.player-stage');
    this.stage.addEventListener('keydown', (e) => {
      if (e.code === 'KeyC') this.cycleCamera();
      else if (e.code === 'KeyR') this.newEpisode();
      else if (e.code === 'Space') this.togglePause();
    });
    this.stage.addEventListener('focus', () => this.root.classList.add('is-focused'));
    this.stage.addEventListener('blur', () => this.root.classList.remove('is-focused'));
    this.stage.addEventListener('pointerdown', () => this.stage.focus());
    this.controls = new Controls(this.stage);
    // Touch pad: hold buttons to thrust / turn.
    this.root.querySelectorAll('[data-pad]').forEach((b) => {
      const [axis, value] = b.dataset.pad.split(':');
      const set = (v) => (e) => { e.preventDefault(); this.controls.touch[axis] = v; };
      b.addEventListener('pointerdown', set(Number(value)));
      ['pointerup', 'pointerleave', 'pointercancel'].forEach((ev) => b.addEventListener(ev, set(0)));
    });
    this.syncButtons();
  }

  async start() {
    this.root.classList.add('is-loading');
    this.setStatus(`Loading MuJoCo ${MUJOCO_VERSION} (WebAssembly)…`);
    try {
      const { default: loadMujoco } = await import(MUJOCO_URL);
      this.mujoco = await loadMujoco();
      this.view = new SceneView(this.canvas);
      new ResizeObserver(() => this.onResize()).observe(this.stage);
      this.onResize();
      await this.switchEnv();
      this.root.classList.remove('is-loading');
      this.root.classList.add('is-running');
      this.stage.focus();
      this.last = performance.now();
      requestAnimationFrame((t) => this.frame(t));
    } catch (err) {
      console.error(err);
      this.root.classList.remove('is-loading');
      this.setStatus(`Could not start the simulator: ${err.message}`);
      this.root.classList.add('is-error');
    }
  }

  setStatus(text) { this.$('.player-status').textContent = text; }

  onResize() {
    const { width, height } = this.stage.getBoundingClientRect();
    this.view?.resize(Math.max(1, width), Math.max(1, height));
  }

  syncButtons() {
    this.root.querySelectorAll('[data-env-choice]').forEach((b) => b.setAttribute('aria-pressed', b.dataset.envChoice === this.envName));
    this.root.querySelectorAll('[data-level-choice]').forEach((b) => b.setAttribute('aria-pressed', Number(b.dataset.levelChoice) === this.level));
    this.root.querySelectorAll('[data-env-info]').forEach((el) => { el.hidden = el.dataset.envInfo !== this.envName; });
  }

  async switchEnv() {
    this.syncButtons();
    if (!this.mujoco) return;
    const file = `${this.envName}_level${this.level}`;
    const token = (this.loadToken = Symbol(file));
    let env = this.cache.get(file);
    if (!env) {
      this.root.classList.add('is-switching');
      env = await loadEnv(this.mujoco, ENV_DIR, file);
      this.cache.set(file, env);
      this.root.classList.remove('is-switching');
    }
    if (token !== this.loadToken) return;
    this.env = null;
    this.view.setModel(env.model);
    this.cameras = CAMERAS
      .map((name) => [name, env.spec.cameras.indexOf(name)])
      .filter(([, id]) => id >= 0);
    this.camIdx = Math.min(this.camIdx, this.cameras.length - 1);
    this.env = env;
    this.episode = 0;
    this.newEpisode();
    this.stage.focus();
  }

  newEpisode() {
    if (!this.env) return;
    this.env.reset();
    this.episode += 1;
    Object.assign(this, { steps: 0, substep: 0, epReturn: 0, epCost: 0, stepCost: 0, goals: 0, action: [0, 0] });
    this.flash = 0;
    this.updateHud();
  }

  cycleCamera() {
    if (!this.cameras) return;
    this.camIdx = (this.camIdx + 1) % this.cameras.length;
    this.updateHud();
  }

  togglePause() {
    this.paused = !this.paused;
    this.root.classList.toggle('is-paused', this.paused);
    this.$('[data-action=pause]').textContent = this.paused ? 'Resume' : 'Pause';
    this.updateHud();
  }

  // Physics advances one MuJoCo step (opt.timestep) at a time, so the scene is
  // drawn at 50 Hz; the action is held for the n_frames sub-steps of an env step.
  frame(now) {
    requestAnimationFrame((t) => this.frame(t));
    const env = this.env;
    if (!env) return;
    const h = env.dt / env.nFrames;
    const budget = Math.min((now - this.last) / 1000, 0.1);
    this.last = now;
    if (!this.paused) {
      this.acc = (this.acc || 0) + budget;
      while (this.acc >= h) {
        this.acc -= h;
        if (this.substep === 0) {
          this.action = this.controls.action();
          env.beginStep();
        }
        env.physicsSubstep(this.action);
        this.substep += 1;
        if (this.substep === env.nFrames) {
          this.substep = 0;
          const r = env.finishStep(this.action);
          this.steps += 1;
          this.epReturn += r.reward;
          this.epCost += r.cost;
          this.stepCost = r.cost;
          this.goals += r.goalsReached;
          if (r.cost > 0) this.flash = 1;
          if (r.done || this.steps >= MAX_STEPS) { this.newEpisode(); break; }
        }
      }
      this.updateHud();
    }
    this.flash *= 0.9;
    this.$('.player-flash').style.opacity = (0.55 * this.flash).toFixed(3);
    this.view.sync(env.displayPose(), this.cameras[this.camIdx][1]);
    this.view.render();
  }

  updateHud() {
    const set = (k, v) => { this.$(`[data-hud=${k}]`).textContent = v; };
    set('episode', this.episode);
    set('steps', `${this.steps} / ${MAX_STEPS}`);
    set('return', this.epReturn.toFixed(2));
    set('cost', this.epCost.toFixed(2));
    set('goals', this.goals);
    set('camera', this.cameras ? this.cameras[this.camIdx][0] : '–');
    set('action', `[${this.action.map((a) => (a > 0 ? '+' : a < 0 ? '−' : ' ') + Math.abs(a)).join(', ')}]`);
    this.root.classList.toggle('is-costly', this.stepCost > 0);
  }
}

document.querySelectorAll('[data-crax-player]').forEach((el) => new Player(el));
