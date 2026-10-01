// JavaScript port of the CRAX point-agent tasks (Goal, Push, Circle, Button).
//
// Physics runs on the MuJoCo WebAssembly bindings instead of MJX; the task
// logic (layout sampling, rewards, costs, goal respawn) mirrors
// crax/envs/safe_goal.py, safe_push.py, safe_circle.py and safe_button.py. Models and task
// parameters are exported by scripts/export_web_envs.py.

// Must match the MuJoCo version that wrote the .mjb files (see <env>.json).
export const MUJOCO_VERSION = '3.11.0';

const EPS = 1e-8;
const uniform = (lo, hi) => lo + (hi - lo) * Math.random();
const hypot2 = (dx, dy) => Math.sqrt(dx * dx + dy * dy + EPS);

function sdfCylinder(p, c, r) {
  return Math.hypot(p[0] - c[0], p[1] - c[1]) - r;
}

function sdfCube(p, c, he, yaw) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const dx = p[0] - c[0], dy = p[1] - c[1];
  // (p - c) @ R, with R = [[cos, -sin], [sin, cos]]
  const lx = dx * cy + dy * sy, ly = -dx * sy + dy * cy;
  const qx = Math.abs(lx) - he[0], qy = Math.abs(ly) - he[1];
  const outside = Math.hypot(Math.max(qx, 0), Math.max(qy, 0));
  return outside + Math.min(Math.max(qx, qy), 0);
}

// env_utils.choose_valid_position: best-of-K candidates, nudged if still invalid.
function chooseValidPosition(placed, keepout, numCandidates, extents, margin) {
  const [minx, miny, maxx, maxy] = extents;
  let best = null, bestSlack = -Infinity;
  for (let k = 0; k < numCandidates; k++) {
    const x = uniform(minx + keepout, maxx - keepout);
    const y = uniform(miny + keepout, maxy - keepout);
    let worst = 1e9;
    for (const o of placed) {
      worst = Math.min(worst, hypot2(x - o.x, y - o.y) - (o.keepout + keepout + margin));
    }
    if (worst > bestSlack) { bestSlack = worst; best = [x, y]; }
  }
  if (bestSlack < 0) {
    let nearest = null, nd = Infinity;
    for (const o of placed) {
      const d = hypot2(best[0] - o.x, best[1] - o.y);
      if (d < nd) { nd = d; nearest = o; }
    }
    const req = nearest.keepout + keepout + margin;
    const dx = best[0] - nearest.x, dy = best[1] - nearest.y;
    const n = Math.hypot(dx, dy) + EPS;
    const push = req - nd + 1e-3;
    best = [
      Math.min(Math.max(best[0] + push * dx / n, minx + keepout), maxx - keepout),
      Math.min(Math.max(best[1] + push * dy / n, miny + keepout), maxy - keepout),
    ];
  }
  return [best[0], best[1], 0.09];
}

function placeObjects(placed, keepouts, numCandidates, extents, margin) {
  return keepouts.map((keepout) => {
    const pos = chooseValidPosition(placed, keepout, numCandidates, extents, margin);
    placed.push({ x: pos[0], y: pos[1], keepout });
    return pos;
  });
}

// env_utils.choose_valid_position_shape_aware: rejection sampling against
// circle / rectangle keepouts, used when a goal respawns mid-episode.
function chooseValidPositionShapeAware(objects, radius, attempts, halfExt, margin, fallback) {
  const [ex, ey] = halfExt;
  for (let i = 0; i < attempts; i++) {
    const x = uniform(-ex + margin, ex - margin);
    const y = uniform(-ey + margin, ey - margin);
    const ok = objects.every((o) => {
      const dx = Math.abs(o.x - x), dy = Math.abs(o.y - y);
      if (o.isRect) return !(dx <= o.he[0] + radius && dy <= o.he[1] + radius);
      return !(Math.hypot(dx, dy) < o.r + radius);
    });
    if (ok) return [x, y, 0.0];
  }
  return [fallback[0], fallback[1], 0.0];
}

// SafeButton._fixed_clearances: signed clearance to a fixed rect / circle keepout.
function fixedClearance(x, y, keepout, fixed, margin) {
  let best = Infinity;
  for (const f of fixed) {
    const dx = Math.abs(x - f.x), dy = Math.abs(y - f.y);
    let d;
    if (f.isRect) {
      const qx = dx - f.he[0], qy = dy - f.he[1];
      d = Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0);
    } else {
      d = Math.hypot(dx, dy) - f.r;
    }
    best = Math.min(best, d - keepout - margin);
  }
  return best;
}

// SafeButton._sample_layout: place objects largest-keepout first, each at the
// best of K candidates, and retry the whole layout until every slack is >= 0.
function sampleButtonLayout(spec, keepouts, agentXY, fixed) {
  const order = keepouts.map((k, i) => [k, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).map(([, i]) => i);
  const [minx, miny, maxx, maxy] = spec.placement_extents, m = spec.placement_margin;
  let positions = null;
  for (let attempt = 0; attempt < spec.max_layout_attempts; attempt++) {
    positions = new Array(keepouts.length);
    let valid = true;
    order.forEach((idx, i) => {
      const r = keepouts[idx];
      let best = null, bestSlack = -Infinity;
      for (let k = 0; k < spec.max_placement_attempts; k++) {
        const x = uniform(minx + r, maxx - r), y = uniform(miny + r, maxy - r);
        let slack = Math.hypot(x - agentXY[0], y - agentXY[1]) - r - spec.agent_keepout - m;
        for (let j = 0; j < i; j++) {
          const p = positions[order[j]];
          slack = Math.min(slack, Math.hypot(x - p[0], y - p[1]) - r - keepouts[order[j]] - m);
        }
        if (fixed.length) slack = Math.min(slack, fixedClearance(x, y, r, fixed, m));
        if (slack > bestSlack) { bestSlack = slack; best = [x, y]; }
      }
      positions[idx] = best;
      valid = valid && bestSlack >= 0;
    });
    if (valid) break;
  }
  return positions;
}

export class CraxEnv {
  constructor(mujoco, model, spec) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = new mujoco.MjData(model);
    this.spec = spec;
    this.task = spec.task;
    this.dt = spec.dt;
    this.nFrames = spec.n_frames;
    this.agentGeoms = new Set(spec.agent_geom_ids);
  }

  get agentPos() {
    const b = this.spec.agent_body, xpos = this.data.xpos;
    return [xpos[3 * b], xpos[3 * b + 1], xpos[3 * b + 2]];
  }

  bodyPos(b) {
    const xpos = this.data.xpos;
    return [xpos[3 * b], xpos[3 * b + 1], xpos[3 * b + 2]];
  }

  setMocap(id, pos) {
    const m = this.data.mocap_pos;
    m[3 * id] = pos[0]; m[3 * id + 1] = pos[1]; m[3 * id + 2] = pos[2];
  }

  mocap(id) {
    const m = this.data.mocap_pos;
    return [m[3 * id], m[3 * id + 1], m[3 * id + 2]];
  }

  // ------------------------------------------------------------------ reset
  reset() {
    const { mujoco, model, data, spec } = this;
    mujoco.mj_resetData(model, data);
    const noise = spec.reset_noise_scale;
    const qpos = data.qpos, qvel = data.qvel, qpos0 = model.qpos0;
    for (let i = 0; i < model.nq; i++) qpos[i] = qpos0[i] + uniform(-noise, noise);
    for (let i = 0; i < model.nv; i++) qvel[i] = uniform(-noise, noise);
    if (this.task === 'circle') {
      qpos[0] = uniform(-spec.init_xy_range, spec.init_xy_range);
      qpos[1] = uniform(-spec.init_xy_range, spec.init_xy_range);
      qpos[2] = uniform(-spec.init_angle_range, spec.init_angle_range);
    }
    mujoco.mj_forward(model, data);

    const agent = this.agentPos;
    const placed = [{ x: agent[0], y: agent[1], keepout: spec.agent_keepout }];
    const movable = spec.hazards.slice(0, spec.num_movable_hazards);
    const K = spec.max_placement_attempts, margin = spec.placement_margin;

    if (this.task === 'button') {
      this.resetButton(agent);
    } else if (this.task === 'circle') {
      spec.goal_mocap_ids.forEach((id, i) => this.setMocap(id, spec.goal_positions[i]));
      const pos = placeObjects(placed, movable.map((h) => h.keepout), K, spec.hazard_placement_extents, margin);
      movable.forEach((h, i) => this.setMocap(h.mocap_id, pos[i]));
    } else {
      const ext = spec.placement_extents;
      this.goalPositions = placeObjects(placed, spec.goal_keepouts, K, ext, margin);
      const hpos = placeObjects(placed, movable.map((h) => h.keepout), K, ext, margin);
      spec.goal_mocap_ids.forEach((id, i) => this.setMocap(id, this.goalPositions[i]));
      movable.forEach((h, i) => this.setMocap(h.mocap_id, hpos[i]));
    }
    if (this.task !== 'button') this.hazardPositions = spec.hazards.map((h) => this.mocap(h.mocap_id));
    mujoco.mj_forward(model, data);

    if (this.task === 'goal') {
      const a = this.agentPos;
      this.lastDistGoal = Math.min(...this.goalPositions.map((g) => hypot2(g[0] - a[0], g[1] - a[1])));
    } else if (this.task === 'push') {
      const a = this.agentPos, b = this.bodyPos(spec.block_body);
      this.lastDistGoal = Math.min(...this.goalPositions.map((g) => hypot2(g[0] - b[0], g[1] - b[1])));
      this.lastDistBlock = hypot2(a[0] - b[0], a[1] - b[1]);
      this.goalDirections = this.goalPositions.map(() => {
        const t = uniform(0, 2 * Math.PI);
        return [Math.cos(t), Math.sin(t)];
      });
    }
    this.stepCount = 0;
    return { reward: 0, cost: 0, done: false, goalsReached: 0 };
  }

  resetButton(agent) {
    const s = this.spec;
    const movable = s.hazards.map((h, i) => i).filter((i) => !s.hazards[i].fixed);
    const fixed = s.hazards.map((h, i) => i).filter((i) => s.hazards[i].fixed).map((i) => {
      const p = this.mocap(s.hazards[i].mocap_id);
      return { x: p[0], y: p[1], isRect: s.hazard_is_rect[i], he: s.hazard_half_extents[i], r: s.hazard_radii[i] };
    });
    const nb = s.button_mocap_ids.length;
    const keepouts = [...Array(nb).fill(s.button_keepout), ...movable.map((i) => s.hazards[i].keepout)];
    const layout = sampleButtonLayout(s, keepouts, agent, fixed);
    this.buttonPositions = layout.slice(0, nb).map(([x, y]) => [x, y, s.button_height]);
    s.button_mocap_ids.forEach((id, i) => this.setMocap(id, this.buttonPositions[i]));
    movable.forEach((hi, k) => {
      const z = this.mocap(s.hazards[hi].mocap_id)[2];
      this.setMocap(s.hazards[hi].mocap_id, [layout[nb + k][0], layout[nb + k][1], z]);
    });
    this.hazardPositions = s.hazards.map((h) => this.mocap(h.mocap_id));
    this.gremlins = s.hazards.map((h, i) => i).filter((i) => s.hazards[i].type === 'gremlin')
      .map((i) => ({ i, center: [...this.hazardPositions[i]], travel: s.hazards[i].travel }));
    this.moveGremlins(0);
    this.activeButton = Math.floor(Math.random() * nb);
    this.buttonTimer = 0;
    const b = this.buttonPositions[this.activeButton];
    this.lastDistGoal = hypot2(b[0] - agent[0], b[1] - agent[1]);
    this.placeMarker();
  }

  // env_utils.compute_gremlin_positions: circular orbit around the centre.
  moveGremlins(stepCount) {
    const phase = stepCount * this.dt;
    for (const g of this.gremlins) {
      const p = [g.center[0] + g.travel * Math.sin(phase), g.center[1] + g.travel * Math.cos(phase), g.center[2]];
      this.hazardPositions[g.i] = p;
      this.setMocap(this.spec.hazards[g.i].mocap_id, p);
    }
  }

  placeMarker() {
    const b = this.buttonPositions[this.activeButton];
    this.setMocap(this.spec.goal_marker_mocap_id, [b[0], b[1], this.spec.marker_z]);
  }

  // Called before the first physics sub-step of every env step.
  beginStep() {
    if (this.task === 'button') this.moveGremlins(this.stepCount + 1);
  }

  // Sub-stepping lets the page render every physics frame (50 Hz) while the
  // task logic still runs once per env step, exactly as in `pipeline_step`.
  physicsSubstep(action) {
    const ctrl = this.data.ctrl;
    for (let i = 0; i < action.length; i++) ctrl[i] = action[i];
    this.mujoco.mj_step(this.model, this.data);
  }

  // Task logic after `n_frames` physics sub-steps.
  finishStep(action) {
    let out;
    if (this.task === 'goal') out = this.goalStep();
    else if (this.task === 'push') out = this.pushStep();
    else if (this.task === 'button') out = this.buttonStep();
    else out = this.circleStep();
    const z = this.agentPos[2];
    const [zmin, zmax] = this.spec.healthy_z_range;
    out.done = !(z >= zmin && z <= zmax) || Number.isNaN(z) || Boolean(out.doneGoal);
    this.stepCount += 1;
    return out;
  }

  // ------------------------------------------------------------------ costs
  // Geom ids the agent touches (contact dist <= 0), as in compute_hazard_costs.
  agentContacts() {
    const touching = new Set();
    const contacts = this.data.contact;
    try {
      const n = Math.min(this.data.ncon, contacts.size());
      for (let i = 0; i < n; i++) {
        const c = contacts.get(i);
        if (c.dist <= 0) {
          if (this.agentGeoms.has(c.geom1)) touching.add(c.geom2);
          if (this.agentGeoms.has(c.geom2)) touching.add(c.geom1);
        }
        c.delete();
      }
    } finally {
      contacts.delete();
    }
    return touching;
  }

  hazardCost(touching = this.agentContacts()) {
    const { spec } = this;
    const a = this.agentPos;
    let total = 0;
    spec.hazards.forEach((h, i) => {
      if (h.collidable) {
        if (touching.has(h.geom_id)) total += spec.collision_cost;
        return;
      }
      const p = this.hazardPositions[i];
      let prox;
      if (h.type === 'rect') {
        prox = Math.abs(a[0] - p[0]) <= h.size[0] && Math.abs(a[1] - p[1]) <= h.size[1] ? 1 : 0;
      } else {
        const reach = h.type === 'gremlin' ? h.size[0] + h.travel : h.size[0];
        prox = Math.max(0, 1 - hypot2(a[0] - p[0], a[1] - p[1]) / reach);
      }
      total += spec.proximity_cost_scaler * prox;
    });
    return total;
  }

  goalSdf(p) {
    const s = this.spec;
    return this.goalPositions.map((g, i) => (s.goal_type_ids[i] === 0
      ? sdfCube(p, g, s.goal_box_he[i], s.goal_yaws[i])
      : sdfCylinder(p, g, s.goal_radii[i])));
  }

  respawnGoals(reached, agentXY) {
    const s = this.spec;
    const objects = [{ x: agentXY[0], y: agentXY[1], isRect: false, r: s.agent_keepout }];
    this.hazardPositions.forEach((p, i) => objects.push({
      x: p[0], y: p[1], isRect: s.hazard_is_rect[i], he: s.hazard_half_extents[i], r: s.hazard_radii[i],
    }));
    const goalSlot = objects.length;
    this.goalPositions.forEach((g, i) => objects.push({ x: g[0], y: g[1], isRect: false, r: s.goal_keepouts[i] }));
    const [minx, miny, maxx, maxy] = s.placement_extents;
    const halfExt = [(maxx - minx) / 2, (maxy - miny) / 2];
    reached.forEach((hit, i) => {
      if (!hit) return;
      const slot = objects[goalSlot + i];
      slot.r = 0;
      const pos = chooseValidPositionShapeAware(objects, s.goal_keepouts[i], s.max_placement_attempts,
        halfExt, s.placement_margin, [objects[0].x, objects[0].y]);
      Object.assign(slot, { x: pos[0], y: pos[1], r: s.goal_keepouts[i] });
      this.goalPositions[i] = pos;
      this.setMocap(s.goal_mocap_ids[i], pos);
      if (this.goalDirections) {
        const t = uniform(0, 2 * Math.PI);
        this.goalDirections[i] = [Math.cos(t), Math.sin(t)];
      }
    });
  }

  // ------------------------------------------------------------ safe_goal
  goalStep() {
    const s = this.spec, a = this.agentPos;
    const sdf = this.goalSdf(a);
    const reached = sdf.map((d) => d <= 0);
    const n = reached.filter(Boolean).length;
    const distGoal = Math.min(...sdf.map((d) => Math.max(d, 0)));
    const reward = (this.lastDistGoal - distGoal) * s.reward_distance + s.reward_goal * n;
    this.respawnGoals(reached, a);
    this.lastDistGoal = distGoal;
    return { reward, cost: this.hazardCost(), goalsReached: n };
  }

  // ------------------------------------------------------------ safe_push
  pushStep() {
    const s = this.spec, a = this.agentPos, b = this.bodyPos(s.block_body);
    const distBefore = Math.min(...this.goalPositions.map((g) => hypot2(g[0] - b[0], g[1] - b[1])));
    const distReward = (this.lastDistGoal - distBefore) * s.reward_distance;
    const distBlock = hypot2(a[0] - b[0], a[1] - b[1]);
    const blockReward = (this.lastDistBlock - distBlock) * s.reward_agent_block;

    if (s.goal_velocity > 0) {
      const [minx, miny, maxx, maxy] = s.placement_extents, m = s.placement_margin;
      this.goalPositions = this.goalPositions.map((g, i) => {
        const d = this.goalDirections[i];
        const x = g[0] + d[0] * s.goal_velocity * this.dt;
        const y = g[1] + d[1] * s.goal_velocity * this.dt;
        if (x < minx + m || x > maxx - m || y < miny + m || y > maxy - m) {
          const t = uniform(0, 2 * Math.PI);
          this.goalDirections[i] = [Math.cos(t), Math.sin(t)];
        }
        return [Math.min(Math.max(x, minx + m), maxx - m), Math.min(Math.max(y, miny + m), maxy - m), g[2]];
      });
      s.goal_mocap_ids.forEach((id, i) => this.setMocap(id, this.goalPositions[i]));
    }

    const reached = this.goalSdf(b).map((d) => d <= 0);
    const n = reached.filter(Boolean).length;
    this.respawnGoals(reached, a);
    this.lastDistGoal = Math.min(...this.goalPositions.map((g) => hypot2(g[0] - b[0], g[1] - b[1])));
    this.lastDistBlock = distBlock;
    return { reward: distReward + blockReward + s.reward_goal * n, cost: this.hazardCost(), goalsReached: n };
  }

  // ---------------------------------------------------------- safe_button
  buttonStep() {
    const s = this.spec, a = this.agentPos;
    const touching = this.agentContacts();
    const pressed = s.button_geom_ids.map((g) => touching.has(g));
    const achieved = pressed[this.activeButton];
    const wrong = s.buttons_constrained && pressed.some((p, i) => p && i !== this.activeButton);

    const b = this.buttonPositions[this.activeButton];
    const distGoal = hypot2(b[0] - a[0], b[1] - a[1]);
    const reward = (this.lastDistGoal - distGoal) * s.reward_distance + (achieved ? s.reward_goal : 0);
    const wrongCost = wrong && this.buttonTimer === 0 ? s.wrong_button_cost : 0;
    const cost = this.hazardCost(touching) + wrongCost;

    if (achieved && s.continue_goal) {
      this.activeButton = Math.floor(Math.random() * this.buttonPositions.length);
      this.buttonTimer = s.resampling_delay;
      const nb = this.buttonPositions[this.activeButton];
      this.lastDistGoal = hypot2(nb[0] - a[0], nb[1] - a[1]);
    } else {
      this.buttonTimer = Math.max(0, this.buttonTimer - 1);
      this.lastDistGoal = distGoal;
    }
    this.placeMarker();
    return { reward, cost, goalsReached: achieved ? 1 : 0, doneGoal: achieved && !s.continue_goal };
  }

  // ---------------------------------------------------------- safe_circle
  circleStep() {
    const s = this.spec, a = this.agentPos, qvel = this.data.qvel;
    const dx = a[0] - s.circle_center[0], dy = a[1] - s.circle_center[1];
    const radius = hypot2(dx, dy);
    const radialError = Math.abs(radius - s.circle_radius);
    const tangentVel = (-qvel[0] * dy + qvel[1] * dx) / radius;
    const reward = tangentVel / (1 + radialError) * s.reward_factor;

    let out = false;
    if (s.boundary_x !== null && Math.abs(dx) > s.boundary_x) out = true;
    if (s.boundary_y !== null && Math.abs(dy) > s.boundary_y) out = true;
    const cost = this.hazardCost() + (out ? s.boundary_cost : 0);
    return { reward, cost, goalsReached: 0 };
  }

  dispose() {
    this.data.delete();
    this.model.delete();
  }
}

// Loads `<file>.mjb` + `<file>.json` from `baseUrl` into a ready-to-use env.
export async function loadEnv(mujoco, baseUrl, file) {
  const [spec, buf] = await Promise.all([
    fetch(`${baseUrl}/${file}.json`).then((r) => r.json()),
    fetch(`${baseUrl}/${file}.mjb`).then((r) => r.arrayBuffer()),
  ]);
  const vfs = new mujoco.MjVFS();
  try {
    vfs.addBuffer(`${file}.mjb`, new Uint8Array(buf));
    const model = mujoco.MjModel.from_binary_path(`${file}.mjb`, vfs);
    return new CraxEnv(mujoco, model, spec);
  } finally {
    vfs.delete();
  }
}
