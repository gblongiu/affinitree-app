'use strict';

const study = JSON.parse(document.getElementById('research-data').textContent);
const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
const reducedMotion = motionQuery.matches;
const traitPalette = {
  NS: { color: '#FFA500', soft: '#ffd184', kind: 'TEMPERAMENT', description: 'Novelty Seeking describes tendencies toward exploration and responsiveness to unfamiliar experience.' },
  HA: { color: '#FF0000', soft: '#ff929f', kind: 'TEMPERAMENT', description: 'Harm Avoidance describes caution, anticipatory concern, and sensitivity to potential adverse outcomes.' },
  RD: { color: '#FF69B4', soft: '#ffabd1', kind: 'TEMPERAMENT', description: 'Reward Dependence describes sensitivity to interpersonal reinforcement and social feedback.' },
  P:  { color: '#8B4513', soft: '#e5ae88', kind: 'TEMPERAMENT', description: 'Persistence describes continued effort and consistency when tasks become difficult.' },
  SD: { color: '#0000FF', soft: '#9baeff', kind: 'CHARACTER', description: 'Self-Directedness describes goal orientation, responsibility, and self-regulation.' },
  CO: { color: '#4CAF50', soft: '#9becaa', kind: 'CHARACTER', description: 'Cooperativeness describes social acceptance, empathy, and collaborative orientation.' },
  ST: { color: '#7E2F94', soft: '#c6a0f5', kind: 'CHARACTER', description: 'Self-Transcendence describes a sense of connection beyond oneself and openness to broader meaning.' },
};
const traitNames = Object.fromEntries(study.traits.map(item => [item.code, item.name]));
const traitCodes = study.traits.map(item => item.code);
const $ = id => document.getElementById(id);
const $$ = selector => [...document.querySelectorAll(selector)];
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const svgNS = 'http://www.w3.org/2000/svg';

function element(tag, className, label) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (label !== undefined) node.textContent = label;
  return node;
}

function svgNode(tag, attrs = {}, label) {
  const node = document.createElementNS(svgNS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (label !== undefined) node.textContent = String(label);
  return node;
}

function seedRandom(initial) {
  let state = initial >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function correlationFormat(value) {
  return `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(3)}`;
}

function probabilityFormat(value) {
  return value < 0.0001 ? '< 0.0001' : value.toFixed(4);
}

function setActiveButton(selector, key, current) {
  $$(selector).forEach(button => {
    const active = button.dataset[key] === current;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
}

function rgba(hex, alpha) {
  const raw = hex.slice(1);
  const rgb = [0, 2, 4].map(i => parseInt(raw.slice(i, i + 2), 16));
  return `rgba(${rgb.join(',')},${alpha})`;
}

function attachScrollPresentation() {
  let pending = false;
  window.addEventListener('scroll', () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      const max = Math.max(1, document.documentElement.scrollHeight - innerHeight);
      $('scroll-progress').style.width = `${100 * scrollY / max}%`;
      pending = false;
    });
  }, { passive: true });

  const targets = $$('.chapter .section-intro, .chapter .stage, .chapter .metric-rail, .chapter .trait-layout, .chapter .shape-experiment, .chapter .feature-story, .chapter .cluster-layout, .chapter .method-grid, .chapter .future-panel');
  if (reducedMotion || !('IntersectionObserver' in window)) return;
  targets.forEach(target => target.classList.add('reveal'));
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: .065, rootMargin: '0px 0px 65px 0px' });
  targets.forEach(target => observer.observe(target));

  const chapterObserver = new IntersectionObserver(entries => {
    const active = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!active) return;
    $$('.nav-items a').forEach(link => link.classList.toggle('active', link.hash === `#${active.target.id}`));
  }, { rootMargin: '-24% 0px -54% 0px', threshold: [0, .2, .5] });
  $$('.chapter').forEach(chapter => chapterObserver.observe(chapter));
}

function drawHero() {
  const canvas = $('hero-canvas');
  if (!canvas) return;
  const context = canvas.getContext('2d');
  if (!context) return;

  const random = seedRandom(73719);
  const branches = traitCodes.map((code, i) => ({
    code,
    spread: (i - 3) / 3,
    sway: random() * Math.PI * 2,
    leaves: Array.from({ length: 5 }, () => random()),
  }));
  const motes = Array.from({ length: 65 }, () => ({
    x: random(), y: random(), radius: .5 + random() * 1.4,
    phase: random() * Math.PI * 2,
  }));
  let width = 1;
  let height = 1;
  let ratio = 1;
  let visible = true;
  let frameRequest = 0;
  let lastFrame = 0;

  function branchPoint(start, control1, control2, end, t) {
    const inverse = 1 - t;
    return {
      x: inverse ** 3 * start.x + 3 * inverse ** 2 * t * control1.x + 3 * inverse * t ** 2 * control2.x + t ** 3 * end.x,
      y: inverse ** 3 * start.y + 3 * inverse ** 2 * t * control1.y + 3 * inverse * t ** 2 * control2.y + t ** 3 * end.y,
    };
  }

  function drawLeaf(x, y, heading, length, color, alpha) {
    context.save();
    context.translate(x, y);
    context.rotate(heading);
    context.beginPath();
    context.moveTo(0, 0);
    context.quadraticCurveTo(length * .56, -length * .43, length, 0);
    context.quadraticCurveTo(length * .50, length * .38, 0, 0);
    context.fillStyle = rgba(color, alpha);
    context.fill();
    context.strokeStyle = rgba(color, Math.min(1, alpha + .20));
    context.lineWidth = .75;
    context.stroke();
    context.beginPath();
    context.moveTo(0, 0);
    context.lineTo(length * .88, 0);
    context.strokeStyle = rgba('#e4e8d6', .15);
    context.stroke();
    context.restore();
  }

  function paint(seconds) {
    context.clearRect(0, 0, width, height);
    const centerX = width / 2;
    const baseY = height * 1.08;
    const canopyY = height * .71;
    const scale = Math.min(width / 1040, height / 930, 1.25);
    const breadth = Math.min(width * .38, 480);

    // Growth rings anchor the illustration in a single observation.
    for (let i = 0; i < 11; i += 1) {
      const rx = Math.max(65, breadth * (.22 + i * .10));
      context.beginPath();
      context.ellipse(centerX, canopyY + height * .075, rx, rx * .36, -.04, Math.PI * .96, Math.PI * 2.04);
      context.strokeStyle = rgba('#bbbd9f', i % 3 === 0 ? .082 : .042);
      context.lineWidth = i % 3 === 0 ? 1.1 : .65;
      context.stroke();
    }
    // A small stipple pattern resembles ink impressions on archival paper.
    for (const mote of motes) {
      const x = centerX + (mote.x - .5) * breadth * 2.4;
      const y = height * (.54 + mote.y * .47);
      const pulse = reducedMotion ? 1 : .72 + .28 * Math.sin(seconds * .47 + mote.phase);
      context.beginPath();
      context.arc(x, y, mote.radius * scale, 0, Math.PI * 2);
      context.fillStyle = rgba('#e9dec5', .11 * pulse);
      context.fill();
    }

    // Seven branches retain the seven fixed TCI trait-color assignments.
    branches.forEach((branch, i) => {
      const trait = traitPalette[branch.code];
      const spread = branch.spread;
      const wave = reducedMotion ? 0 : Math.sin(seconds * .34 + branch.sway) * 5 * scale;
      const start = { x: centerX + spread * 7, y: baseY };
      const end = {
        x: centerX + spread * breadth * .97 + wave,
        y: height * (.665 + .065 * Math.abs(spread)) - (3 - Math.abs(i - 3)) * 10 * scale,
      };
      const control1 = { x: centerX - spread * breadth * .07, y: height * .855 };
      const control2 = { x: centerX + spread * breadth * .71 + wave, y: height * .785 };
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.bezierCurveTo(control1.x, control1.y, control2.x, control2.y, end.x, end.y);
      context.strokeStyle = rgba('#d9dcc3', .12);
      context.lineWidth = (4.5 - Math.abs(spread)) * scale;
      context.lineCap = 'round';
      context.stroke();
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.bezierCurveTo(control1.x, control1.y, control2.x, control2.y, end.x, end.y);
      context.strokeStyle = rgba(trait.soft, .25);
      context.lineWidth = (3.2 + (1 - Math.abs(spread)) * .8) * scale;
      context.stroke();
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.bezierCurveTo(control1.x, control1.y, control2.x, control2.y, end.x, end.y);
      context.strokeStyle = rgba(trait.color, .82);
      context.lineWidth = (1.7 + (1 - Math.abs(spread)) * .75) * scale;
      context.stroke();

      branch.leaves.forEach((leaf, j) => {
        const t = .36 + j * .123;
        const point = branchPoint(start, control1, control2, end, t);
        const next = branchPoint(start, control1, control2, end, Math.min(.995, t + .012));
        const tangent = Math.atan2(next.y - point.y, next.x - point.x);
        const direction = (j % 2 === 0 ? -1 : 1);
        const leafLength = (20 + leaf * 19) * scale;
        const angle = tangent + direction * (.75 + leaf * .35);
        drawLeaf(point.x, point.y, angle, leafLength, trait.color, .31 + .14 * (1 - t));
        if (j < 4) {
          drawLeaf(point.x, point.y, tangent - direction * .82, leafLength * .65, trait.soft, .24);
        }
      });
      context.beginPath();
      context.arc(end.x, end.y, 6.2 * scale, 0, Math.PI * 2);
      context.strokeStyle = rgba(trait.soft, .9);
      context.lineWidth = 1.6;
      context.stroke();
      context.beginPath();
      context.arc(end.x, end.y, 2.8 * scale, 0, Math.PI * 2);
      context.fillStyle = trait.color;
      context.fill();
    });

    // The trunk is a reference line, not an eighth trait.
    context.beginPath();
    context.moveTo(centerX, baseY);
    context.bezierCurveTo(centerX + 11, height * .95, centerX - 12, height * .85, centerX, height * .78);
    context.strokeStyle = rgba('#e7dac1', .23);
    context.lineWidth = 2.1 * scale;
    context.stroke();
    context.beginPath();
    context.arc(centerX, height * .78, 3.2 * scale, 0, Math.PI * 2);
    context.fillStyle = rgba('#efebd9', .74);
    context.fill();
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    width = Math.max(1, rect.width);
    height = Math.max(1, rect.height);
    ratio = Math.min(devicePixelRatio || 1, 1.75);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    paint(0);
  }

  function schedule() {
    if (!frameRequest && visible && !document.hidden && !reducedMotion) frameRequest = requestAnimationFrame(frame);
  }
  function frame(time) {
    frameRequest = 0;
    if (document.hidden || !visible || reducedMotion) return;
    if (time - lastFrame >= 40) {
      paint(time / 1000);
      lastFrame = time;
    }
    schedule();
  }

  let pendingResize;
  window.addEventListener('resize', () => {
    clearTimeout(pendingResize);
    pendingResize = setTimeout(resize, 140);
  });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) schedule();
      else if (frameRequest) { cancelAnimationFrame(frameRequest); frameRequest = 0; }
    }, { threshold: 0 }).observe(canvas);
  }
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && frameRequest) { cancelAnimationFrame(frameRequest); frameRequest = 0; }
    else schedule();
  });
  resize();
  schedule();
}

function drawShapeLab() {
  const target = $('shape-svg');
  const controls = $('shape-sliders');
  if (!target || !controls) return;

  // Synthetic, unit-scaled profiles illustrate distance without exposing records.
  const fixed = [.45, .24, .61, .77, .38, .69, .31];
  const initial = [.68, .49, .35, .55, .52, .34, .66];
  const values = [...initial];
  const center = { x: 265, y: 269 };
  const radius = 187;
  const angles = traitCodes.map((_, i) => -Math.PI / 2 + i * Math.PI * 2 / 7);
  const position = (i, magnitude) => ({
    x: center.x + Math.cos(angles[i]) * radius * magnitude,
    y: center.y + Math.sin(angles[i]) * radius * magnitude,
  });
  const polygon = valuesToRender => valuesToRender.map((value, i) => {
    const point = position(i, value);
    return `${i === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`;
  }).join(' ') + ' Z';

  const grid = svgNode('g', { 'aria-hidden': true });
  [0.25, 0.5, 0.75, 1].forEach(level => {
    grid.appendChild(svgNode('path', {
      d: polygon(Array(7).fill(level)), fill: 'none',
      stroke: '#d8d0b6', 'stroke-opacity': level === 1 ? .32 : .15,
      'stroke-dasharray': level === 1 ? '2 7' : 'none', 'stroke-width': level === 1 ? 1.1 : .9,
    }));
  });
  traitCodes.forEach((code, i) => {
    const end = position(i, 1);
    const label = position(i, 1.17);
    grid.appendChild(svgNode('line', {
      x1: center.x, y1: center.y, x2: end.x, y2: end.y,
      stroke: traitPalette[code].color, 'stroke-width': 1.5, 'stroke-opacity': .70,
    }));
    grid.appendChild(svgNode('circle', {
      cx: end.x, cy: end.y, r: 3.8, fill: traitPalette[code].color,
    }));
    grid.appendChild(svgNode('text', {
      x: label.x, y: label.y + 5, 'text-anchor': 'middle',
      fill: traitPalette[code].soft, 'font-size': 14, 'font-weight': 800,
    }, code));
  });
  const liveLayer = svgNode('g', { 'aria-hidden': true });
  target.append(grid, liveLayer);
  const labels = [];
  const sliders = [];

  traitCodes.forEach((code, i) => {
    const group = element('div', 'shape-slider-row');
    group.style.setProperty('--trait-color', traitPalette[code].color);
    const label = element('label', '', traitNames[code]);
    const input = element('input');
    input.type = 'range';
    input.min = '0'; input.max = '1'; input.step = '0.01';
    input.value = String(values[i]);
    input.id = `shape-${code}`;
    input.setAttribute('aria-label', `${traitNames[code]} variation`);
    label.htmlFor = input.id;
    const value = element('output', '', values[i].toFixed(2));
    value.htmlFor = input.id;
    group.append(label, input, value);
    controls.appendChild(group);
    labels.push(value); sliders.push(input);
    input.addEventListener('input', () => {
      values[i] = clamp(Number(input.value), 0, 1);
      update();
    });
  });

  function update() {
    liveLayer.replaceChildren();
    // Both outlines remain visible when the profiles coincide.
    liveLayer.appendChild(svgNode('path', {
      d: polygon(fixed), fill: '#dbe0cb', 'fill-opacity': .075,
      stroke: '#dce7ce', 'stroke-opacity': .76, 'stroke-width': 2.1,
      'stroke-dasharray': '5 7', 'stroke-linejoin': 'round',
    }));
    liveLayer.appendChild(svgNode('path', {
      d: polygon(values), fill: '#e2b67b', 'fill-opacity': .12,
      stroke: '#efd4a7', 'stroke-width': 2.3, 'stroke-linejoin': 'round',
    }));
    values.forEach((score, i) => {
      const point = position(i, score);
      const reference = position(i, fixed[i]);
      liveLayer.appendChild(svgNode('line', {
        x1: reference.x, y1: reference.y, x2: point.x, y2: point.y,
        stroke: traitPalette[traitCodes[i]].soft, 'stroke-width': 2.1, 'stroke-opacity': .68,
      }));
      liveLayer.appendChild(svgNode('circle', {
        cx: point.x, cy: point.y, r: 6,
        fill: traitPalette[traitCodes[i]].color,
        stroke: '#f1ebd8', 'stroke-width': 1.6,
      }));
      liveLayer.appendChild(svgNode('circle', {
        cx: reference.x, cy: reference.y, r: 2.4,
        fill: '#e4e9d8',
      }));
    });

    const squares = values.map((value, i) => (value - fixed[i]) ** 2);
    const sum = squares.reduce((a, b) => a + b, 0);
    const distance = Math.sqrt(sum);
    $('shape-distance-value').textContent = distance.toFixed(3);
    $('shape-distance-meter').style.width = `${100 * distance / Math.sqrt(7)}%`;
    values.forEach((value, i) => {
      const pct = 100 * (sum === 0 ? 0 : squares[i] / sum);
      const code = traitCodes[i];
      sliders[i].value = String(value);
      sliders[i].style.setProperty('--pct', `${value * 100}%`);
      labels[i].textContent = value.toFixed(2);
    });
    const contributions = $('shape-contributions');
    contributions.replaceChildren();
    traitCodes.forEach((code, i) => {
      const portion = sum === 0 ? 0 : 100 * squares[i] / sum;
      const row = element('div', 'shape-contributor');
      row.style.setProperty('--signal', traitPalette[code].color);
      const name = element('span', '', code);
      name.title = traitNames[code];
      const track = element('div', 'shape-contributor-track');
      const fill = element('i'); fill.style.width = `${portion}%`;
      track.appendChild(fill);
      row.append(name, track, element('b', '', `${portion.toFixed(0)}%`));
      contributions.appendChild(row);
    });
  }

  const setValues = next => {
    next.forEach((value, index) => { values[index] = value; });
    update();
  };
  $('shape-reset').addEventListener('click', () => setValues(initial));
  const match = $('shape-match');
  if (match) match.addEventListener('click', () => setValues(fixed));
  update();
}

function drawOrigin() {
  const target = $('origin-diagram');
  const slider = $('time-slider');
  const output = $('time-output');
  const labels = {
    snapshot: {
      index: 'AFFINITREE / 01', title: 'A position at one point in time.',
      description: 'A single questionnaire produces one profile. Nearby points suggest similarity under a chosen scoring and distance model, not an enduring classification of a person.',
      one: 'One submitted snapshot', two: 'Fixed relationships on the map', caption: 'FIXED OBSERVATION / CONCEPT ILLUSTRATION'
    },
    evolving: {
      index: 'RHIZOME / CONCEPT', title: 'A relationship that can change.',
      description: 'A future network could update positions as permissioned interactions and choices accumulate. Movement would represent changing evidence, not a permanent label.',
      one: 'Signals may evolve with time', two: 'Relationships require context and consent', caption: 'SIMULATED CHANGE / NO LONGITUDINAL DATA'
    }
  };
  const random = seedRandom(81041);
  const profiles = Array.from({ length: 44 }, (_, i) => ({
    id: i, color: traitPalette[traitCodes[i % 7]].soft,
    x: 70 + random() * 640, y: 75 + random() * 302, z: random() * 2 - 1,
    phase: random() * Math.PI * 2,
    speed: .6 + random() * 1.3,
  }));
  let era = 'snapshot', automatic = false, rafId = 0, lastTick = 0, visible = true;

  function coordinates(record, progress) {
    if (era === 'snapshot') return { x: record.x, y: record.y, z: 0 };
    const time = progress / 100;
    const angle = time * Math.PI * 2 * record.speed + record.phase;
    const movement = Math.min(.95, time * 1.5);
    const x = record.x + Math.sin(angle) * 56 * movement + Math.cos(time * 4 + record.phase) * 15;
    const y = record.y + Math.cos(angle * .75) * 50 * movement;
    const z = Math.sin(angle + record.phase) * movement;
    const perspective = 1 / (1 + z * .16);
    return { x: 390 + (x - 390) * perspective, y: 220 + (y - 220) * perspective, z };
  }

  function redraw() {
    target.replaceChildren();
    const progress = Number(slider.value);
    const positions = profiles.map(item => ({ ...item, ...coordinates(item, progress) }));
    const defs = svgNode('defs');
    const grad = svgNode('linearGradient', { id: 'origin-line', x1: '0%', x2: '100%' });
    grad.append(svgNode('stop', { offset: '0%', 'stop-color': '#9baeff', 'stop-opacity': '.70' }), svgNode('stop', { offset: '100%', 'stop-color': '#ffabd1', 'stop-opacity': '.55' }));
    defs.appendChild(grad); target.appendChild(defs);
    for (const r of [100, 170, 230, 290]) {
      target.appendChild(svgNode('ellipse', { cx: 390, cy: 225, rx: r * 1.13, ry: r * .66, stroke: '#a7a5f5', 'stroke-width': 1, 'stroke-opacity': .10, fill: 'none' }));
    }
    target.appendChild(svgNode('path', { d: 'M 32 225 H 748 M 390 23 V 432', stroke: '#8898cc', 'stroke-width': 1, 'stroke-opacity': .10, fill: 'none' }));
    positions.forEach((p, i) => {
      const nearest = positions.map((q, j) => ({ q, j, d: Math.hypot(q.x - p.x, q.y - p.y) })).filter(item => item.j !== i).sort((a, b) => a.d - b.d)[0];
      if (nearest && i < nearest.j) target.appendChild(svgNode('line', { x1: p.x, y1: p.y, x2: nearest.q.x, y2: nearest.q.y, stroke: era === 'snapshot' ? '#a4a8ed' : 'url(#origin-line)', 'stroke-opacity': era === 'snapshot' ? .18 : .36, 'stroke-width': era === 'snapshot' ? 1 : 1.3 }));
    });
    positions.sort((a, b) => a.z - b.z);
    positions.forEach(p => {
      target.appendChild(svgNode('circle', { cx: p.x, cy: p.y, r: era === 'snapshot' ? 11 : 12 + p.z * 3.5, fill: p.color, 'fill-opacity': .095, stroke: p.color, 'stroke-opacity': .23 }));
      target.appendChild(svgNode('circle', { cx: p.x, cy: p.y, r: era === 'snapshot' ? 3.6 : 3.8 + p.z, fill: p.color, opacity: .9 }));
    });
    target.appendChild(svgNode('text', { x: 32, y: 31, fill: '#90a3bf', 'font-size': 10, 'font-weight': 850, 'letter-spacing': 2 }, era === 'snapshot' ? 'SINGLE OBSERVATION' : 'ILLUSTRATIVE TIME-VARYING POSITIONS'));
    output.value = progress < 20 ? 'Beginning' : progress > 80 ? 'Later' : 'Changing';
  }

  function tick(now) {
    rafId = 0;
    if (!automatic || reducedMotion || !visible || document.hidden || era !== 'evolving') return;
    if (now - lastTick > 70) {
      slider.value = (Number(slider.value) + 1) % 101;
      redraw(); lastTick = now;
    }
    rafId = requestAnimationFrame(tick);
  }
  function play() { if (!automatic || reducedMotion || !visible || document.hidden || rafId) return; rafId = requestAnimationFrame(tick); }
  function stop() { if (rafId) cancelAnimationFrame(rafId); rafId = 0; }
  $$('.seg[data-era]').forEach(button => button.addEventListener('click', () => {
    era = button.dataset.era;
    setActiveButton('.seg[data-era]', 'era', era);
    const label = labels[era];
    $('era-index').textContent = label.index;
    $('era-title').textContent = label.title;
    $('era-description').textContent = label.description;
    $('era-rule-one').textContent = label.one;
    $('era-rule-two').textContent = label.two;
    $('era-caption').textContent = label.caption;
    $('timeline').hidden = era !== 'evolving';
    automatic = era === 'evolving';
    slider.value = 0;
    redraw(); stop(); play();
  }));
  slider.addEventListener('input', () => { automatic = false; stop(); redraw(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); else play(); });
  if ('IntersectionObserver' in window) new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (visible) play(); else stop();
  }, { threshold: 0 }).observe(target);
  redraw();
}

function drawTraitOrbit() {
  const target = $('trait-orbit');
  const selection = $('trait-select');
  const center = [280, 280], radius = 190;
  let active = 'NS';
  const spokes = new Map();
  const labels = new Map();
  for (const r of [92, 143, 193, 227]) target.appendChild(svgNode('circle', { cx: center[0], cy: center[1], r, fill: 'none', stroke: '#8999d9', 'stroke-width': '.8', 'stroke-opacity': r === 193 ? .25 : .14, 'stroke-dasharray': r === 227 ? '3 9' : 'none' }));
  traitCodes.forEach((code, i) => {
    const angle = -Math.PI / 2 + i / 7 * Math.PI * 2;
    const x = center[0] + Math.cos(angle) * radius, y = center[1] + Math.sin(angle) * radius;
    const outerX = center[0] + Math.cos(angle) * 247, outerY = center[1] + Math.sin(angle) * 247;
    const spoke = svgNode('line', { x1: center[0], y1: center[1], x2: x, y2: y, stroke: traitPalette[code].soft, 'stroke-width': 1.6, 'stroke-opacity': .62 });
    target.appendChild(spoke); spokes.set(code, spoke);
    const halo = svgNode('circle', { cx: x, cy: y, r: 26, fill: traitPalette[code].color, 'fill-opacity': .065, stroke: traitPalette[code].soft, 'stroke-opacity': .28 });
    const dot = svgNode('circle', { cx: x, cy: y, r: 8, fill: traitPalette[code].soft });
    const label = svgNode('text', { x: outerX, y: outerY + 5, fill: '#cad6ef', 'font-size': 14, 'font-weight': 900, 'text-anchor': 'middle' }, code);
    const group = svgNode('g', { role: 'button', tabindex: 0, 'aria-label': `${traitNames[code]}. Inspect this dimension.` });
    group.append(halo, dot, label);
    group.addEventListener('click', () => selectTrait(code));
    group.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectTrait(code); } });
    target.appendChild(group); labels.set(code, halo);
    const b = element('button', 'trait-button', code);
    b.type = 'button'; b.style.setProperty('--signal', traitPalette[code].soft); b.dataset.trait = code;
    b.setAttribute('aria-label', traitNames[code]); b.addEventListener('click', () => selectTrait(code));
    selection.appendChild(b);
  });
  target.appendChild(svgNode('circle', { cx: 280, cy: 280, r: 58, fill: '#0e152d', stroke: '#b4aedf', 'stroke-width': 1.5, 'stroke-opacity': .36 }));
  target.appendChild(svgNode('text', { x: 280, y: 282, 'text-anchor': 'middle', 'font-size': 50, 'font-weight': 900, 'letter-spacing': -3, fill: '#edf4ff' }, '7'));
  target.appendChild(svgNode('text', { x: 280, y: 309, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 900, 'letter-spacing': 1.5, fill: '#a6b7d0' }, 'TRAITS'));

  function selectTrait(code) {
    active = code;
    selection.querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.trait === code)));
    for (const [key, node] of spokes) { node.setAttribute('stroke-opacity', key === code ? '1' : '.27'); node.setAttribute('stroke-width', key === code ? '4' : '1.5'); }
    for (const [key, node] of labels) { node.setAttribute('fill-opacity', key === code ? '.21' : '.055'); node.setAttribute('stroke-opacity', key === code ? '.8' : '.22'); }
    const detail = $('trait-detail');
    detail.style.setProperty('--trait-color', traitPalette[code].soft);
    detail.replaceChildren(
      element('span', 'micro-title', `${traitPalette[code].kind} · ${study.traits.find(t => t.code === code).subscales} SUBSCALES`),
      element('h3', '', traitNames[code]),
      element('p', '', traitPalette[code].description),
      element('small', '', 'This axis contributes to the descriptive similarity calculation.')
    );
  }
  selectTrait(active);
}

function findCorrelation(left, right) {
  return study.correlations.find(test => (test.a === left && test.b === right) || (test.a === right && test.b === left));
}

function drawPairVisualization(left, right, sign) {
  const target = $('pair-viz');
  target.replaceChildren();
  const svg = svgNode('svg', { viewBox: '0 0 300 150', 'aria-hidden': true });
  const p1 = traitPalette[left].soft, p2 = traitPalette[right].soft;
  svg.appendChild(svgNode('path', { d: 'M 67 75 C 110 20, 191 130, 233 75', stroke: sign > 0 ? '#d0b5ff' : '#a5bddf', 'stroke-width': 2.7, 'stroke-dasharray': sign < 0 ? '8 9' : 'none', fill: 'none', opacity: .92 }));
  svg.append(svgNode('circle', { cx: 67, cy: 75, r: 41, fill: p1, 'fill-opacity': .09, stroke: p1, 'stroke-opacity': .25 }), svgNode('circle', { cx: 233, cy: 75, r: 41, fill: p2, 'fill-opacity': .09, stroke: p2, 'stroke-opacity': .25 }));
  svg.append(svgNode('circle', { cx: 67, cy: 75, r: 23, fill: '#11192e', stroke: p1, 'stroke-width': 2 }), svgNode('circle', { cx: 233, cy: 75, r: 23, fill: '#11192e', stroke: p2, 'stroke-width': 2 }));
  svg.append(svgNode('text', { x: 67, y: 81, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 900, fill: p1 }, left), svgNode('text', { x: 233, y: 81, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 900, fill: p2 }, right));
  svg.appendChild(svgNode('text', { x: 150, y: 145, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 850, 'letter-spacing': 1.4, fill: '#a2b2d0' }, sign < 0 ? 'INVERSE ASSOCIATION' : 'POSITIVE ASSOCIATION'));
  target.appendChild(svg);
}

function drawCorrelations() {
  const matrix = $('correlation-matrix');
  const traitKey = $('trait-key');
  matrix.appendChild(element('div', 'matrix-label', ''));
  for (const code of traitCodes) {
    const label = element('div', 'matrix-label', code);
    label.style.setProperty('--signal', traitPalette[code].soft);
    label.style.setProperty('--soft-text', traitPalette[code].soft);
    matrix.appendChild(label);
    const key = element('span', '', code + ' ' + traitNames[code]);
    const dot = element('i'); dot.style.setProperty('--signal', traitPalette[code].soft);
    key.prepend(dot); traitKey.appendChild(key);
  }

  for (const [rowIndex, rowCode] of traitCodes.entries()) {
    const label = element('div', 'matrix-label', rowCode);
    label.style.setProperty('--signal', traitPalette[rowCode].soft);
    label.style.setProperty('--soft-text', traitPalette[rowCode].soft);
    matrix.appendChild(label);
    for (const [columnIndex, columnCode] of traitCodes.entries()) {
      const r = study.correlation_matrix[rowIndex][columnIndex];
      const diagonal = rowIndex === columnIndex;
      const cell = element(diagonal ? 'div' : 'button', 'heat-cell' + (diagonal ? ' diagonal' : ''), diagonal ? '1.00' : `${r < 0 ? '−' : '+'}${Math.abs(r).toFixed(2)}`);
      cell.style.setProperty('--fill', `rgba(${r < 0 ? '107,112,169' : '126,87,172'},${(.08 + Math.abs(r) * .58).toFixed(2)})`);
      if (diagonal) {
        cell.style.setProperty('--signal', traitPalette[rowCode].soft);
        cell.style.setProperty('--soft-text', traitPalette[rowCode].soft);
      } else {
        cell.type = 'button';
        cell.dataset.left = rowCode; cell.dataset.right = columnCode;
        cell.setAttribute('aria-label', `${traitNames[rowCode]} and ${traitNames[columnCode]}. Pearson ${correlationFormat(r)}`);
        cell.setAttribute('aria-pressed', 'false');
        cell.addEventListener('click', () => describePair(findCorrelation(rowCode, columnCode)));
      }
      matrix.appendChild(cell);
    }
  }
  describePair(study.strongest_pairs[0]);
  $('strongest-r').textContent = study.strongest_pairs[0].r.toFixed(3);
}

function describePair(test) {
  const target = $('matrix-details');
  const header = target.querySelector('h3');
  header.textContent = `${traitNames[test.a]} × ${traitNames[test.b]}`;
  target.querySelector('strong').textContent = correlationFormat(test.r);
  target.querySelector('p:not(.micro-title)').textContent =
    `Pearson r · unadjusted p ${probabilityFormat(test.p)} · adjusted q ${probabilityFormat(test.q)}. ${test.fdr_significant ? 'This association falls below the 5% false-discovery-rate threshold in this cohort.' : 'This association does not meet the 5% false-discovery-rate threshold.'} Neither outcome establishes causation.`;
  drawPairVisualization(test.a, test.b, test.r);
  $$('.heat-cell[data-left]').forEach(button => {
    button.setAttribute('aria-pressed', String((button.dataset.left === test.a && button.dataset.right === test.b) || (button.dataset.left === test.b && button.dataset.right === test.a)));
  });
}

function drawTopPairs() {
  const target = $('top-pairs');
  study.strongest_pairs.forEach((test, index) => {
    const row = element('div', 'top-pair');
    row.appendChild(element('span', 'top-index', String(index + 1).padStart(2, '0')));
    row.appendChild(element('span', 'pair-name', `${traitNames[test.a]} × ${traitNames[test.b]}`));
    const meter = element('span', 'pair-meter'); const fill = element('span', 'pair-fill');
    fill.style.width = `${Math.abs(test.r) * 100}%`;
    fill.style.setProperty('--left-color', traitPalette[test.a].soft);
    fill.style.setProperty('--right-color', traitPalette[test.b].soft);
    meter.appendChild(fill); row.appendChild(meter);
    row.appendChild(element('span', 'pair-r', correlationFormat(test.r)));
    target.appendChild(row);
  });
}

function drawProjectionConcept() {
  const target = $('projection-demo');
  let mode = 'traits';
  const random = seedRandom(12988);
  const markers = Array.from({ length: 35 }, (_, i) => ({ x: 535 + random() * 250, y: 65 + random() * 240, code: traitCodes[i % 7] }));

  function render() {
    target.replaceChildren();
    const left = 126, mid = 376, right = 645;
    const lines = svgNode('g', { fill: 'none' });
    const dots = svgNode('g');
    for (let y = 50; y <= 340; y += 48) {
      lines.appendChild(svgNode('line', { x1: 470, y1: y, x2: 790, y2: y, stroke: '#8d9fc7', 'stroke-opacity': .12 }));
    }
    for (let x = 482; x <= 790; x += 52) lines.appendChild(svgNode('line', { x1: x, y1: 36, x2: x, y2: 349, stroke: '#8d9fc7', 'stroke-opacity': .12 }));
    target.appendChild(lines);
    if (mode === 'traits') {
      traitCodes.forEach((code, i) => {
        const y = 53 + i * 45;
        const col = traitPalette[code].soft;
        dots.appendChild(svgNode('line', { x1: left + 27, y1: y, x2: mid - 37, y2: 185, stroke: col, 'stroke-opacity': .52, 'stroke-width': 1.7 }));
        dots.appendChild(svgNode('circle', { cx: left, cy: y, r: 18, fill: col, 'fill-opacity': .10, stroke: col, 'stroke-width': 1.5, 'stroke-opacity': .85 }));
        dots.appendChild(svgNode('text', { x: left, y: y + 4, fill: col, 'font-size': 11, 'font-weight': 900, 'text-anchor': 'middle' }, code));
      });
      dots.append(svgNode('circle', { cx: mid, cy: 185, r: 36, fill: '#111a36', stroke: '#d8bfff', 'stroke-width': 2.4 }), svgNode('text', { x: mid, y: 191, fill: '#edf5ff', 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 850 }, 'PROFILE'));
      dots.appendChild(svgNode('path', { d: `M ${mid + 39} 185 C 473 185, 477 185, 505 185`, stroke: '#c3aaff', 'stroke-width': 2, 'stroke-dasharray': '5 7', fill: 'none' }));
      markers.slice(0, 22).forEach(p => dots.appendChild(svgNode('circle', { cx: p.x, cy: p.y, r: 3.7, fill: '#99aacd', 'fill-opacity': .28 })));
      dots.appendChild(svgNode('text', { x: 620, y: 369, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 850, fill: '#8d9dbd' }, 'THE TARGET MAP'));
    } else {
      traitCodes.forEach((code, i) => {
        const y = 53 + i * 45;
        dots.appendChild(svgNode('circle', { cx: left, cy: y, r: 8, fill: traitPalette[code].soft, 'fill-opacity': .65 }));
        dots.appendChild(svgNode('line', { x1: left + 12, y1: y, x2: mid - 40, y2: 185, stroke: traitPalette[code].soft, 'stroke-opacity': .22, 'stroke-width': 1.4 }));
      });
      dots.append(svgNode('circle', { cx: mid, cy: 185, r: 30, fill: '#141d39', stroke: '#e0b9ff', 'stroke-width': 2 }), svgNode('text', { x: mid, y: 191, fill: '#edf5ff', 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 850 }, 'MDS'));
      dots.appendChild(svgNode('path', { d: `M ${mid + 35} 185 C 460 185, 468 185, 505 185`, stroke: '#c8bcff', 'stroke-width': 2.4, fill: 'none' }));
      markers.forEach(p => {
        const x = right + (p.x - right) * .85, y = 185 + (p.y - 185) * .85;
        dots.appendChild(svgNode('circle', { cx: x, cy: y, r: 7, fill: traitPalette[p.code].soft, 'fill-opacity': .12 }));
        dots.appendChild(svgNode('circle', { cx: x, cy: y, r: 3.7, fill: traitPalette[p.code].soft, opacity: .85 }));
      });
      dots.appendChild(svgNode('text', { x: 650, y: 369, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 850, fill: '#b9b9ec' }, 'TWO-DIMENSIONAL COORDINATES'));
    }
    target.appendChild(dots);
    if (!reducedMotion && typeof dots.animate === 'function') dots.animate([{ opacity: .14 }, { opacity: 1 }], { duration: 650, easing: 'ease-out' });
  }
  $$('.seg[data-projection]').forEach(button => button.addEventListener('click', () => {
    mode = button.dataset.projection;
    setActiveButton('.seg[data-projection]', 'projection', mode);
    $('projection-state-label').textContent = mode === 'traits' ? 'SOURCE / SEVEN TRAITS' : 'RESULT / TWO COORDINATES';
    $('projection-state-title').textContent = mode === 'traits' ? 'Seven separate signals' : 'One spatial compromise';
    $('projection-state-copy').textContent = mode === 'traits'
      ? 'The original profile contains seven composite dimensions. Every dimension contributes to pairwise similarity after normalization.'
      : 'Metric MDS seeks two coordinates that reproduce the original pairwise distances as closely as possible. Some relationships shift in the process.';
    render();
  }));
  render();
}

function drawDistanceChart() {
  const bins = study.projection.distance_bins;
  const target = $('distance-chart');
  const chart = svgNode('svg', { viewBox: '0 0 690 360', 'aria-hidden': true });
  const left = 65, right = 653, top = 25, bottom = 299;
  const max = Math.ceil(Math.max(...bins.flatMap(p => [p.original, p.embedded])) * 5) / 5;
  const x = d => left + (d / max) * (right - left);
  const y = d => bottom - (d / max) * (bottom - top);
  [0, .25, .5, .75, 1].forEach(fraction => {
    const v = fraction * max;
    chart.append(svgNode('line', { x1: x(v), y1: top, x2: x(v), y2: bottom, class: 'plot-guide' }), svgNode('line', { x1: left, y1: y(v), x2: right, y2: y(v), class: 'plot-guide' }));
    chart.append(svgNode('text', { x: x(v), y: bottom + 20, 'text-anchor': 'middle', class: 'plot-text' }, v.toFixed(1)), svgNode('text', { x: left - 11, y: y(v) + 4, 'text-anchor': 'end', class: 'plot-text' }, v.toFixed(1)));
  });
  chart.appendChild(svgNode('line', { x1: left, y1: bottom, x2: right, y2: top, class: 'plot-diagonal' }));
  const line = svgNode('path', { d: bins.map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(p.original)} ${y(p.embedded)}`).join(' '), fill: 'none', stroke: '#aca8f6', 'stroke-width': 2.4, opacity: .7 });
  chart.appendChild(line);
  bins.forEach(p => {
    const dot = svgNode('circle', { cx: x(p.original), cy: y(p.embedded), r: 6, class: 'plot-mark' });
    dot.appendChild(svgNode('title', {}, `${p.pairs} profile pairs. Original median ${p.original.toFixed(3)}; projected median ${p.embedded.toFixed(3)}.`));
    chart.appendChild(dot);
  });
  chart.append(svgNode('text', { x: (left + right) / 2, y: 351, 'text-anchor': 'middle', class: 'plot-text' }, 'DISTANCE IN SEVEN DIMENSIONS'), svgNode('text', { x: 18, y: 165, 'text-anchor': 'middle', transform: 'rotate(-90 18 165)', class: 'plot-text' }, 'DISTANCE IN TWO DIMENSIONS'));
  target.appendChild(chart);
}

function drawClusters() {
  const target = $('cluster-bars');
  for (const k of [2, 4]) {
    const metric = study.clustering.solutions[String(k)];
    const row = element('div', `cluster-row k-${k === 4 ? 'four' : 'two'}`);
    const label = element('div', 'cluster-label');
    label.append(element('span', '', `${k} groups · silhouette`), element('strong', '', metric.silhouette.toFixed(3)));
    const track = element('div', 'cluster-track');
    const fill = element('span', 'cluster-fill');
    fill.style.width = `${clamp(metric.silhouette / .35 * 100, 0, 100)}%`;
    track.appendChild(fill);
    row.append(label, track, element('p', 'figure-note', `Group sizes: ${metric.sizes.join(' / ')}`));
    target.appendChild(row);
  }
}

function drawFutureOrbit() {
  const svg = $('future-orbit');
  const random = seedRandom(37419), cx = 260, cy = 260;
  for (const radius of [80, 132, 189, 235]) svg.appendChild(svgNode('ellipse', { cx, cy, rx: radius, ry: radius * .8, transform: `rotate(-22 ${cx} ${cy})`, fill: 'none', stroke: '#9b91df', 'stroke-opacity': radius === 235 ? .21 : .14, 'stroke-width': 1 }));
  const points = Array.from({ length: 59 }, (_, i) => {
    const angle = random() * Math.PI * 2, radius = 48 + Math.pow(random(), .6) * 181;
    return { x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius * .8, code: traitCodes[i % 7], z: random() };
  });
  for (let i = 0; i < points.length; i++) {
    const a = points[i], nearest = points.slice(i + 1).map(b => ({ b, d: Math.hypot(a.x - b.x, a.y - b.y) })).sort((x, y) => x.d - y.d)[0];
    if (nearest && nearest.d < 120) svg.appendChild(svgNode('line', { x1: a.x, y1: a.y, x2: nearest.b.x, y2: nearest.b.y, stroke: '#aaa8dc', 'stroke-opacity': .22, 'stroke-width': 1.1 }));
  }
  for (const point of points) {
    const col = traitPalette[point.code].soft;
    svg.append(svgNode('circle', { cx: point.x, cy: point.y, r: 9 + point.z * 10, fill: col, 'fill-opacity': .08 }), svgNode('circle', { cx: point.x, cy: point.y, r: 2.4 + point.z * 2, fill: col, 'fill-opacity': .92 }));
  }
  svg.appendChild(svgNode('circle', { cx, cy, r: 25, fill: '#15152b', stroke: '#d0b7ff', 'stroke-width': 2 }));
  svg.appendChild(svgNode('circle', { cx, cy, r: 6.5, fill: '#f8b7de' }));
}

function enableCopy() {
  $('copy-command').addEventListener('click', async event => {
    const button = event.currentTarget;
    try {
      await navigator.clipboard.writeText('python scripts/research_analysis.py');
      button.textContent = 'Copied';
      setTimeout(() => { button.textContent = 'Copy'; }, 2000);
    } catch {
      button.textContent = 'Select text';
    }
  });
}

attachScrollPresentation();
drawHero();
drawOrigin();
drawTraitOrbit();
drawShapeLab();
drawCorrelations();
drawTopPairs();
drawProjectionConcept();
drawDistanceChart();
drawClusters();
drawFutureOrbit();
enableCopy();
