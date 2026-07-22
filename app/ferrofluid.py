"""Ferrofluid animated WebGL page background (ported from React Bits to vanilla JS).

A full-page background is awkward in Streamlit: components render inside a
sandboxed iframe that sits inline in the layout. The trick here: from the
component iframe we inject a `<script type="module">` into the *parent*
document, which creates a fixed, full-viewport canvas behind all content and
runs the shader in the parent's context — so the animation keeps going across
Streamlit reruns (an iframe-local loop would freeze on every interaction).

The GLSL is the React Bits Ferrofluid shader; only the wiring is rewritten.
"""

import json

import streamlit.components.v1 as components

_VERTEX = r"""
attribute vec2 position;
attribute vec2 uv;
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 0.0, 1.0);
}
"""

_FRAGMENT = r"""
precision highp float;

uniform vec3  iResolution;
uniform vec2  iMouse;
uniform float iTime;

uniform vec3  uColor0;
uniform vec3  uColor1;
uniform vec3  uColor2;
uniform vec3  uColor3;
uniform vec3  uColor4;
uniform vec3  uColor5;
uniform vec3  uColor6;
uniform vec3  uColor7;
uniform int   uColorCount;

uniform vec3  uMouseColor;
uniform vec2  uFlow;
uniform float uSpeed;
uniform float uScale;
uniform float uTurbulence;
uniform float uFluidity;
uniform float uRimWidth;
uniform float uSharpness;
uniform float uShimmer;
uniform float uGlow;
uniform float uOpacity;
uniform float uMouseEnabled;
uniform float uMouseStrength;
uniform float uMouseRadius;

varying vec2 vUv;

#define PI 3.14159265

vec3 palette(float h) {
  int count = uColorCount;
  if (count < 1) count = 1;
  int idx = int(floor(clamp(h, 0.0, 0.999999) * float(count)));
  if (idx <= 0) return uColor0;
  if (idx == 1) return uColor1;
  if (idx == 2) return uColor2;
  if (idx == 3) return uColor3;
  if (idx == 4) return uColor4;
  if (idx == 5) return uColor5;
  if (idx == 6) return uColor6;
  return uColor7;
}

float hash(vec3 p3) {
  p3 = fract(p3 * 0.1031);
  p3 += dot(p3, p3.zyx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float smin(float a, float b, float k) {
  float r = exp2(-a / k) + exp2(-b / k);
  return -k * log2(r);
}

float sinlerp(float a, float b, float w) {
  return mix(a, b, (sin(w * PI - PI / 2.0) + 1.0) / 2.0);
}

float vn(vec2 p, float s, float seed) {
  vec2 cellp = floor(p / s);
  vec2 relp = mod(p, s);
  float g1 = hash(vec3(cellp, seed));
  float g2 = hash(vec3(cellp.x + 1.0, cellp.y, seed));
  float g3 = hash(vec3(cellp.x + 1.0, cellp.y + 1.0, seed));
  float g4 = hash(vec3(cellp.x, cellp.y + 1.0, seed));
  float bx = sinlerp(g1, g2, relp.x / s);
  float tx = sinlerp(g4, g3, relp.x / s);
  return sinlerp(bx, tx, relp.y / s);
}

float dbn(vec2 p, float s, float seed) {
  float o = s / 2.0;
  float n0 = vn(p, s, seed);
  float n1 = vn(p + vec2(o, o), s, seed + 0.1);
  float n2 = vn(p + vec2(-o, o), s, seed + 0.2);
  float n3 = vn(p + vec2(o, -o), s, seed + 0.3);
  float n4 = vn(p + vec2(-o, -o), s, seed + 0.4);
  return (2.0 * n0 + 1.5 * n1 + 1.25 * n2 + 1.125 * n3 + n4) / 7.0;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  float ref = 700.0 / max(uScale, 0.05);
  vec2 p = fragCoord / iResolution.y * ref;

  float spd = 200.0 * uSpeed;
  float t = iTime;

  vec2 dir = uFlow;
  vec2 perp = vec2(-dir.y, dir.x);

  float distort1 = vn(p + perp * (t * spd), 60.0, 10.0) * 50.0 * uTurbulence;
  float distort2 = vn(p - perp * (t * spd), 120.0, 15.0) * 100.0 * uTurbulence;

  float peaks = dbn(p + distort1 + dir * (t * spd * 0.5), 40.0, 1.0);
  float peaks2 = dbn(p + distort2 - dir * (t * spd * 0.5), 40.0, 0.0);

  float mapeaks = smin(peaks, peaks2, max(uFluidity, 0.001));

  float mGlow = 0.0;
  if (uMouseEnabled > 0.5) {
    vec2 mp = iMouse / iResolution.y * ref;
    float md = length(p - mp) / ref;
    float rr = max(uMouseRadius, 0.02);
    mGlow = exp(-md * md / (rr * rr)) * uMouseStrength;
  }

  float band = (uRimWidth - abs((mapeaks - 0.4) * 2.0)) * 5.0;
  float ltn = clamp(band - vn(p + dir * (t * spd * 0.5), 60.0, 12.0) * uShimmer, 0.0, 1.0);
  ltn = pow(ltn, uSharpness) * uGlow;
  ltn *= clamp(1.0 - mGlow, 0.0, 1.0);

  float h = clamp(0.5 + (peaks - peaks2) * 0.8, 0.0, 1.0);
  vec3 col = palette(h);

  vec3 outc = col * ltn;
  float a = clamp(max(outc.r, max(outc.g, outc.b)), 0.0, 1.0);
  fragColor = vec4(outc, a * uOpacity);
}

void main() {
  vec4 color;
  mainImage(color, vUv * iResolution.xy);
  gl_FragColor = color;
}
"""

_FLOW = {"up": [0, 1], "down": [0, -1], "left": [-1, 0], "right": [1, 0]}


def _hex_to_rgb(h):
    h = h.lstrip("#").ljust(6, "0")
    return [int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255]


def _prep_colors(colors):
    base = (colors or ["#4F46E5", "#06B6D4", "#E0F2FE"])[:8]
    rgb = [_hex_to_rgb(base[min(i, len(base) - 1)]) for i in range(8)]
    count = len(base)
    avg = [sum(rgb[i][c] for i in range(count)) / count for c in range(3)]
    return rgb, count, avg


_SETUP_JS = r"""
const host = document.createElement('div');
host.id = 'ff-bg-host';
host.style.cssText = 'position:fixed;inset:0;z-index:-1;pointer-events:none;overflow:hidden;background:' + cfg.bg + ';';
document.body.appendChild(host);

const renderer = new Renderer({ dpr: Math.min(window.devicePixelRatio || 1, cfg.dprCap), alpha: true, antialias: true });
const gl = renderer.gl;
gl.clearColor(0, 0, 0, 0);
const canvas = gl.canvas;
canvas.style.cssText = 'width:100%;height:100%;display:block;';
host.appendChild(canvas);

const c = cfg.colors;
const uniforms = {
  iResolution: { value: [1, 1, 1] },
  iMouse: { value: [0, 0] },
  iTime: { value: 0 },
  uColor0: { value: c[0] }, uColor1: { value: c[1] }, uColor2: { value: c[2] }, uColor3: { value: c[3] },
  uColor4: { value: c[4] }, uColor5: { value: c[5] }, uColor6: { value: c[6] }, uColor7: { value: c[7] },
  uColorCount: { value: cfg.count },
  uMouseColor: { value: cfg.avg },
  uFlow: { value: cfg.flow },
  uSpeed: { value: cfg.speed },
  uScale: { value: cfg.scale },
  uTurbulence: { value: cfg.turbulence },
  uFluidity: { value: cfg.fluidity },
  uRimWidth: { value: cfg.rimWidth },
  uSharpness: { value: cfg.sharpness },
  uShimmer: { value: cfg.shimmer },
  uGlow: { value: cfg.glow },
  uOpacity: { value: cfg.opacity },
  uMouseEnabled: { value: cfg.mouseEnabled ? 1 : 0 },
  uMouseStrength: { value: cfg.mouseStrength },
  uMouseRadius: { value: cfg.mouseRadius }
};

const program = new Program(gl, { vertex, fragment, uniforms });
const mesh = new Mesh(gl, { geometry: new Triangle(gl), program });

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight);
  uniforms.iResolution.value = [gl.drawingBufferWidth, gl.drawingBufferHeight, 1];
}
resize();
window.addEventListener('resize', resize);

let mx = 0, my = 0;
if (cfg.mouseEnabled) {
  window.addEventListener('pointermove', (e) => {
    const sc = renderer.dpr || 1;
    mx = e.clientX * sc;
    my = (window.innerHeight - e.clientY) * sc;
  });
}

function loop(t) {
  requestAnimationFrame(loop);
  uniforms.iTime.value = t * 0.001;
  const cur = uniforms.iMouse.value;
  cur[0] += (mx - cur[0]) * 0.1;
  cur[1] += (my - cur[1]) * 0.1;
  try { renderer.render({ scene: mesh }); } catch (e) {}
}
requestAnimationFrame(loop);
"""


def render_background(
    colors=("#3987e5", "#4f7ff0", "#7b6cf6"),
    background="#0d0d0d",
    speed=0.28,
    scale=1.5,
    turbulence=0.9,
    fluidity=0.12,
    rim_width=0.2,
    sharpness=3.0,
    shimmer=1.0,
    glow=1.3,
    flow_direction="down",
    opacity=0.85,
    mouse_interaction=True,
    mouse_strength=1.0,
    mouse_radius=0.3,
    dpr_cap=1.5,
):
    """Inject a fixed, full-viewport Ferrofluid canvas behind all app content."""
    rgb, count, avg = _prep_colors(list(colors))
    cfg = {
        "bg": background,
        "colors": rgb,
        "count": count,
        "avg": avg,
        "flow": _FLOW.get(flow_direction, [0, -1]),
        "speed": speed,
        "scale": scale,
        "turbulence": turbulence,
        "fluidity": fluidity,
        "rimWidth": rim_width,
        "sharpness": sharpness,
        "shimmer": shimmer,
        "glow": glow,
        "opacity": opacity,
        "mouseEnabled": mouse_interaction,
        "mouseStrength": mouse_strength,
        "mouseRadius": mouse_radius,
        "dprCap": dpr_cap,
    }

    module_code = (
        "import { Renderer, Program, Mesh, Triangle } from 'https://esm.sh/ogl@1.0.11';\n"
        "if (!document.getElementById('ff-bg-host')) {\n"
        "const cfg = " + json.dumps(cfg) + ";\n"
        "const vertex = " + json.dumps(_VERTEX) + ";\n"
        "const fragment = " + json.dumps(_FRAGMENT) + ";\n"
        + _SETUP_JS +
        "\n}"
    )

    loader = (
        "<script>(function(){try{"
        "var doc = window.parent.document;"
        "if (!doc.getElementById('ff-bg-script')) {"
        "var s = doc.createElement('script');"
        "s.type = 'module'; s.id = 'ff-bg-script';"
        "s.textContent = " + json.dumps(module_code) + ";"
        "doc.head.appendChild(s);"
        "}"
        # Collapse this component's own Streamlit container so its (invisible)
        # iframe slot doesn't leave an empty flex gap above the page header.
        "var fe = window.frameElement;"
        "if (fe) { var c = fe.closest('[data-testid=\"stElementContainer\"]');"
        " if (c) c.style.display = 'none'; }"
        "}catch(e){console.error('ferrofluid bg inject failed', e);}})();</script>"
    )
    components.html(loader, height=0)
