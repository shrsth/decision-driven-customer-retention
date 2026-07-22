"""LightRays animated WebGL page background (ported from React Bits to vanilla JS).

Same full-page technique as the other backgrounds: from the component iframe we
inject a `<script type="module">` into the parent document, which creates a
fixed, full-viewport canvas behind all content and runs the shader in the
parent's context so the animation survives Streamlit reruns. Guards keep it to
a single canvas. Soft light rays fan down from the top — a calm, characterful
ambient rather than a flat gradient.
"""

import json

import streamlit.components.v1 as components

_VERTEX = r"""
attribute vec2 position;
varying vec2 vUv;
void main() {
  vUv = position * 0.5 + 0.5;
  gl_Position = vec4(position, 0.0, 1.0);
}
"""

_FRAGMENT = r"""precision highp float;

uniform float iTime;
uniform vec2  iResolution;

uniform vec2  rayPos;
uniform vec2  rayDir;
uniform vec3  raysColor;
uniform float raysSpeed;
uniform float lightSpread;
uniform float rayLength;
uniform float pulsating;
uniform float fadeDistance;
uniform float saturation;
uniform vec2  mousePos;
uniform float mouseInfluence;
uniform float noiseAmount;
uniform float distortion;

varying vec2 vUv;

float noise(vec2 st) {
  return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float rayStrength(vec2 raySource, vec2 rayRefDirection, vec2 coord,
                  float seedA, float seedB, float speed) {
  vec2 sourceToCoord = coord - raySource;
  vec2 dirNorm = normalize(sourceToCoord);
  float cosAngle = dot(dirNorm, rayRefDirection);

  float distortedAngle = cosAngle + distortion * sin(iTime * 2.0 + length(sourceToCoord) * 0.01) * 0.2;

  float spreadFactor = pow(max(distortedAngle, 0.0), 1.0 / max(lightSpread, 0.001));

  float distance = length(sourceToCoord);
  float maxDistance = iResolution.x * rayLength;
  float lengthFalloff = clamp((maxDistance - distance) / maxDistance, 0.0, 1.0);

  float fadeFalloff = clamp((iResolution.x * fadeDistance - distance) / (iResolution.x * fadeDistance), 0.5, 1.0);
  float pulse = pulsating > 0.5 ? (0.8 + 0.2 * sin(iTime * speed * 3.0)) : 1.0;

  float baseStrength = clamp(
    (0.45 + 0.15 * sin(distortedAngle * seedA + iTime * speed)) +
    (0.3 + 0.2 * cos(-distortedAngle * seedB + iTime * speed)),
    0.0, 1.0
  );

  return baseStrength * lengthFalloff * fadeFalloff * spreadFactor * pulse;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 coord = vec2(fragCoord.x, iResolution.y - fragCoord.y);

  vec2 finalRayDir = rayDir;
  if (mouseInfluence > 0.0) {
    vec2 mouseScreenPos = mousePos * iResolution.xy;
    vec2 mouseDirection = normalize(mouseScreenPos - rayPos);
    finalRayDir = normalize(mix(rayDir, mouseDirection, mouseInfluence));
  }

  vec4 rays1 = vec4(1.0) *
               rayStrength(rayPos, finalRayDir, coord, 36.2214, 21.11349,
                           1.5 * raysSpeed);
  vec4 rays2 = vec4(1.0) *
               rayStrength(rayPos, finalRayDir, coord, 22.3991, 18.0234,
                           1.1 * raysSpeed);

  fragColor = rays1 * 0.5 + rays2 * 0.4;

  if (noiseAmount > 0.0) {
    float n = noise(coord * 0.01 + iTime * 0.1);
    fragColor.rgb *= (1.0 - noiseAmount + noiseAmount * n);
  }

  float brightness = 1.0 - (coord.y / iResolution.y);
  fragColor.x *= 0.1 + brightness * 0.8;
  fragColor.y *= 0.3 + brightness * 0.6;
  fragColor.z *= 0.5 + brightness * 0.5;

  if (saturation != 1.0) {
    float gray = dot(fragColor.rgb, vec3(0.299, 0.587, 0.114));
    fragColor.rgb = mix(vec3(gray), fragColor.rgb, saturation);
  }

  fragColor.rgb *= raysColor;
}

void main() {
  vec4 color;
  mainImage(color, gl_FragCoord.xy);
  gl_FragColor = color;
}
"""


def _hex_to_rgb(h):
    h = h.lstrip("#").ljust(6, "0")
    return [int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255]


_SETUP_JS = r"""
const host = document.createElement('div');
host.id = 'lr-bg-host';
host.style.cssText = 'position:fixed;inset:0;z-index:-1;pointer-events:none;overflow:hidden;background:' + cfg.bg + ';';
document.body.appendChild(host);

const renderer = new Renderer({ dpr: Math.min(window.devicePixelRatio || 1, cfg.dprCap), alpha: true });
const gl = renderer.gl;
gl.clearColor(0, 0, 0, 0);
const canvas = gl.canvas;
canvas.style.cssText = 'width:100%;height:100%;display:block;opacity:' + cfg.opacity + ';';
host.appendChild(canvas);

const uniforms = {
  iTime: { value: 0 },
  iResolution: { value: [1, 1] },
  rayPos: { value: [0, 0] },
  rayDir: { value: [0, 1] },
  raysColor: { value: cfg.color },
  raysSpeed: { value: cfg.speed },
  lightSpread: { value: cfg.spread },
  rayLength: { value: cfg.rayLength },
  pulsating: { value: cfg.pulsating },
  fadeDistance: { value: cfg.fadeDistance },
  saturation: { value: cfg.saturation },
  mousePos: { value: [0.5, 0.5] },
  mouseInfluence: { value: cfg.mouseInfluence },
  noiseAmount: { value: cfg.noise },
  distortion: { value: cfg.distortion }
};

const program = new Program(gl, { vertex, fragment, uniforms });
const mesh = new Mesh(gl, { geometry: new Triangle(gl), program });

function anchorDir(origin, w, h) {
  const o = 0.2;
  switch (origin) {
    case 'top-left': return { anchor: [0, -o * h], dir: [0, 1] };
    case 'top-right': return { anchor: [w, -o * h], dir: [0, 1] };
    case 'left': return { anchor: [-o * w, 0.5 * h], dir: [1, 0] };
    case 'right': return { anchor: [(1 + o) * w, 0.5 * h], dir: [-1, 0] };
    case 'bottom-left': return { anchor: [0, (1 + o) * h], dir: [0, -1] };
    case 'bottom-center': return { anchor: [0.5 * w, (1 + o) * h], dir: [0, -1] };
    case 'bottom-right': return { anchor: [w, (1 + o) * h], dir: [0, -1] };
    default: return { anchor: [0.5 * w, -o * h], dir: [0, 1] };
  }
}

function resize() {
  const wCSS = window.innerWidth, hCSS = window.innerHeight;
  renderer.setSize(wCSS, hCSS);
  const dpr = renderer.dpr;
  const w = wCSS * dpr, h = hCSS * dpr;
  uniforms.iResolution.value = [w, h];
  const ad = anchorDir(cfg.origin, w, h);
  uniforms.rayPos.value = ad.anchor;
  uniforms.rayDir.value = ad.dir;
}
resize();
window.addEventListener('resize', resize);

let mx = 0.5, my = 0.5, smx = 0.5, smy = 0.5;
if (cfg.followMouse) {
  window.addEventListener('mousemove', (e) => {
    mx = e.clientX / window.innerWidth;
    my = e.clientY / window.innerHeight;
  });
}

function loop(t) {
  requestAnimationFrame(loop);
  uniforms.iTime.value = t * 0.001;
  if (cfg.followMouse && cfg.mouseInfluence > 0.0) {
    const s = 0.92;
    smx = smx * s + mx * (1 - s);
    smy = smy * s + my * (1 - s);
    uniforms.mousePos.value = [smx, smy];
  }
  try { renderer.render({ scene: mesh }); } catch (e) {}
}
requestAnimationFrame(loop);
"""


def render_lightrays(
    color="#6ea8ea",
    background="#0d0d0d",
    origin="top-center",
    speed=0.9,
    spread=0.9,
    ray_length=1.5,
    pulsating=False,
    fade_distance=1.1,
    saturation=0.85,
    follow_mouse=True,
    mouse_influence=0.08,
    noise=0.06,
    distortion=0.03,
    opacity=0.9,
    dpr_cap=2,
):
    """Inject a fixed, full-viewport LightRays canvas behind all app content."""
    cfg = {
        "bg": background,
        "color": _hex_to_rgb(color),
        "origin": origin,
        "speed": speed,
        "spread": spread,
        "rayLength": ray_length,
        "pulsating": 1.0 if pulsating else 0.0,
        "fadeDistance": fade_distance,
        "saturation": saturation,
        "followMouse": follow_mouse,
        "mouseInfluence": mouse_influence,
        "noise": noise,
        "distortion": distortion,
        "opacity": opacity,
        "dprCap": dpr_cap,
    }

    module_code = (
        "import { Renderer, Program, Triangle, Mesh } from 'https://esm.sh/ogl@1.0.11';\n"
        "if (!document.getElementById('lr-bg-host')) {\n"
        "const cfg = " + json.dumps(cfg) + ";\n"
        "const vertex = " + json.dumps(_VERTEX) + ";\n"
        "const fragment = " + json.dumps(_FRAGMENT) + ";\n"
        + _SETUP_JS +
        "\n}"
    )

    loader = (
        "<script>(function(){try{"
        "var doc = window.parent.document;"
        "if (doc.getElementById('lr-bg-script')) return;"
        "var s = doc.createElement('script');"
        "s.type = 'module'; s.id = 'lr-bg-script';"
        "s.textContent = " + json.dumps(module_code) + ";"
        "doc.head.appendChild(s);"
        "}catch(e){console.error('lightrays bg inject failed', e);}})();</script>"
    )
    components.html(loader, height=0)
