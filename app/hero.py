"""Hero title card with MagicBento-style interactive effects (vanilla JS).

Adapts the React Bits MagicBento visual language — a cursor-following border
glow, a soft spotlight, and floating particles — into a single self-contained
hero card that holds the app title. Rendered via st.components.v1.html; no gsap
(CSS transitions + a tiny rAF loop do the same job). The glow uses the app's
blue accent rather than MagicBento's default purple.
"""

import streamlit.components.v1 as components

GLOW_RGB = "57, 135, 229"  # #3987e5, the app accent


def render_hero(
    title: str = "Decision-Driven Customer Retention",
    subtitle: str = "Turning churn predictions into budget-constrained decisions",
    badge: str = "Decision Intelligence",
    height: int = 260,
):
    html = _TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__SUBTITLE__", subtitle)
    html = html.replace("__BADGE__", badge)
    html = html.replace("__GLOW__", GLOW_RGB)
    html = html.replace("__HEIGHT__", str(height))
    components.html(html, height=height + 12)


_TEMPLATE = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&display=swap');

.hero-card {
  --glow-x: 50%; --glow-y: 50%; --glow-intensity: 0;
  position: relative;
  height: __HEIGHT__px;
  border-radius: 20px;
  padding: 34px 44px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  overflow: hidden;
  background:
    radial-gradient(120% 140% at 12% 15%, rgba(57,135,229,0.10) 0%, transparent 45%),
    linear-gradient(135deg, #15171d 0%, #101116 60%, #0d0d10 100%);
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 20px 50px -18px rgba(0,0,0,0.65);
  transition: transform 0.25s ease;
  cursor: default;
}

/* Cursor-following border glow (masked to the border ring). */
.hero-card::after {
  content: '';
  position: absolute; inset: 0;
  padding: 1.5px;
  border-radius: inherit;
  background: radial-gradient(260px circle at var(--glow-x) var(--glow-y),
    rgba(__GLOW__, calc(var(--glow-intensity) * 0.9)) 0%,
    rgba(__GLOW__, calc(var(--glow-intensity) * 0.35)) 35%,
    transparent 65%);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  mask-composite: exclude;
  pointer-events: none;
  z-index: 3;
}

.hero-spot {
  position: absolute;
  width: 520px; height: 520px;
  border-radius: 50%;
  pointer-events: none;
  transform: translate(-50%, -50%);
  background: radial-gradient(circle,
    rgba(__GLOW__, 0.14) 0%, rgba(__GLOW__, 0.06) 30%, transparent 60%);
  mix-blend-mode: screen;
  opacity: 0;
  transition: opacity 0.3s ease;
  z-index: 1;
}

.hero-badge {
  align-self: flex-start;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #8fbcf3;
  padding: 5px 13px;
  border-radius: 999px;
  border: 1px solid rgba(57,135,229,0.38);
  background: rgba(57,135,229,0.09);
  margin-bottom: 16px;
  position: relative; z-index: 2;
}

.hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: clamp(2rem, 3.6vw, 3.1rem);
  line-height: 1.04;
  letter-spacing: -1.2px;
  margin: 0;
  background: linear-gradient(92deg, #ffffff 0%, #cfe0f7 45%, #7fb0f5 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
  position: relative; z-index: 2;
}

.hero-sub {
  font-family: 'Inter', sans-serif;
  font-size: 1.05rem;
  color: #a4abbb;
  margin: 12px 0 0;
  position: relative; z-index: 2;
}

.hero-particle {
  position: absolute;
  width: 3px; height: 3px;
  border-radius: 50%;
  background: rgba(__GLOW__, 0.9);
  box-shadow: 0 0 7px rgba(__GLOW__, 0.7);
  pointer-events: none;
  z-index: 1;
  opacity: 0;
  animation: floatp 7s ease-in-out infinite;
}
@keyframes floatp {
  0%   { transform: translate(0,0);        opacity: 0; }
  15%  { opacity: 0.7; }
  50%  { transform: translate(14px,-20px); opacity: 0.9; }
  85%  { opacity: 0.6; }
  100% { transform: translate(-8px,-38px); opacity: 0; }
}
</style>

<div class="hero-card" id="heroCard">
  <div class="hero-spot" id="heroSpot"></div>
  <div class="hero-badge">◆ __BADGE__</div>
  <h1 class="hero-title">__TITLE__</h1>
  <p class="hero-sub">__SUBTITLE__</p>
</div>

<script>
(function(){
  const card = document.getElementById('heroCard');
  const spot = document.getElementById('heroSpot');

  // Floating particles
  const N = 14;
  for (let i = 0; i < N; i++) {
    const p = document.createElement('div');
    p.className = 'hero-particle';
    p.style.left = (Math.random() * 100) + '%';
    p.style.top = (40 + Math.random() * 60) + '%';
    p.style.animationDelay = (Math.random() * 7) + 's';
    p.style.animationDuration = (5 + Math.random() * 5) + 's';
    card.appendChild(p);
  }

  card.addEventListener('pointermove', (e) => {
    const r = card.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    card.style.setProperty('--glow-x', (x / r.width * 100) + '%');
    card.style.setProperty('--glow-y', (y / r.height * 100) + '%');
    card.style.setProperty('--glow-intensity', '1');
    spot.style.left = x + 'px';
    spot.style.top = y + 'px';
    spot.style.opacity = '1';
    // subtle magnetism
    const mx = (x - r.width/2) * 0.012, my = (y - r.height/2) * 0.012;
    card.style.transform = 'translate(' + mx + 'px,' + my + 'px)';
  });
  card.addEventListener('pointerleave', () => {
    card.style.setProperty('--glow-intensity', '0');
    spot.style.opacity = '0';
    card.style.transform = 'translate(0,0)';
  });
})();
</script>
"""
