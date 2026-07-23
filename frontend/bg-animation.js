/**
 * REALYTICS — Animated Cyber Background
 * Draws a neural-net particle field on a fixed canvas behind all page content.
 * Particles drift slowly, connect with glowing edges when close,
 * and pulse in the accent colour palette.
 */
(function () {
    /* ── helpers ─────────────────────────────────────────────────── */
    const rand = (min, max) => Math.random() * (max - min) + min;
    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

    /* ── canvas setup ────────────────────────────────────────────── */
    const canvas = document.createElement('canvas');
    canvas.id = 'bg-canvas';
    Object.assign(canvas.style, {
        position: 'fixed',
        top: 0, left: 0,
        width: '100%', height: '100%',
        zIndex: 0,
        pointerEvents: 'none',
        opacity: 1,
    });
    document.body.prepend(canvas);
    const ctx = canvas.getContext('2d');

    /* ── theme palette ───────────────────────────────────────────── */
    const isLight = () => document.body.getAttribute('data-theme') === 'light';

    const PALETTE_DARK = {
        bg: '#07090e',
        nodes: ['rgba(212,175,55,', 'rgba(0,242,254,', 'rgba(138,43,226,'],
        edge: 'rgba(212,175,55,',
        orbs: [
            { x: 0.15, y: 0.12, r: 380, color: 'rgba(212,175,55,0.07)' },
            { x: 0.82, y: 0.75, r: 420, color: 'rgba(0,242,254,0.055)' },
            { x: 0.5,  y: 0.45, r: 300, color: 'rgba(138,43,226,0.05)' },
        ],
    };
    const PALETTE_LIGHT = {
        bg: '#f0f4f8',
        nodes: ['rgba(184,147,32,', 'rgba(0,180,200,', 'rgba(100,40,180,'],
        edge: 'rgba(184,147,32,',
        orbs: [
            { x: 0.15, y: 0.12, r: 380, color: 'rgba(184,147,32,0.06)' },
            { x: 0.82, y: 0.75, r: 420, color: 'rgba(0,180,200,0.05)' },
            { x: 0.5,  y: 0.45, r: 300, color: 'rgba(120,60,200,0.04)' },
        ],
    };

    /* ── resize ──────────────────────────────────────────────────── */
    let W, H;
    function resize() {
        W = canvas.width  = window.innerWidth;
        H = canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    /* ── nodes ───────────────────────────────────────────────────── */
    const COUNT   = Math.min(90, Math.max(50, Math.floor((W * H) / 18000)));
    const MAX_DIST = 170;      // px — max edge length
    const SPEED   = 0.35;      // max px/frame

    class Node {
        constructor() { this.reset(true); }
        reset(initial = false) {
            this.x  = rand(0, W);
            this.y  = initial ? rand(0, H) : (Math.random() < 0.5 ? -8 : H + 8);
            this.vx = rand(-SPEED, SPEED);
            this.vy = rand(-SPEED, SPEED);
            this.r  = rand(1.5, 3.5);
            this.colorIdx = Math.floor(rand(0, 3));
            this.alpha = rand(0.55, 1);
            this.pulseOffset = rand(0, Math.PI * 2);
            this.pulseSpeed  = rand(0.008, 0.022);
        }
        update(t) {
            this.x += this.vx;
            this.y += this.vy;
            const pad = 20;
            if (this.x < -pad || this.x > W + pad || this.y < -pad || this.y > H + pad) this.reset();
            this.pulse = 0.7 + 0.3 * Math.sin(t * this.pulseSpeed + this.pulseOffset);
        }
        draw(palette) {
            const a = clamp(this.alpha * this.pulse, 0, 1);
            const col = palette.nodes[this.colorIdx];

            // glow halo
            const grd = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.r * 5);
            grd.addColorStop(0,   col + (a * 0.45).toFixed(3) + ')');
            grd.addColorStop(1,   col + '0)');
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.r * 5, 0, Math.PI * 2);
            ctx.fillStyle = grd;
            ctx.fill();

            // core dot
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.r * this.pulse, 0, Math.PI * 2);
            ctx.fillStyle = col + a.toFixed(3) + ')';
            ctx.fill();
        }
    }

    const nodes = Array.from({ length: COUNT }, () => new Node());

    /* ── draw edges ─────────────────────────────────────────────── */
    function drawEdges(palette) {
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist > MAX_DIST) continue;
                const alpha = clamp((1 - dist / MAX_DIST) * 0.45, 0, 1);
                ctx.beginPath();
                ctx.moveTo(nodes[i].x, nodes[i].y);
                ctx.lineTo(nodes[j].x, nodes[j].y);
                ctx.strokeStyle = palette.edge + alpha.toFixed(3) + ')';
                ctx.lineWidth = 0.8;
                ctx.stroke();
            }
        }
    }

    /* ── draw ambient gradient orbs ────────────────────────────── */
    function drawOrbs(palette, t) {
        palette.orbs.forEach((orb, idx) => {
            // slow drift
            const ox = orb.x * W + Math.sin(t * 0.0004 + idx * 2.1) * 60;
            const oy = orb.y * H + Math.cos(t * 0.0003 + idx * 1.7) * 50;
            const grd = ctx.createRadialGradient(ox, oy, 0, ox, oy, orb.r);
            grd.addColorStop(0, orb.color);
            grd.addColorStop(1, 'transparent');
            ctx.fillStyle = grd;
            ctx.fillRect(0, 0, W, H);
        });
    }

    /* ── mouse repulse ──────────────────────────────────────────── */
    let mx = -9999, my = -9999;
    window.addEventListener('mousemove', e => { mx = e.clientX; my = e.clientY; });
    window.addEventListener('mouseleave', () => { mx = -9999; my = -9999; });

    function applyRepulse() {
        const REPULSE_R = 120, REPULSE_FORCE = 1.2;
        nodes.forEach(n => {
            const dx = n.x - mx, dy = n.y - my;
            const d = Math.sqrt(dx * dx + dy * dy);
            if (d < REPULSE_R && d > 0) {
                const force = (1 - d / REPULSE_R) * REPULSE_FORCE;
                n.x += (dx / d) * force;
                n.y += (dy / d) * force;
            }
        });
    }

    /* ── main loop ──────────────────────────────────────────────── */
    let t = 0;
    function frame() {
        const palette = isLight() ? PALETTE_LIGHT : PALETTE_DARK;
        t++;

        // clear
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = palette.bg;
        ctx.fillRect(0, 0, W, H);

        drawOrbs(palette, t);
        applyRepulse();
        drawEdges(palette);
        nodes.forEach(n => { n.update(t); n.draw(palette); });

        requestAnimationFrame(frame);
    }

    // start after DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', frame);
    } else {
        frame();
    }

    // watch for theme changes
    const observer = new MutationObserver(() => {});
    observer.observe(document.body, { attributes: true, attributeFilter: ['data-theme'] });
})();
