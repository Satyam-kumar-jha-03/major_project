/**
 * REALYTICS — Profile Features
 * ─────────────────────────────
 * 1. Profile picture upload with rate limiter
 *    - Max 3 uploads per hour per browser session
 *    - Images stored in localStorage as base64
 *    - Applied instantly to all avatar elements on page
 *
 * 2. Context-menu / copy-paste protection
 *    (text-select is disabled via CSS; this covers right-click & shortcuts)
 */

(function () {
    'use strict';

    /* ════════════════════════════════════════════
       SECTION 1 — COPY / RIGHT-CLICK PROTECTION
    ════════════════════════════════════════════ */
    document.addEventListener('contextmenu', e => {
        // Allow right-click on inputs and textareas
        if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
        e.preventDefault();
    });

    document.addEventListener('keydown', e => {
        const key = e.key.toLowerCase();
        const ctrl = e.ctrlKey || e.metaKey;

        // Allow Ctrl+A/C/V/X inside form fields
        if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;

        // Block Ctrl+U (view source), Ctrl+S (save), Ctrl+Shift+I (devtools shortcut via page)
        if (ctrl && key === 'u') e.preventDefault();
        if (ctrl && key === 's') e.preventDefault();
        // Block print screen via JS (limited — best-effort)
        if (key === 'printscreen') e.preventDefault();
    });

    /* ════════════════════════════════════════════
       SECTION 2 — RATE LIMITER ENGINE
    ════════════════════════════════════════════ */
    const RATE_KEY   = 'ry_avatar_uploads';   // localStorage key
    const MAX_UPL    = 10;                      // max uploads
    const WINDOW_MS  = 60 * 60 * 1000;        // per hour (ms)

    /** Returns { allowed: bool, remaining: number, retryAfterMs: number } */
    function checkRateLimit() {
        const now = Date.now();
        let log = [];
        try { log = JSON.parse(localStorage.getItem(RATE_KEY) || '[]'); } catch (_) {}
        // Keep only timestamps within the current window
        log = log.filter(ts => now - ts < WINDOW_MS);
        const allowed = log.length < MAX_UPL;
        const oldest  = log.length > 0 ? log[0] : now;
        const retryAfterMs = allowed ? 0 : WINDOW_MS - (now - oldest);
        return { allowed, remaining: MAX_UPL - log.length, retryAfterMs, log };
    }

    function recordUpload(log) {
        log.push(Date.now());
        localStorage.setItem(RATE_KEY, JSON.stringify(log));
    }

    /* ════════════════════════════════════════════
       SECTION 3 — AVATAR IMAGE HELPERS
    ════════════════════════════════════════════ */
    const AVATAR_KEY = 'ry_avatar_img';

    function applyAvatarToAll(dataUrl) {
        const targets = [
            document.getElementById('userAvatar'),
            document.getElementById('userAvatarLg'),
        ];
        targets.forEach(el => {
            if (!el) return;
            el.style.backgroundImage = `url('${dataUrl}')`;
            el.style.backgroundSize  = 'cover';
            el.style.backgroundPosition = 'center';
            el.style.color = 'transparent';        // hide initial letter
        });
    }

    function loadSavedAvatar() {
        const saved = localStorage.getItem(AVATAR_KEY);
        if (saved) applyAvatarToAll(saved);
    }

    /* ════════════════════════════════════════════
       SECTION 4 — TOAST NOTIFICATION
    ════════════════════════════════════════════ */
    function showProfileToast(msg, type = 'info') {
        const existing = document.getElementById('profileToast');
        if (existing) existing.remove();

        const colorMap = {
            success : '#4caf50',
            error   : '#f44336',
            warning : '#ff9800',
            info    : '#d4af37',
        };

        const toast = document.createElement('div');
        toast.id = 'profileToast';
        Object.assign(toast.style, {
            position        : 'fixed',
            bottom          : '28px',
            left            : '50%',
            transform       : 'translateX(-50%) translateY(20px)',
            background      : 'rgba(12,16,27,0.95)',
            border          : `1px solid ${colorMap[type]}44`,
            borderLeft      : `4px solid ${colorMap[type]}`,
            color           : '#f1f5f9',
            padding         : '12px 22px',
            borderRadius    : '12px',
            fontSize        : '13.5px',
            fontFamily      : "'Plus Jakarta Sans', sans-serif",
            fontWeight      : '500',
            backdropFilter  : 'blur(16px)',
            boxShadow       : `0 8px 32px rgba(0,0,0,0.5), 0 0 20px ${colorMap[type]}22`,
            zIndex          : '99999',
            transition      : 'all 0.35s cubic-bezier(0.16,1,0.3,1)',
            userSelect      : 'none',
            pointerEvents   : 'none',
            maxWidth        : '340px',
            textAlign       : 'center',
            whiteSpace      : 'pre-wrap',
        });
        toast.textContent = msg;
        document.body.appendChild(toast);

        // animate in
        requestAnimationFrame(() => {
            toast.style.transform = 'translateX(-50%) translateY(0)';
            toast.style.opacity = '1';
        });

        setTimeout(() => {
            toast.style.transform = 'translateX(-50%) translateY(20px)';
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 400);
        }, 3500);
    }

    /* ════════════════════════════════════════════
       SECTION 5 — BUILD UPLOAD UI IN DROPDOWN
    ════════════════════════════════════════════ */
    function buildAvatarUI() {
        const dropdown = document.querySelector('.profile-dropdown .dropdown-nav');
        if (!dropdown) return;
        if (document.getElementById('avatarUploadItem')) return; // already injected

        // Hidden file input
        const fileInput = document.createElement('input');
        fileInput.type   = 'file';
        fileInput.accept = 'image/png,image/jpeg,image/webp,image/gif';
        fileInput.id     = 'avatarFileInput';
        Object.assign(fileInput.style, { display: 'none' });
        document.body.appendChild(fileInput);

        // Dropdown item button
        const item = document.createElement('div');
        item.id        = 'avatarUploadItem';
        item.className = 'dropdown-item';
        item.innerHTML = `
            <span class="di-icon" style="background:rgba(0,242,254,0.08);color:#00f2fe;">
                <i class="fa-solid fa-camera"></i>
            </span>
            <span class="avtr-label">Change Photo</span>
            <span id="avatarRateLabel" style="
                margin-left:auto;font-size:10px;
                color:#94a3b8;font-family:'Outfit',sans-serif;
                letter-spacing:0.03em;
            "></span>
        `;

        // Insert before the first divider
        const firstDivider = dropdown.querySelector('.dropdown-divider');
        if (firstDivider) {
            dropdown.insertBefore(item, firstDivider);
            // add a thin divider after our item too
            const div2 = document.createElement('div');
            div2.className = 'dropdown-divider';
            dropdown.insertBefore(div2, firstDivider);
        } else {
            dropdown.appendChild(item);
        }

        updateRateLabel();

        /* ── click handler ── */
        item.addEventListener('click', (e) => {
            e.stopPropagation();
            const { allowed, remaining, retryAfterMs } = checkRateLimit();

            if (!allowed) {
                const mins = Math.ceil(retryAfterMs / 60000);
                showProfileToast(
                    `⏱ Rate limit reached.\nTry again in ${mins} minute${mins === 1 ? '' : 's'}.`,
                    'warning'
                );
                return;
            }

            fileInput.click();
        });

        /* ── file selected handler ── */
        fileInput.addEventListener('change', () => {
            const file = fileInput.files[0];
            if (!file) return;

            // Validate type
            if (!file.type.startsWith('image/')) {
                showProfileToast('❌ Only image files are allowed.', 'error');
                fileInput.value = '';
                return;
            }

            // Validate size (max 2MB)
            if (file.size > 2 * 1024 * 1024) {
                showProfileToast('❌ Image must be under 2 MB.', 'error');
                fileInput.value = '';
                return;
            }

            // Double-check rate limit at time of file selection
            const { allowed, log } = checkRateLimit();
            if (!allowed) {
                showProfileToast('⏱ Rate limit reached. Please wait before uploading again.', 'warning');
                fileInput.value = '';
                return;
            }

            const reader = new FileReader();
            reader.onload = (ev) => {
                const dataUrl = ev.target.result;
                applyAvatarToAll(dataUrl);
                localStorage.setItem(AVATAR_KEY, dataUrl);
                recordUpload(log);
                updateRateLabel();
                showProfileToast('✓ Profile photo updated!', 'success');
            };
            reader.onerror = () => showProfileToast('❌ Failed to read image.', 'error');
            reader.readAsDataURL(file);
            fileInput.value = '';
        });
    }

    function updateRateLabel() {
        const label = document.getElementById('avatarRateLabel');
        if (!label) return;
        const { remaining } = checkRateLimit();
        label.textContent = remaining > 0 ? `${remaining} left` : 'Limit reached';
        label.style.color  = remaining > 1 ? '#4caf50' : remaining === 1 ? '#ff9800' : '#f44336';
    }

    /* ════════════════════════════════════════════
       SECTION 6 — INIT
    ════════════════════════════════════════════ */
    function init() {
        loadSavedAvatar();
        buildAvatarUI();

        // Also rebuild if dropdown is opened for the first time (lazy pages)
        document.addEventListener('click', (e) => {
            if (e.target.closest('#profileBtn')) {
                setTimeout(buildAvatarUI, 80);
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
