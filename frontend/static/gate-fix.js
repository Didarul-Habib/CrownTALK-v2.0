// CrownTALK gate hotfix: binds Enter + lock click, saves key, hides overlay, boots UI.
// No class/id/style changes required. Safe to keep deployed forever.

(function(){
  const AUTH_FLAG   = 'crowntalk_access_v1';
  const KEY_STORAGE = 'crowntalk_key_v1';
  const COOKIE_AUTH = 'crowntalk_access_v1';
  const COOKIE_KEY  = 'crowntalk_key_v1';

  function saveKey(key) {
    try { localStorage.setItem(KEY_STORAGE, key); } catch {}
    try { sessionStorage.setItem(KEY_STORAGE, key); } catch {}
    try { document.cookie = `${COOKIE_KEY}=${encodeURIComponent(key)}; max-age=${365*24*3600}; path=/; samesite=lax`; } catch {}
  }
  function markAuthorized() {
    try { localStorage.setItem(AUTH_FLAG, '1'); } catch {}
    try { sessionStorage.setItem(AUTH_FLAG, '1'); } catch {}
    try { document.cookie = `${COOKIE_AUTH}=1; max-age=${365*24*3600}; path=/; samesite=lax`; } catch {}
  }
  function hideGate(gate){
    if (!gate) return;
    gate.hidden = true;
    gate.style.display = 'none';
    document.body.style.overflow = '';
  }
  function tryAuth(){
    const gate  = document.getElementById('adminGate');
    if (!gate) return;
    const input = gate.querySelector('#password') || gate.querySelector('input[type="password"]') || gate.querySelector('input');
    const val = (input && input.value || '').trim();
    if (!val) {
      if (input) {
        input.classList.add('ct-shake');
        setTimeout(()=>input.classList.remove('ct-shake'), 350);
      }
      return;
    }
    saveKey(val);
    markAuthorized();
    hideGate(gate);

    // Prefer your main submit path if present
    if (window.CTGate && typeof window.CTGate.submit === 'function') {
      try { window.CTGate.submit(); } catch {}
    }
    // Or just boot the app UI if defined globally
    else if (typeof window.bootAppUI === 'function') {
      try { window.bootAppUI(); } catch {}
    }
  }

  function bind(){
    const gate  = document.getElementById('adminGate');
    if (!gate) return;

    const input = gate.querySelector('#password') || gate.querySelector('input[type="password"]') || gate.querySelector('input');
    const lock  = gate.querySelector('svg');

    // Enter in input
    if (input) {
      input.addEventListener('keydown', (e)=>{
        if (e.key === 'Enter') { e.preventDefault(); tryAuth(); }
      }, { once:false });
    }
    // Click lock
    if (lock) {
      lock.style.cursor = 'pointer';
      lock.addEventListener('click', (e)=>{ e.preventDefault(); tryAuth(); }, { once:false });
    }
    // If gate was already filled and listeners were missing, pressing Enter/click now works.
  }

  // Bind late to survive other script load orders
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bind);
  } else {
    bind();
  }
})();
