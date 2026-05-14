"""
Browser-native Text-to-Speech component for Streamlit.

Uses the Web Speech API (window.speechSynthesis), which runs entirely in the
user's browser. No audio data leaves the local machine. No Python TTS
dependencies are required.

Voice quality notes
-------------------
Quality depends entirely on the voices installed in the user's browser/OS:
- Chrome on any OS: loads Google's neural Spanish voices (e.g. "Google español
  de Estados Unidos") automatically when online. These sound near-natural.
- Firefox / Safari: use OS voices only. macOS has good "Monica"/"Paulina"
  voices; Windows built-in voices are lower quality.
- For the best experience, use Chrome and ensure an internet connection so
  that Google's network voices can be loaded.

Environment variables
---------------------
ENABLE_TTS    "true" (default) / "false" — master on/off switch.
TTS_LANGUAGE  Spanish BCP-47 language tag, default "es-MX".
TTS_RATE      Speech rate multiplier, default "0.9" (slightly slower than
              the browser default of 1.0; clearer for language learners).
"""
from __future__ import annotations

import json
import os
import re

import streamlit.components.v1 as components

ENABLE_TTS: bool = os.getenv("ENABLE_TTS", "true").strip().lower() not in (
    "false", "0", "no", "off"
)
TTS_LANGUAGE: str = os.getenv("TTS_LANGUAGE", "es-MX").strip() or "es-MX"
try:
    TTS_RATE: float = float(os.getenv("TTS_RATE", "0.9"))
    TTS_RATE = max(0.5, min(2.0, TTS_RATE))  # clamp to a safe range
except ValueError:
    TTS_RATE = 0.9

# Single HTML template — placeholders replaced at call time with json.dumps
# so arbitrary Unicode text and punctuation are safely embedded in JS.
_TEMPLATE = """\
<style>
  body{margin:0;padding:0;overflow:hidden;}
  .tw{display:flex;align-items:center;gap:6px;font-family:sans-serif;}
  .tb{
    cursor:pointer;border:1px solid #d1d5db;border-radius:6px;
    padding:3px 11px;font-size:13px;background:#fff;white-space:nowrap;
  }
  .tb:hover{background:#f3f4f6;}
  .ts{font-size:11px;color:#6b7280;}
</style>
<div class="tw" role="group" aria-label="Text-to-speech controls">
  <button class="tb" onclick="ttsspeak()"
          aria-label="Read Spanish text aloud">🔊 Read aloud</button>
  <button class="tb" onclick="ttsstop()"
          aria-label="Stop reading">⏹ Stop</button>
  <span class="ts" id="ttss" aria-live="polite"></span>
</div>
<script>
(function(){
  var T=TTS_TEXT;
  var L="TTS_LANG";
  var R=TTS_RATE;
  function _norm(tag){
    return (tag||'').replace('_','-').toLowerCase();
  }
  function _isSpanish(tag){
    return _norm(tag).startsWith('es');
  }
  // Latin American Spanish locale codes to prefer over Spain Spanish
  var _latamLocales=['es-mx','es-ar','es-cl','es-co','es-pe','es-ve','es-uy','es-ec','es-bo','es-py','es-hn','es-sv','es-ni','es-cr','es-pa','es-do','es-cu'];
  function _isLatinAmerican(lang){
    return _latamLocales.indexOf(_norm(lang))>=0;
  }
  // Neural (cloud) voices report localService===false in Chrome and are
  // dramatically higher quality than OS built-in voices.
  function _pickSpanishVoice(vv, preferred){
    var p=_norm(preferred);
    var base=p.split('-')[0]||'es';

    // Tier 1 — neural voice, exact locale (best quality, e.g. "Google español de Estados Unidos")
    var t1=vv.find(function(x){
      return !x.localService && _norm(x.lang)===p && _isSpanish(x.lang);
    });
    if(t1) return t1;

    // Tier 2 — local voice, exact locale
    var t2=vv.find(function(x){
      return _norm(x.lang)===p && _isSpanish(x.lang);
    });
    if(t2) return t2;

    // Tier 3 — neural voice, any Latin American Spanish locale (prefer over Spain)
    var t3=vv.find(function(x){
      return !x.localService && _isLatinAmerican(x.lang);
    });
    if(t3) return t3;

    // Tier 4 — local voice, any Latin American Spanish locale
    var t4=vv.find(function(x){
      return _isLatinAmerican(x.lang);
    });
    if(t4) return t4;

    // Tier 5 — neural voice, any Spanish locale (last resort before generic)
    var t5=vv.find(function(x){
      var xl=_norm(x.lang);
      return !x.localService && _isSpanish(x.lang) && (xl===base || xl.startsWith(base+'-'));
    });
    if(t5) return t5;

    // Tier 6 — any Spanish voice (fallback)
    return vv.find(function(x){ return _isSpanish(x.lang); }) || null;
  }
  function ttsspeak(){
    window.speechSynthesis.cancel();
    var u=new SpeechSynthesisUtterance(T);
    u.lang=L;
    u.rate=R;
    function _speak(){
      var vv=window.speechSynthesis.getVoices();
      var v=_pickSpanishVoice(vv,L);
      var label='Speaking\u2026';
      if(v){
        u.voice=v;
        u.lang=v.lang;
        label='Speaking\u2026 ('+(v.localService?'local':'neural')+')';
      } else {
        u.lang='es-MX';
      }
      u.onstart=function(){document.getElementById('ttss').textContent=label;};
      u.onend=function(){document.getElementById('ttss').textContent='';};
      u.onerror=function(e){document.getElementById('ttss').textContent='Error: '+e.error;};
      window.speechSynthesis.speak(u);
    }
    if(window.speechSynthesis.getVoices().length===0){
      window.speechSynthesis.addEventListener('voiceschanged',_speak,{once:true});
    } else {
      _speak();
    }
  }
  function ttsstop(){
    window.speechSynthesis.cancel();
    document.getElementById('ttss').textContent='';
  }
  window.ttsspeak=ttsspeak;
  window.ttsstop=ttsstop;
})();
</script>
"""


def render_tts_button(text: str, lang: str = TTS_LANGUAGE, rate: float = TTS_RATE) -> None:
    """Render a browser-native 'Read aloud / Stop' button pair.

    No-ops silently when ``ENABLE_TTS`` is falsy or ``text`` is empty.
    The text is JSON-encoded before injection so quotes, backslashes, and
    Unicode are handled safely without manual escaping.

    Parameters
    ----------
    text:
        The Spanish text to speak.
    lang:
        BCP-47 Spanish locale tag (e.g. ``"es-MX"``, ``"es-ES"``).
        Non-Spanish values are silently replaced with ``"es-MX"``.
    rate:
        Speech rate multiplier (0.5–2.0). Defaults to ``TTS_RATE`` (0.9).
    """
    if not ENABLE_TTS or not text or not text.strip():
        return

    js_text = json.dumps(text)
    candidate_lang = (lang or "").strip()
    safe_lang = re.sub(r"[^A-Za-z0-9_-]", "", candidate_lang).replace("_", "-")
    if not safe_lang.lower().startswith("es"):
        safe_lang = "es-MX"

    safe_rate = max(0.5, min(2.0, float(rate)))

    html_content = (
        _TEMPLATE
        .replace("TTS_TEXT", js_text)
        .replace('"TTS_LANG"', json.dumps(safe_lang))
        .replace("TTS_RATE", str(safe_rate))
    )

    components.html(html_content, height=38, scrolling=False)
