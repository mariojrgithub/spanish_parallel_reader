"""
Browser-native Text-to-Speech component for Streamlit.

Uses the Web Speech API (window.speechSynthesis), which runs entirely in the
user's browser. No audio data leaves the local machine. No Python TTS
dependencies are required.

Environment variables
---------------------
ENABLE_TTS    "true" (default) / "false" — master on/off switch.
TTS_LANGUAGE  BCP-47 language tag, default "es" (Spanish).
"""
from __future__ import annotations

import json
import os

import streamlit.components.v1 as components

ENABLE_TTS: bool = os.getenv("ENABLE_TTS", "true").strip().lower() not in (
    "false", "0", "no", "off"
)
TTS_LANGUAGE: str = os.getenv("TTS_LANGUAGE", "es").strip() or "es"

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
  function ttsspeak(){
    window.speechSynthesis.cancel();
    var u=new SpeechSynthesisUtterance(T);
    u.lang=L;
    function _speak(){
      var vv=window.speechSynthesis.getVoices();
      var v=vv.find(function(x){
        return x.lang.replace('_','-').startsWith(L)||x.lang.startsWith(L.split('-')[0]);
      });
      if(v)u.voice=v;
      u.onstart=function(){document.getElementById('ttss').textContent='Speaking\u2026';};
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


def render_tts_button(text: str, lang: str = TTS_LANGUAGE) -> None:
    """Render a browser-native 'Read aloud / Stop' button pair.

    No-ops silently when ``ENABLE_TTS`` is falsy or ``text`` is empty.
    The text is JSON-encoded before injection so quotes, backslashes, and
    Unicode are handled safely without manual escaping.
    """
    if not ENABLE_TTS or not text or not text.strip():
        return

    # json.dumps produces a properly escaped JS string literal including
    # surrounding double-quotes, e.g. "She said \"hola\""
    js_text = json.dumps(text)
    # lang is a BCP-47 tag; guard against injection by stripping quotes
    safe_lang = lang.replace('"', "").replace("'", "").replace("<", "").replace(">", "")

    html_content = (
        _TEMPLATE
        .replace("TTS_TEXT", js_text)
        .replace("TTS_LANG", safe_lang)
    )

    components.html(html_content, height=38, scrolling=False)
