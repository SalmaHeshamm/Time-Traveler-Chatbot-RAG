import os, hashlib
from gtts import gTTS

def text_to_speech_ar(text, lang="ar", out_dir="audio"):
    os.makedirs(out_dir, exist_ok=True)
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    filename = f"tts_{h}.mp3"
    path = os.path.join(out_dir, filename)
    if not os.path.exists(path):
        tts = gTTS(text=text, lang=lang)
        tts.save(path)
    return path
