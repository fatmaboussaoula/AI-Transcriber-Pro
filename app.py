import streamlit as st
import whisper
from gtts import gTTS
import os, base64, re
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.set_page_config(page_title="AI Transcriber Pro", page_icon="🎧", layout="wide")

# --------------------- STYLES (Glass + layout + cards) ---------------------
st.markdown("""
<style>
:root{
  --bg-1: rgba(10,14,20,1);
  --panel: rgba(17,25,40,0.62);
  --muted: #a7b0c2;
  --accent: #6ee7b7;
}
body {
  background: radial-gradient(circle at top left, #0b1220 0%, #03050a 60%);
  color: #e6eef8;
}
.css-18e3th9 { padding: 0rem 0rem; }
.main .block-container { padding-top: 1.6rem; padding-left: 2rem; padding-right: 2rem; }

.header {
  display:flex; align-items:center; justify-content:space-between;
  gap:12px; margin-bottom:18px;
}
.brand { display:flex; gap:12px; align-items:center; }
.logo {
  width:58px; height:58px; border-radius:12px;
  background: linear-gradient(135deg, #0ea5a, #7c3aed);
  display:flex; align-items:center; justify-content:center;
  font-weight:700; color:white;
  box-shadow:0 8px 24px rgba(0,0,0,0.6);
}
.glass{
    background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
    padding:20px; border-radius:16px;
    border:1px solid rgba(255,255,255,0.06);
    box-shadow: 0 10px 40px rgba(2,6,23,0.7);
    backdrop-filter: blur(12px);
}
.side-panel {
  background: rgba(255,255,255,0.02);
  padding:14px; border-radius:12px; border: 1px solid rgba(255,255,255,0.03);
}
.card {
  border-radius:12px; padding:12px;
  background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01));
  border:1px solid rgba(255,255,255,0.03); margin-bottom:10px;
}
.stButton>button {
  background: linear-gradient(90deg,#10b981,#06b6d4); color: #021018;
  font-weight:700; border-radius:10px; padding:8px 14px;
  box-shadow: 0 6px 18px rgba(10, 90, 80, 0.12);
}
.transcript { white-space: pre-wrap; line-height:1.5; }
.summary-badge { font-weight:700; color:var(--accent); margin-bottom:6px; }
.footer { color:var(--muted); font-size:12px; margin-top:10px; }
</style>
""", unsafe_allow_html=True)

# --------------------- Arabic utils ---------------------
def clean_ar(txt):
    txt = re.sub(r"[\u064B-\u065F]", "", txt)
    txt = re.sub(r"[^\u0600-\u06FF\.\؟\?،؛ ]"," ",txt)
    return re.sub(r"\s+"," ",txt).strip()

def summarize_ar(txt, n=4):
    parts = re.split(r"[\.!\؟\?،]+", clean_ar(txt))
    return "، ".join(p.strip() for p in parts[:n] if p.strip())

# --------------------- MODEL LOADING ---------------------
@st.cache_resource
def load_models():
    return (
        whisper.load_model("base", device="cpu"),
        T5Tokenizer.from_pretrained("t5-small"),
        T5ForConditionalGeneration.from_pretrained("t5-small")
    )

whisper_model, tok, t5 = load_models()

def transcribe(path):
    res = whisper_model.transcribe(path)
    return res["language"], res["text"], res["segments"]

def summarize_fr(txt):
    x = tok("summarize: "+txt, return_tensors="pt", truncation=True, max_length=512)
    y = t5.generate(x["input_ids"], max_length=150, min_length=25, num_beams=4)
    return tok.decode(y[0], skip_special_tokens=True)

# --------------------- TTS ---------------------
def generate_voice(text, lang, voice, out="voice.mp3"):

    # Avoid "file exists" error
    if os.path.exists("raw.mp3"):
        os.remove("raw.mp3")
    if os.path.exists(out):
        os.remove(out)

    gTTS(text=text, lang=lang).save("raw.mp3")

    if voice == "female":
        os.system(f'ffmpeg -i raw.mp3 -filter:a "asetrate=20000,atempo=1.13" -y "{out}"')

    elif voice == "male":
        os.system(f'ffmpeg -i raw.mp3 -filter:a "asetrate=13500,atempo=0.87" -y "{out}"')

    return out

def save_srt(seg):
    with open("subtitles.srt","w",encoding="utf-8") as s:
        for i,x in enumerate(seg,1):
            s.write(f"{i}\n00:{x['start']//60:02}:{x['start']%60:05.2f} --> "
                    f"00:{x['end']//60:02}:{x['end']%60:05.2f}\n{x['text'].strip()}\n\n")

def download(path,name):
    with open(path,"rb") as f:
        b64=base64.b64encode(f.read()).decode()
    st.markdown(f'<a download="{name}" href="data:file/octet-stream;base64,{b64}">⬇ {name}</a>',unsafe_allow_html=True)

# --------------------- SIDEBAR ---------------------
st.sidebar.title("Settings")
st.sidebar.subheader("🎙 Voice Type")


voice = st.sidebar.selectbox("Select Voice", ["female", "male"])

st.sidebar.markdown("---")
st.sidebar.info("FFmpeg required for voice effects.")
st.sidebar.markdown(" ")

# --------------------- HEADER ---------------------
st.markdown('<div class="header">', unsafe_allow_html=True)
st.markdown("""
<div class="brand">
  <div class="logo">AT</div>
  <div>
    <div style="font-size:18px;font-weight:800">AI Transcriber Pro</div>
    <div style="color:#9fb1c9;font-size:12px;margin-top:2px">Auto Mode — Audio → Text → Summary → Voice</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------- MAIN LAYOUT ---------------------
col1, col2, col3 = st.columns([1,2.2,1])

# LEFT PANEL
with col1:
    st.markdown('<div class="glass side-panel">', unsafe_allow_html=True)
    st.write("### 🎤 Record Audio")

    if st.button("Record 6 sec"):
        st.info("Recording...")
        fs=16000; duration=6
        audio = sd.rec(int(duration*fs), samplerate=fs, channels=1)
        sd.wait()
        write("mic.wav",fs,audio)
        st.success("Saved: mic.wav")
        st.audio("mic.wav")
        uploaded_audio="mic.wav"
    else:
        uploaded_audio=None

    st.markdown("</div>", unsafe_allow_html=True)

# CENTER PANEL
with col2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.write("### Upload Audio File")

    if not uploaded_audio:
        audio_file = st.file_uploader("Upload MP3 / WAV / FLAC", type=["mp3","wav","flac"])
    else:
        audio_file = None

    audio = uploaded_audio if uploaded_audio else audio_file

    if audio:
        if isinstance(audio,str):
            path="mic.wav"
        else:
            with open("temp.wav","wb") as f:
                f.write(audio.read())
            path="temp.wav"

        with st.spinner("Transcribing..."):
            lang,text,seg = transcribe(path)

        st.write(f"🌍 Language detected: **{lang.upper()}**")
        st.write("### Transcript")
        st.markdown(f'<div class="card transcript">{text}</div>', unsafe_allow_html=True)

        summary = summarize_fr(text) if lang=="fr" else summarize_ar(text)

        st.write("### Summary")
        st.markdown(f'<div class="card">{summary}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("🔊 Generate Voice"):
                out = generate_voice(summary, "fr" if lang=="fr" else "ar", voice)
                st.audio(out)
                st.success("Voice ready!")
                download(out,"summary.mp3")

        with c2:
            if st.button("💾 Save SRT"):
                save_srt(seg)
                st.success("subtitles.srt created")
                download("subtitles.srt","subtitles.srt")

    else:
        st.markdown('<div class="card">Upload or record audio to begin.</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT PANEL
with col3:
    st.markdown('<div class="glass side-panel">', unsafe_allow_html=True)
    st.write("### Project Summary — AI Transcriber Pro")
    st.write("""
- Detects language (FR/AR)  
- Transcribes audio → text  
- Generates summary  
- Exports SRT  
- Converts summary → voice (male/female)  
""")
    st.markdown("</div>", unsafe_allow_html=True)

# FOOTER
st.markdown('<div class="footer">Built with Whisper · T5 · Streamlit · gTTS · FFmpeg</div>', unsafe_allow_html=True)
