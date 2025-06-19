import streamlit as st
import requests
import tempfile
import os
import numpy as np
from moviepy.editor import concatenate_videoclips, ImageClip, CompositeVideoClip, AudioFileClip, TextClip, concatenate_audioclips, VideoFileClip
from PIL import Image
import io
from gtts import gTTS
import nltk
import re
import random
import wave
import struct

try:
    nltk.data.find('tokenizers.punkt')
except LookupError:
    nltk.download('punkt')

# ====== API Keys ======
PEXELS_API_KEY = ""           # ضع مفتاحك هنا (اختياري)
UNSPLASH_ACCESS_KEY = ""      # ضع مفتاحك هنا (اختياري)
PIXABAY_API_KEY = ""          # ضع مفتاحك هنا (اختياري)
COHERE_API_KEY = "K1GW0y2wWiwW7xlK7db7zZnqX7sxfRVGiWopVfCD"  # ضع مفتاح Cohere هنا

GTTS_VOICES = [
    {"name": "English (US) - Female", "lang": "en", "tld": "com"},
    {"name": "English (UK) - Female", "lang": "en", "tld": "co.uk"},
    {"name": "French (France) - Female", "lang": "fr", "tld": "fr"},
    {"name": "Spanish (Spain) - Female", "lang": "es", "tld": "es"},
    {"name": "German (Germany) - Female", "lang": "de", "tld": "de"},
]

# ========== مصادر الصور والفيديو ==========
def search_pexels_photos_with_desc(query, per_page=1):
    if not PEXELS_API_KEY: return []
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/v1/search?query={query}&per_page={per_page}"
    try:
        data = requests.get(url, headers=headers, timeout=10).json()
        return [("image", photo["src"]["large"], photo.get("alt") or query) for photo in data.get("photos", [])]
    except Exception: return []

def search_pexels_videos_with_desc(query, per_page=1):
    if not PEXELS_API_KEY: return []
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/videos/search?query={query}&per_page={per_page}"
    try:
        data = requests.get(url, headers=headers, timeout=10).json()
        result = []
        for v in data.get("videos", []):
            if v.get("video_files"):
                result.append(("video", v["video_files"][0]["link"], v.get("url", query)))
        return result
    except Exception: return []

def search_unsplash_photos_with_desc(query, per_page=1):
    if not UNSPLASH_ACCESS_KEY: return []
    url = f"https://api.unsplash.com/search/photos?query={query}&per_page={per_page}&client_id={UNSPLASH_ACCESS_KEY}"
    try:
        data = requests.get(url, timeout=10).json()
        return [("image", photo["urls"]["regular"], photo.get("alt_description") or query) for photo in data.get("results", [])]
    except Exception: return []

def search_pixabay_photos_with_desc(query, per_page=1):
    if not PIXABAY_API_KEY: return []
    url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={query}&per_page={per_page}&image_type=photo"
    try:
        data = requests.get(url, timeout=10).json()
        return [("image", hit["largeImageURL"], hit.get("tags", query)) for hit in data.get("hits", [])]
    except Exception: return []

def search_pixabay_videos_with_desc(query, per_page=1):
    if not PIXABAY_API_KEY: return []
    url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={query}&per_page={per_page}"
    try:
        data = requests.get(url, timeout=10).json()
        return [("video", v["videos"]["medium"]["url"], v.get("tags", query)) for v in data.get("hits", []) if "videos" in v]
    except Exception: return []

def search_wikimedia_photos_with_desc(query, limit=1):
    url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={query}&gsrlimit={limit}&prop=imageinfo|description&iiprop=url&format=json"
    try:
        response = requests.get(url, timeout=10)
        pages = response.json().get("query", {}).get("pages", {})
        result = []
        for v in pages.values():
            img_url = v.get("imageinfo", [{}])[0].get("url")
            desc = v.get("title", query)
            if img_url:
                result.append(("image", img_url, desc))
        return result
    except Exception: return []

def search_pollinations_photos_with_desc(prompt, per_page=1):
    results = []
    for _ in range(per_page):
        img_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
        results.append(("image", img_url, prompt))
    return results

# ========== ذكاء اصطناعي (Cohere) ==========
def generate_scene_prompt_via_cohere(sentence, topic):
    api_url = "https://api.cohere.ai/v1/generate"
    headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
    cohere_prompt = f"""For the documentary topic "{topic}", generate a creative, detailed, visual English prompt for an AI image generator, based on this scene description: "{sentence}". The prompt should include scene, style, setting, mood, lighting, and avoid mentioning 'photo', 'picture', 'render', or 'image'. Make it suitable for Pollinations or Stable Diffusion. Example: "A futuristic BMW electric SUV driving on a scenic mountain road at sunrise, vibrant colors, cinematic, high detail, documentary style"
Prompt:
"""
    data = {"model": "command", "prompt": cohere_prompt, "max_tokens": 70, "temperature": 0.8}
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=25)
        if response.status_code == 200:
            text = response.json()["generations"][0]["text"].strip()
            text = re.sub(r"^(Prompt:)?\s*", '', text)
            return text
        else:
            st.error(f"Cohere error generating scene prompt: {response.status_code} - {response.text}")
            return sentence
    except Exception as e:
        st.error(f"Cohere Scene Prompt Exception: {e}")
        return sentence

def generate_script_via_cohere(topic, num_media, cohere_temp=0.4):
    api_url = "https://api.cohere.ai/v1/generate"
    headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
    cohere_prompt = f"""Write a smooth, well-connected, short documentary script about "{topic}" in {num_media} sentences. Each sentence continues the previous, as if the viewer is following a story."""
    data = {"model": "command","prompt": cohere_prompt,"max_tokens": 300,"temperature": cohere_temp}
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=25)
        if response.status_code == 200:
            text = response.json()["generations"][0]["text"].strip()
            return text
        elif response.status_code == 429:
            st.warning("تجاوزت الحد الأقصى للطلبات في الدقيقة على Cohere. انتظر دقيقة ثم حاول مجددًا أو استخدم مفتاح Production.")
            return ""
        else:
            st.error(f"Cohere error generating script: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        st.error(f"Cohere Script Exception: {e}")
        return ""

def filter_script_sentences(raw_text, num_media):
    try: sentences = nltk.sent_tokenize(raw_text)
    except Exception: sentences = raw_text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    filtered = [s for s in sentences if len(s) > 5]
    if len(filtered) > num_media: filtered = filtered[:num_media]
    while len(filtered) < num_media: filtered.append("...")
    return filtered

# ========== أدوات الصوت والصورة ==========
def safe_download_and_convert_image(media_url, temp_files):
    try:
        response = requests.get(media_url, timeout=10)
        img_data = response.content
        if len(img_data) < 80_000: return None
        img_bytes = io.BytesIO(img_data)
        with Image.open(img_bytes) as pil_img:
            pil_img = pil_img.convert("RGB")
            width, height = pil_img.size
            if width < 50 or height < 50: return None
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
                pil_img.save(tmp_img.name)
                img_path = tmp_img.name
                temp_files.append(img_path)
        return img_path
    except Exception: return None

def get_audio_duration(audio_path):
    try:
        ac = AudioFileClip(audio_path)
        duration = ac.duration
        ac.close()
        return duration
    except Exception: return 2

def animated_text_clip(img_clip, text, duration, lang="en", font_size=40, color="white", text_pos="bottom"):
    txt = TextClip(
        text, fontsize=font_size, color=color, font="Arial",
        size=img_clip.size, method='caption', align='center'
    ).set_duration(duration)
    margin = 30
    try: h = txt.h
    except: h = font_size + 10
    if text_pos == "bottom":
        txt = txt.set_position(("center", img_clip.h - h - margin))
    elif text_pos == "top":
        txt = txt.set_position(("center", margin))
    elif text_pos == "center":
        txt = txt.set_position("center")
    else:
        txt = txt.set_position(text_pos)
    return CompositeVideoClip([img_clip, txt]).set_duration(duration)

def ken_burns_effect(img_clip, duration, zoom=1.08, pan_direction="random"):
    w, h = img_clip.size
    directions = ["left_to_right", "top_to_bottom", "right_to_left", "bottom_to_top"]
    if pan_direction == "random":
        pan_direction = random.choice(directions)
    if pan_direction == "left_to_right":
        start = (0, 0)
        end = (w * (zoom - 1), 0)
    elif pan_direction == "right_to_left":
        start = (w * (zoom - 1), 0)
        end = (0, 0)
    elif pan_direction == "top_to_bottom":
        start = (0, 0)
        end = (0, h * (zoom - 1))
    elif pan_direction == "bottom_to_top":
        start = (0, h * (zoom - 1))
        end = (0, 0)
    else:
        start = (0, 0)
        end = (w * (zoom - 1), 0)
    def crop_func(get_frame, t):
        f = get_frame(t)
        frac = t / duration
        x = int(start[0] + (end[0] - start[0]) * frac)
        y = int(start[1] + (end[1] - start[1]) * frac)
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)
        return f[y:y+crop_h, x:x+crop_w]
    return img_clip.fl(crop_func, apply_to=["mask"]).resize(img_clip.size).set_duration(duration)

def resize_and_letterbox(img_clip, target_w=1280, target_h=720):
    img_clip = img_clip.resize(height=target_h)
    if img_clip.w > target_w:
        img_clip = img_clip.crop(x_center=img_clip.w/2, width=target_w)
    elif img_clip.w < target_w:
        img_clip = img_clip.margin(left=(target_w-img_clip.w)//2, right=(target_w-img_clip.w)//2, color=(0,0,0))
    return img_clip

def safe_tts_save(text, mp3_path, lang, tld):
    if not text or not text.strip() or text.strip() == "...":
        with wave.open(mp3_path, 'w') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(22050)
            duration_seconds = 1
            samples = [0] * int(22050 * duration_seconds)
            for s in samples:
                f.writeframes(struct.pack('<h', s))
        return
    tts = gTTS(text=text, lang=lang, tld=tld)
    tts.save(mp3_path)

def assemble_video(
    montage, out_path, color="#FFFFFF", text_size=32, text_pos="bottom",
    youtube_export=False, text_anim_lang="en"
):
    clips = []
    audio_clips = []
    temp_files = []
    for idx, (media_type, media_url, audio_path, sent, ken_burns_params) in enumerate(montage):
        try:
            duration = get_audio_duration(audio_path)
            audio_clip = AudioFileClip(audio_path)
            audio_clips.append(audio_clip)
            if media_type == "video":
                try:
                    clip = VideoFileClip(media_url)
                    if clip.duration > duration:
                        clip = clip.subclip(0, duration)
                    else:
                        clip = clip.set_duration(duration)
                    clip = clip.resize(height=720)
                    if clip.w > 1280:
                        clip = clip.crop(x_center=clip.w/2, width=1280)
                    elif clip.w < 1280:
                        clip = clip.margin(left=(1280-clip.w)//2, right=(1280-clip.w)//2, color=(0,0,0))
                    anim_txt = animated_text_clip(
                        clip.set_duration(duration),
                        sent,
                        duration,
                        lang=text_anim_lang,
                        font_size=text_size,
                        color=color,
                        text_pos=text_pos
                    )
                    clips.append(anim_txt)
                except Exception:
                    continue
            elif media_type == "image":
                img_path = media_url
                if isinstance(img_path, str) and img_path.startswith("http"):
                    img_path = safe_download_and_convert_image(img_path, temp_files)
                    if img_path is None:
                        continue
                try:
                    pil = Image.open(img_path).resize((1280, 720))
                    img_clip = ImageClip(np.array(pil))
                    img_clip = resize_and_letterbox(img_clip, target_w=1280, target_h=720)
                except Exception:
                    continue
                img_clip = img_clip.set_duration(duration)
                if ken_burns_params is not None:
                    zoom, pan = ken_burns_params
                    img_clip = ken_burns_effect(img_clip, duration, zoom=zoom, pan_direction=pan)
                anim_txt = animated_text_clip(
                    img_clip,
                    sent,
                    duration,
                    lang=text_anim_lang,
                    font_size=text_size,
                    color=color,
                    text_pos=text_pos
                )
                clips.append(anim_txt)
        except Exception:
            continue
    if not clips or not audio_clips:
        st.error("Could not build the final video.")
        return None, None
    final_audio = concatenate_audioclips(audio_clips)
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip = final_clip.set_audio(final_audio)
    final_clip = final_clip.subclip(0, final_audio.duration)
    if youtube_export:
        final_clip = final_clip.resize(height=720)
        if final_clip.w != 1280:
            final_clip = final_clip.crop(x_center=final_clip.w/2, width=1280, height=720)
    final_clip = final_clip.fadein(1).fadeout(1)
    final_clip.write_videofile(out_path, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, fps=15)
    for c in clips: c.close()
    for a in audio_clips: a.close()
    final_clip.close()
    for f in temp_files:
        try: os.remove(f)
        except Exception: pass
    return out_path, final_audio.duration

# ========== Streamlit App ==========
st.set_page_config(page_title="AI Documentary Generator", layout="wide")
st.title("🎬 AI Documentary Generator (Images, Video, Voice-over)")

if st.button("ابدأ مجدداً (Start new project)"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]

if "step" not in st.session_state:
    st.session_state["step"] = "start"

if st.session_state["step"] == "start":
    topic = st.text_input("Video topic (e.g., BMW iX M70)")
    num_media = st.slider("Number of scenes:", min_value=2, max_value=10, value=5)
    script_mode = st.radio("Script source:", ["AI-generated script (Cohere)", "Write script manually"], index=0)
    script_text = ""
    cohere_temp = st.slider("Creativity:", 0.1, 1.0, 0.4, step=0.05)
    sources_selected = st.multiselect(
        "Media sources:",
        options=["Pollinations", "Pexels", "Pixabay", "Unsplash", "Wikimedia"],
        default=["Pollinations", "Pexels", "Pixabay", "Unsplash", "Wikimedia"]
    )
    pollinations_n_images = st.slider("Number of Pollinations AI options per scene", 1, 2, 1)
    ken_burns_on = st.checkbox("Apply Ken Burns effect (pan/zoom) for images", value=True)
    if ken_burns_on:
        ken_burns_zoom = st.slider("Ken Burns: Zoom factor", min_value=1.01, max_value=1.2, value=1.08, step=0.01)
        ken_burns_random_pan = st.checkbox("Random pan direction", value=True)
    color = st.color_picker("Text color", "#ffffff")
    text_size = st.slider("Text size", 14, 60, 28)
    text_pos = st.radio("Text position", options=["top", "center", "bottom"], index=2)
    youtube_export = st.checkbox("YouTube export (16:9)", value=True)
    voice_choice = st.selectbox("Voice-over voice:", [v["name"] for v in GTTS_VOICES])
    voice_data = next(v for v in GTTS_VOICES if v["name"] == voice_choice)
    if script_mode == "Write script manually":
        script_text = st.text_area("Write your documentary script here:", height=300)
    if st.button("Generate!"):
        if script_mode == "Write script manually" and not script_text.strip():
            st.error("Please enter the script text.")
        elif script_mode != "Write script manually" and not topic.strip():
            st.error("Please enter a topic.")
        elif not COHERE_API_KEY and script_mode == "AI-generated script (Cohere)":
            st.error("Cohere API key not found!")
        else:
            if script_mode == "AI-generated script (Cohere)":
                raw_script = generate_script_via_cohere(topic, num_media, cohere_temp)
                if not raw_script.strip():
                    st.warning("لم يتم توليد السكريبت! قد يكون هناك خطأ في Cohere أو تجاوزت الحد.")
                    st.stop()
                sentences = filter_script_sentences(raw_script, num_media)
                final_text = "\n".join(sentences)
            else:
                final_text = script_text.strip()
                sentences = filter_script_sentences(final_text, num_media)
            montage_choices = []
            for idx, sent in enumerate(sentences):
                ai_prompt = generate_scene_prompt_via_cohere(sent, topic)
                all_media = []
                if "Pollinations" in sources_selected:
                    all_media += search_pollinations_photos_with_desc(ai_prompt, per_page=pollinations_n_images)
                if "Pexels" in sources_selected:
                    all_media += search_pexels_photos_with_desc(sent, per_page=1)
                    all_media += search_pexels_videos_with_desc(sent, per_page=1)
                if "Pixabay" in sources_selected:
                    all_media += search_pixabay_photos_with_desc(sent, per_page=1)
                    all_media += search_pixabay_videos_with_desc(sent, per_page=1)
                if "Unsplash" in sources_selected:
                    all_media += search_unsplash_photos_with_desc(sent, per_page=1)
                if "Wikimedia" in sources_selected:
                    all_media += search_wikimedia_photos_with_desc(sent, limit=1)
                if all_media:
                    media_type, media_url, desc = all_media[0]
                else:
                    media_type, media_url, desc = "image", "", ai_prompt
                ken_burns_params = (ken_burns_zoom if ken_burns_on else 1.0, "random" if ken_burns_on and ken_burns_random_pan else "left_to_right")
                montage_choices.append((media_type, media_url, sent, ken_burns_params))
            st.session_state["editable_script"] = final_text
            st.session_state["montage_choices"] = montage_choices
            st.session_state["last_num_media"] = num_media
            st.session_state["voice_choice"] = voice_choice
            st.session_state["color"] = color
            st.session_state["text_size"] = text_size
            st.session_state["text_pos"] = text_pos
            st.session_state["youtube_export"] = youtube_export
            st.session_state["step"] = "edit_script"

if st.session_state["step"] == "edit_script" and st.session_state.get("editable_script"):
    st.markdown("### ✏️ Edit the script, then click Build Video:")
    script_edit = st.text_area("Script (edit before building video):",
                               value=st.session_state["editable_script"], height=250, key="script_editbox")
    if st.button("Build video / Rebuild after edit"):
        temp_files = []
        sentences = filter_script_sentences(script_edit, st.session_state["last_num_media"])
        voice_data = next(v for v in GTTS_VOICES if v["name"] == st.session_state["voice_choice"])
        montage = []
        for idx, (media_type, media_url, sent, ken_burns_params) in enumerate(st.session_state["montage_choices"]):
            scene_sent = sentences[idx] if idx < len(sentences) else sent
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
                safe_tts_save(scene_sent, tmp_mp3.name, voice_data["lang"], voice_data["tld"])
                mp3_path = tmp_mp3.name
                temp_files.append(mp3_path)
            montage.append((media_type, media_url, mp3_path, scene_sent, ken_burns_params))
        if not montage:
            st.error("No valid scenes for the video.")
            st.stop()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            out_video_path = tmp_video.name
        final_video, video_duration_sec = assemble_video(
            montage,
            out_path=out_video_path,
            color=st.session_state["color"],
            text_size=st.session_state["text_size"],
            text_pos=st.session_state["text_pos"],
            youtube_export=st.session_state["youtube_export"],
            text_anim_lang=voice_data["lang"]
        )
        st.session_state["rendered_video"] = final_video
        st.session_state["video_duration_sec"] = video_duration_sec
        st.session_state["step"] = "video_ready"

if st.session_state["step"] == "video_ready" and st.session_state.get("rendered_video"):
    st.success("Done! See your result 👇")
    st.video(st.session_state["rendered_video"])
    st.info(f"Video duration: {st.session_state['video_duration_sec']/60:.2f} min ({st.session_state['video_duration_sec']:.1f} sec)")
    with open(st.session_state["rendered_video"], "rb") as f:
        st.download_button(label="Download Video", data=f, file_name="documentary_video.mp4", mime="video/mp4")
