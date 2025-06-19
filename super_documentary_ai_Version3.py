import streamlit as st
import requests
import tempfile
import os
import json
import numpy as np
from moviepy.editor import concatenate_videoclips, ImageClip, CompositeVideoClip, AudioFileClip, TextClip, concatenate_audioclips, VideoFileClip, vfx
from PIL import Image
import io
from gtts import gTTS
import nltk
import re
import random
import wave
import struct
import psutil

try:
    nltk.data.find('tokenizers.punkt')
except LookupError:
    nltk.download('punkt')

PEXELS_API_KEY = "pLcIoo3oNdhqna28AfdaBYhkE3SFps9oRGuOsxY3JTe92GcVDZpwZE9i"
UNSPLASH_ACCESS_KEY = "SDK5avSHNm9lcNhhLhT_SzUdzd98hYX0BVjswi3ZHzU"
PIXABAY_API_KEY = "50380897-76243eaec536038f687ff8e15"
COHERE_API_KEY = "K1GW0y2wWiwW7xlK7db7zZnqX7sxfRVGiWopVfCD"

GTTS_VOICES = [
    {"name": "English (US) - Female", "lang": "en", "tld": "com"},
    {"name": "English (UK) - Female", "lang": "en", "tld": "co.uk"},
    {"name": "French (France) - Female", "lang": "fr", "tld": "fr"},
    {"name": "Spanish (Spain) - Female", "lang": "es", "tld": "es"},
    {"name": "German (Germany) - Female", "lang": "de", "tld": "de"},
]

def print_memory_usage(tag=""):
    mem = psutil.virtual_memory()
    msg = f"🔋 RAM used {mem.used // (1024*1024)}MB / {mem.total // (1024*1024)}MB  ({mem.percent}%) [{tag}]"
    print(msg)
    st.info(msg)

def safe_download_and_convert_image(media_url, temp_files):
    try:
        response = requests.get(media_url, timeout=10)
        img_data = response.content
        if len(img_data) < 80_000:
            print(f"Image too small in bytes: {media_url} ({len(img_data)} bytes)")
            return None
        img_bytes = io.BytesIO(img_data)
        with Image.open(img_bytes) as pil_img:
            pil_img = pil_img.convert("RGB")
            width, height = pil_img.size
            if width < 50 or height < 50:
                print(f"Image too small in dimensions: {media_url} ({width}x{height})")
                return None
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
                pil_img.save(tmp_img.name)
                img_path = tmp_img.name
                temp_files.append(img_path)
        return img_path
    except Exception as e:
        print(f"Failed to process image: {media_url}, error: {e}")
        return None

def search_pexels_photos_with_desc(query, per_page=1):
    if not PEXELS_API_KEY: return []
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/v1/search?query={query}&per_page={per_page}"
    try:
        data = requests.get(url, headers=headers, timeout=10).json()
        return [("image", photo["src"]["large"], photo.get("alt") or query) for photo in data.get("photos", [])]
    except Exception as e:
        print(f"Pexels photos error: {e}")
        return []

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
    except Exception as e:
        print(f"Pexels videos error: {e}")
        return []

def search_unsplash_photos_with_desc(query, per_page=1):
    if not UNSPLASH_ACCESS_KEY: return []
    url = f"https://api.unsplash.com/search/photos?query={query}&per_page={per_page}&client_id={UNSPLASH_ACCESS_KEY}"
    try:
        data = requests.get(url, timeout=10).json()
        return [("image", photo["urls"]["regular"], photo.get("alt_description") or query) for photo in data.get("results", [])]
    except Exception as e:
        print(f"Unsplash error: {e}")
        return []

def search_pixabay_photos_with_desc(query, per_page=1):
    if not PIXABAY_API_KEY: return []
    url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={query}&per_page={per_page}&image_type=photo"
    try:
        data = requests.get(url, timeout=10).json()
        return [("image", hit["largeImageURL"], hit.get("tags", query)) for hit in data.get("hits", [])]
    except Exception as e:
        print(f"Pixabay error: {e}")
        return []

def search_pixabay_videos_with_desc(query, per_page=1):
    if not PIXABAY_API_KEY: return []
    url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={query}&per_page={per_page}"
    try:
        data = requests.get(url, timeout=10).json()
        return [("video", v["videos"]["medium"]["url"], v.get("tags", query)) for v in data.get("hits", []) if "videos" in v]
    except Exception as e:
        print(f"Pixabay videos error: {e}")
        return []

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
    except Exception as e:
        print(f"Wikimedia error: {e}")
        return []

def search_pollinations_photos_with_desc(prompt, per_page=1):
    results = []
    # allow user to specify n_images for each scene (customization)
    for _ in range(per_page):
        img_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
        results.append(("image", img_url, prompt))
    return results

def generate_script_with_cohere(prompt, max_tokens=1000, temperature=0.7, model="command"):
    url = "https://api.cohere.ai/v1/generate"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["generations"][0]["text"]
    else:
        st.error(f"Cohere API error: {response.status_code}\n{response.text}")
        return ""

def generate_script_from_media_cohere(media_list, topic, lang="en", max_tokens=1000, temperature=0.4):
    prompt = f"""I have a collection of photos and videos about "{topic}":\n"""
    for i, (_, url, desc) in enumerate(media_list, 1):
        prompt += f"{i}. {desc.strip()}\n"
    prompt += f"""
Write a short, smooth, documentary script (one story, not disconnected sentences) that covers these photos and videos in order, without mentioning the word "photo", "scene", or numbers, and no repetition.
"""
    return generate_script_with_cohere(prompt, max_tokens=max_tokens, temperature=temperature)

def filter_script_sentences(raw_text, num_media):
    try:
        sentences = nltk.sent_tokenize(raw_text)
    except Exception:
        sentences = raw_text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    filtered = []
    for s in sentences:
        if len(s) > 5:
            filtered.append(s)
    if len(filtered) > num_media:
        filtered = filtered[:num_media]
    while len(filtered) < num_media:
        filtered.append("...")
    return filtered

def get_audio_duration(audio_path):
    try:
        ac = AudioFileClip(audio_path)
        duration = ac.duration
        ac.close()
        return duration
    except Exception as e:
        print(f"Audio duration error: {e}")
        return 2

def animated_text_clip(img_clip, text, duration, lang="en", mode="sentence", group_size=1, font_size=40, color="white", text_pos="bottom"):
    items = [text]
    item_dur = duration
    txt_clips = []
    for i, item in enumerate(items):
        font = "Arial"
        if lang == "fr":
            font = "Liberation-Serif"
        txt = TextClip(
            item, fontsize=font_size, color=color, font=font,
            size=img_clip.size, method='caption', align='center'
        ).set_duration(item_dur).set_start(i * item_dur)
        margin = 30
        try:
            h = txt.h
        except:
            h = font_size + 10
        if text_pos == "bottom":
            txt = txt.set_position(("center", img_clip.h - h - margin))
        elif text_pos == "top":
            txt = txt.set_position(("center", margin))
        elif text_pos == "center":
            txt = txt.set_position("center")
        else:
            txt = txt.set_position(text_pos)
        txt_clips.append(txt)
    return CompositeVideoClip([img_clip] + txt_clips).set_duration(duration)

def ken_burns_effect(img_clip, duration, zoom=1.08, pan_direction="random"):
    w, h = img_clip.size
    # Ken Burns effect: zoom + pan
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

def random_watermark_positions(duration, w, h, txt_w=200, txt_h=30, step=0.5):
    positions = []
    t = 0
    while t < duration:
        x = random.randint(0, max(0, w-txt_w))
        y = random.randint(0, max(0, h-txt_h))
        positions.append((t, (x, y)))
        t += step
    return positions

def choose_music_for_topic(topic):
    topic = topic.lower()
    if "nature" in topic:
        return "music/nature.mp3"
    elif "car" in topic:
        return "music/cars.mp3"
    elif "space" in topic:
        return "music/space.mp3"
    elif "history" in topic:
        return "music/history.mp3"
    else:
        return "music/default.mp3"

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
    logo_path=None, music_path=None, watermark_text="", gif_export=False, square_export=False, youtube_export=False,
    text_anim_mode="sentence", text_anim_group_size=1, text_anim_lang="en"
):
    clips = []
    audio_clips = []
    temp_files = []
    print_memory_usage("Start assemble_video")
    for idx, (media_type, media_url, audio_path, sent, ken_burns_params) in enumerate(montage):
        print(f"Scene {idx+1}: media_type={media_type}, media_url={media_url}")
        print_memory_usage(f"scene_{idx+1}_start")
        try:
            duration = get_audio_duration(audio_path)
            audio_clip = AudioFileClip(audio_path)
            audio_clips.append(audio_clip)
            if media_type == "video":
                print(f"Processing video: {media_url}")
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
                        mode=text_anim_mode,
                        group_size=text_anim_group_size,
                        font_size=text_size,
                        color=color,
                        text_pos=text_pos
                    )
                    clips.append(anim_txt)
                except Exception as e:
                    print(f"Video error: {e}, skipping video scene.")
                    continue
            elif media_type == "image":
                print(f"Processing image: {media_url}")
                img_path = media_url
                if isinstance(img_path, str) and img_path.startswith("http"):
                    img_path = safe_download_and_convert_image(img_path, temp_files)
                    if img_path is None:
                        print(f"Skipping image (bad, too small, or too small in bytes): {media_url}")
                        continue
                try:
                    if hasattr(Image, 'Resampling'):
                        pil = Image.open(img_path).resize((1280, 720), Image.Resampling.LANCZOS)
                        img_clip = ImageClip(np.array(pil))
                    else:
                        img_clip = ImageClip(img_path)
                    img_clip = resize_and_letterbox(img_clip, target_w=1280, target_h=720)
                except Exception as e:
                    print(f"ImageClip error: {e}")
                    continue
                img_clip = img_clip.set_duration(duration)
                # تطبيق Ken Burns إذا متاح
                if ken_burns_params is not None:
                    zoom, pan = ken_burns_params
                    img_clip = ken_burns_effect(img_clip, duration, zoom=zoom, pan_direction=pan)
                anim_txt = animated_text_clip(
                    img_clip,
                    sent,
                    duration,
                    lang=text_anim_lang,
                    mode=text_anim_mode,
                    group_size=text_anim_group_size,
                    font_size=text_size,
                    color=color,
                    text_pos=text_pos
                )
                clips.append(anim_txt)
            print(f"Scene {idx+1} processed successfully.")
            print_memory_usage(f"scene_{idx+1}_end")
        except Exception as e:
            print(f"Error in scene {idx+1}: {e}")
    if not clips or not audio_clips:
        st.error("Could not build the final video.")
        return None, None
    print_memory_usage("Before audio/video concat")
    final_audio = concatenate_audioclips(audio_clips)
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip = final_clip.set_audio(final_audio)
    final_clip = final_clip.subclip(0, final_audio.duration)
    if youtube_export:
        final_clip = final_clip.resize(height=720)
        if final_clip.w != 1280:
            final_clip = final_clip.crop(x_center=final_clip.w/2, width=1280, height=720)
    if logo_path:
        logo = (ImageClip(logo_path)
                .set_duration(final_clip.duration)
                .resize(height=50)
                .set_pos(("right", "top")).margin(right=8, top=8, opacity=0))
        final_clip = CompositeVideoClip([final_clip, logo])
    if not music_path or not os.path.exists(music_path):
        music_path_auto = choose_music_for_topic(st.session_state.get("topic","") if "topic" in st.session_state else "")
        if os.path.exists(music_path_auto):
            music_path = music_path_auto
    if music_path and os.path.exists(music_path):
        try:
            music_clip = AudioFileClip(music_path).volumex(0.15)
            final_audio = concatenate_audioclips([final_clip.audio, music_clip])
            final_clip = final_clip.set_audio(final_audio)
        except Exception as e:
            print(f"Music error: {e}")
    if watermark_text:
        try:
            txt_clip = TextClip(
                watermark_text, fontsize=24, color='white', font='Arial-Bold',
                size=(200, 30)
            ).set_duration(final_clip.duration).set_opacity(0.4)
            positions = random_watermark_positions(final_clip.duration, final_clip.w, final_clip.h, 200, 30, step=0.5)
            def moving_position(t):
                idx = int(t // 0.5)
                return positions[idx][1] if idx < len(positions) else positions[-1][1]
            txt_clip = txt_clip.set_position(moving_position)
            final_clip = CompositeVideoClip([final_clip, txt_clip])
        except Exception as e:
            print(f"Watermark error: {e}")
    print_memory_usage("Before write_videofile")
    final_clip = final_clip.fadein(1).fadeout(1)
    final_clip.write_videofile(out_path, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, fps=15)
    print_memory_usage("After write_videofile")
    for c in clips:
        c.close()
    for a in audio_clips:
        a.close()
    final_clip.close()
    for f in temp_files:
        try:
            os.remove(f)
        except Exception:
            pass
    print_memory_usage("End assemble_video")
    return out_path, final_audio.duration

def save_project(project_data, path="saved_project.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project_data, f, ensure_ascii=False, indent=2)

def load_project(path="saved_project.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

st.set_page_config(page_title="AI Documentary Generator", layout="wide")
st.title("🎬 AI Documentary Generator (Images, Video, Voice-over)")

mode = st.radio("Project Type", ["New Project", "Restore Project"])
auto_project_file = "saved_project.json"

if mode == "Restore Project":
    uploaded_project = st.file_uploader("Upload project file (json):", type="json")
    if uploaded_project:
        project_data = json.load(uploaded_project)
        st.success("Project restored!")
        st.json(project_data)
        st.session_state["restored_project"] = project_data
elif os.path.exists(auto_project_file):
    if st.button("Restore last project (auto-save)"):
        project_data = load_project(auto_project_file)
        st.success("Last auto-saved project loaded.")
        st.json(project_data)
        st.session_state["restored_project"] = project_data
else:
    st.markdown("**Enter your topic, choose number of scenes, select media sources, and let AI create a documentary video!**")
    topic = st.text_input("Video topic (e.g., Smart Cars)")
    st.session_state["topic"] = topic
    num_media = st.slider("Number of scenes:", min_value=2, max_value=30, value=5)
    script_mode = st.radio(
        "Script source:",
        ["AI-generated script (Cohere)", "Script from media (Cohere)", "Write script manually"], index=0)
    script_text = ""
    cohere_tokens = st.slider("Approximate script length (tokens):", 100, 10000, 1000, step=50)
    cohere_temp = st.slider("Creativity:", 0.1, 1.0, 0.4, step=0.05)
    if script_mode == "Write script manually":
        script_text = st.text_area("Write your documentary script here:", height=300)
    sources_selected = st.multiselect(
        "Photo/Video sources:",
        options=["Pexels", "Unsplash", "Pixabay", "Wikimedia", "Pollinations"],  # Add more if available
        default=["Pexels", "Unsplash", "Pixabay", "Wikimedia"]
    )
    logo_file = st.file_uploader("Logo (optional):", type=["png", "jpg", "jpeg"])
    music_file = st.file_uploader("Background music (optional):", type=["mp3", "wav"])
    youtube_export = st.checkbox("YouTube export (16:9)", value=True)
    watermark = st.text_input("Watermark text (optional):", value="@SuperAI")
    color = st.color_picker("Text color", "#ffffff")
    text_size = st.slider("Text size", 14, 60, 28)
    text_pos = st.radio("Text position", options=["top", "center", "bottom"], index=2)
    gif_export = st.checkbox("Export as GIF", value=False)
    square_export = st.checkbox("Export square video (Instagram)", value=False)
    voice_choice = st.selectbox("Voice-over voice:", [v["name"] for v in GTTS_VOICES])
    voice_data = next(v for v in GTTS_VOICES if v["name"] == voice_choice)
    text_anim_mode_val = "sentence"
    text_anim_group_size = 1
    text_anim_lang = voice_data["lang"]

    pollinations_custom_desc = st.text_input("Extra prompt for Pollinations AI images (optional):", value="documentary style, professional, vibrant colors")
    pollinations_n_images = st.slider("Number of Pollinations AI options per scene", 1, 4, 2)

    ken_burns_on = st.checkbox("Apply Ken Burns effect (pan/zoom) for images", value=True)
    if ken_burns_on:
        ken_burns_zoom = st.slider("Ken Burns: Zoom factor", min_value=1.01, max_value=1.2, value=1.08, step=0.01)
        ken_burns_random_pan = st.checkbox("Random pan direction", value=True)

    if "editable_script" not in st.session_state:
        st.session_state["editable_script"] = ""
    if "media_list" not in st.session_state:
        st.session_state["media_list"] = []
    if "last_num_media" not in st.session_state:
        st.session_state["last_num_media"] = 0
    if "pollinations_selected" not in st.session_state:
        st.session_state["pollinations_selected"] = {}

    if st.button("Generate!"):
        progress_bar = st.progress(0, text="Starting ...")
        if script_mode == "Write script manually" and not script_text.strip():
            st.error("Please enter the script text.")
        elif script_mode != "Write script manually" and not topic.strip():
            st.error("Please enter a topic.")
        elif not COHERE_API_KEY:
            st.error("Cohere API key not found!")
        else:
            with st.spinner("Generating ..."):
                progress_bar.progress(5, text="Generating script ...")
                media_list = []
                sentences = []
                if script_mode == "Script from media (Cohere)":
                    # -- unchanged --
                    st.warning("Script from media (Cohere) mode is not yet showing Pollinations selection, use AI-generated script for preview/replace.")
                elif script_mode == "AI-generated script (Cohere)":
                    cohere_prompt = f"""Write a smooth, well-connected, short documentary script about "{topic}" in {num_media} sentences. Each sentence continues the previous, as if the viewer is following a story."""
                    script_text_out = generate_script_with_cohere(cohere_prompt, max_tokens=cohere_tokens, temperature=cohere_temp)
                    final_text = script_text_out.strip()
                    sentences = filter_script_sentences(final_text, num_media)
                else:
                    final_text = script_text.strip()
                    sentences = filter_script_sentences(final_text, num_media)

                pollinations_picks = {}
                montage_choices = []
                for idx, sent in enumerate(sentences):
                    pollinations_prompt = f"{sent.strip()}, {pollinations_custom_desc}".strip(", ")
                    pollimgs = []
                    if "Pollinations" in sources_selected:
                        pollimgs = search_pollinations_photos_with_desc(pollinations_prompt, per_page=pollinations_n_images)
                    pick_idx = 0
                    # عرض صور Pollinations للاختيار
                    if pollimgs:
                        st.write(f"Scene {idx+1}: Choose one Pollinations AI image for this script:")
                        cols = st.columns(len(pollimgs))
                        for i, (media_type, media_url, desc) in enumerate(pollimgs):
                            with cols[i]:
                                st.image(media_url, caption=f"Option {i+1}", width=200)
                                if st.button(f"Select Option {i+1} for Scene {idx+1}"):
                                    pollinations_picks[idx] = i
                        # حفظ الاختيار
                        pick_idx = pollinations_picks.get(idx, 0)
                    ken_burns_params = (ken_burns_zoom if ken_burns_on else 1.0,
                                        "random" if ken_burns_on and ken_burns_random_pan else "left_to_right")
                    # المصادر الأخرى
                    found = False
                    media_type = "image"
                    media_url = None
                    for src in sources_selected:
                        if src == "Pollinations" and pollimgs:
                            media_type, media_url, desc = pollimgs[pick_idx]
                            found = True
                            break
                        if src == "Pexels":
                            res = search_pexels_photos_with_desc(sent, per_page=1)
                            if res:
                                media_type, media_url, desc = res[0]
                                found = True
                                break
                        if src == "Pixabay":
                            res = search_pixabay_photos_with_desc(sent, per_page=1)
                            if res:
                                media_type, media_url, desc = res[0]
                                found = True
                                break
                        if src == "Unsplash":
                            res = search_unsplash_photos_with_desc(sent, per_page=1)
                            if res:
                                media_type, media_url, desc = res[0]
                                found = True
                                break
                        if src == "Wikimedia":
                            res = search_wikimedia_photos_with_desc(sent, limit=1)
                            if res:
                                media_type, media_url, desc = res[0]
                                found = True
                                break
                        # مستقبلًا: مصادر الفيديو AI (مثال Kaiber/Pika إذا توفر API)
                    if not found and pollimgs:
                        media_type, media_url, desc = pollimgs[pick_idx]
                    montage_choices.append((media_type, media_url, sent, ken_burns_params))
                # حفظ تلقائي للمشروع
                save_project({
                    "topic": topic,
                    "script": final_text,
                    "sentences": sentences,
                    "montage_choices": montage_choices,
                    "settings": {
                        "color": color, "text_size": text_size, "text_pos": text_pos,
                        "youtube_export": youtube_export, "watermark": watermark,
                        "sources_selected": sources_selected, "ken_burns_on": ken_burns_on,
                        "ken_burns_zoom": ken_burns_zoom if ken_burns_on else 1.0,
                        "ken_burns_random_pan": ken_burns_random_pan if ken_burns_on else False,
                        "voice_choice": voice_choice,
                        "pollinations_custom_desc": pollinations_custom_desc,
                        "pollinations_n_images": pollinations_n_images
                    }
                }, path=auto_project_file)
                st.success("AI images selected. Now you can build your video!")
                st.session_state["editable_script"] = final_text
                st.session_state["montage_choices"] = montage_choices
                st.session_state["last_num_media"] = num_media

    if st.session_state.get("editable_script", "") and st.session_state.get("montage_choices"):
        st.markdown("### ✏️ Edit the script, then click Build Video:")
        script_edit = st.text_area("Script (edit before building video):",
                                   value=st.session_state["editable_script"], height=250, key="script_editbox")
        if st.button("Build video / Rebuild after edit"):
            temp_files = []
            sentences = filter_script_sentences(script_edit, st.session_state["last_num_media"])
            logo_path = None
            if logo_file:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_logo:
                    image = Image.open(logo_file)
                    image.save(tmp_logo.name)
                    logo_path = tmp_logo.name
                    temp_files.append(logo_path)
            music_path = None
            if music_file:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_music:
                    music_file.seek(0)
                    tmp_music.write(music_file.read())
                    music_path = tmp_music.name
                    temp_files.append(music_path)
            montage = []
            not_found_report = []
            for idx, (media_type, media_url, sent, ken_burns_params) in enumerate(st.session_state["montage_choices"]):
                # إعادة توليد الصوت للنص الجديد إذا تغير
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
                montage, out_path=out_video_path, color=color, text_size=text_size, text_pos=text_pos,
                logo_path=logo_path, music_path=music_path, watermark_text=watermark,
                gif_export=gif_export, square_export=square_export, youtube_export=youtube_export,
                text_anim_mode=text_anim_mode_val, text_anim_group_size=text_anim_group_size, text_anim_lang=text_anim_lang
            )
            st.success("Done! See your result 👇")
            st.video(final_video)
            st.info(f"Video duration: {video_duration_sec/60:.2f} min ({video_duration_sec:.1f} sec)")
            if not_found_report:
                st.warning("Failed to find media for some scenes:")
                st.markdown("\n".join(not_found_report))
            with open(final_video, "rb") as f:
                st.download_button(label="Download Video", data=f, file_name="documentary_video.mp4", mime="video/mp4")
            for f in temp_files:
                try:
                    os.remove(f)
                except Exception:
                    pass

# Feature roadmap:
# - Support for more AI video sources (e.g. Kaiber, Pika) when API is available.
# - Allow user to edit/replace media for each scene after initial selection.
# - Smart defaults for scene image description prompt (future: using LLM for richer prompt).
# - Full auto-save/load project experience.
