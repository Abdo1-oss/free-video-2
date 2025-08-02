import streamlit as st
import requests
import tempfile
import os
import json
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
import shutil

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

PEXELS_API_KEY = "pLcIoo3oNdhqna28AfdaBYhkE3SFps9oRGuOsxY3JTe92GcVDZpwZE9i"
UNSPLASH_ACCESS_KEY = "SDK5avSHNm9lcNhhLhT_SzUdzd98hYX0BVjswi3ZHzU"
PIXABAY_API_KEY = "50380897-76243eaec536038f687ff8e15"
COHERE_API_KEY = "K1GW0y2wWiwW7xlK7db7zZnqX7sxfRVGiWopVfCD"

# ضع مفتاح DeepAI الخاص بك هنا (سجّل مجاناً في https://deepai.org/)
DEEPAI_API_KEY = "790f1607-c5ea-4f10-b116-59ceadd77c25"

GTTS_VOICES = [
    {"name": "English (US) - Female", "lang": "en", "tld": "com"},
    {"name": "English (UK) - Female", "lang": "en", "tld": "co.uk"},
    {"name": "French (France) - Female", "lang": "fr", "tld": "fr"},
    {"name": "Spanish (Spain) - Female", "lang": "es", "tld": "es"},
    {"name": "German (Germany) - Female", "lang": "de", "tld": "de"},
]

DEFAULT_PLACEHOLDER_IMG = "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"

def generate_thumbnail_image(prompt, temp_files):
    """
    Uses DeepAI's text-to-image API to generate a thumbnail image based on the prompt.
    Requires a free API key from https://deepai.org/
    """
    if not DEEPAI_API_KEY or DEEPAI_API_KEY == "YOUR_DEEPAI_API_KEY":
        print("DeepAI API key missing, cannot generate thumbnail.")
        return None
    url = "https://api.deepai.org/api/text2img"
    try:
        response = requests.post(
            url,
            data={'text': prompt},
            headers={'api-key': DEEPAI_API_KEY}
        )
        data = response.json()
        img_url = data.get("output_url")
        if img_url:
            img_data = requests.get(img_url).content
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
                tmp_img.write(img_data)
                temp_files.append(tmp_img.name)
                return tmp_img.name
        else:
            print("DeepAI response missing output_url:", data)
    except Exception as e:
        print(f"DeepAI thumbnail error: {e}")
    return None

def safe_download_and_convert_image(media_url, temp_files):
    try:
        img_data = requests.get(media_url, timeout=10).content
        img_bytes = io.BytesIO(img_data)
        with Image.open(img_bytes) as pil_img:
            pil_img = pil_img.convert("RGB")
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
                pil_img.save(tmp_img.name)
                img_path = tmp_img.name
                temp_files.append(img_path)
        return img_path
    except Exception as e:
        print(f"Failed to process image: {media_url}, error: {e}")
        return None

def get_media_alternative(query, exclude_sources=[], per_page=1):
    sources = ["Pexels", "Pixabay", "Unsplash", "Wikimedia"]
    for src in sources:
        if src in exclude_sources:
            continue
        if src == "Pexels":
            res = search_pexels_photos_with_desc(query, per_page=1)
            if res: return res[0]
        if src == "Pixabay":
            res = search_pixabay_photos_with_desc(query, per_page=1)
            if res: return res[0]
        if src == "Unsplash":
            res = search_unsplash_photos_with_desc(query, per_page=1)
            if res: return res[0]
        if src == "Wikimedia":
            res = search_wikimedia_photos_with_desc(query, limit=1)
            if res: return res[0]
    return ("image", DEFAULT_PLACEHOLDER_IMG, "No image found")

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
    if lang != "en":
        prompt += f"\nWrite the script in {lang}."
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

def resize_and_letterbox(img_clip, target_w=1280, target_h=720):
    img_clip = img_clip.resize(height=target_h)
    if img_clip.w > target_w:
        img_clip = img_clip.crop(x_center=img_clip.w/2, width=target_w)
    elif img_clip.w < target_w:
        img_clip = img_clip.margin(left=(target_w-img_clip.w)//2, right=(target_w-img_clip.w)//2, color=(0,0,0))
    return img_clip

def resize_to_vertical(img_clip, target_w=720, target_h=1280):
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
    vertical_export=False,
    text_anim_mode="sentence", text_anim_group_size=1, text_anim_lang="en"
):
    clips = []
    audio_clips = []
    temp_files = []
    thumbnail_path = None
    try:
        # توليد الصورة المصغرة عبر DeepAI أولاً (حسب الطلب)
        if not thumbnail_path:
            topic = st.session_state.get('topic', 'my topic')
            thumbnail_prompt = f"cinematic, colorful thumbnail for a documentary about {topic}, trending YouTube style"
            thumbnail_path = generate_thumbnail_image(thumbnail_prompt, temp_files)
        # إذا فشل DeepAI، fallback لأول صورة/فريم
        for idx, (media_type, media_url, audio_path, sent) in enumerate(montage):
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
                    if vertical_export:
                        clip = resize_to_vertical(clip, target_w=720, target_h=1280)
                    else:
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
                    # إذا لم يتمكن DeepAI من توليد thumbnail، خذ الفريم الأول
                    if idx == 0 and not thumbnail_path:
                        thumb_img = clip.get_frame(0)
                        thumb_pil = Image.fromarray(thumb_img)
                        thumb_temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                        thumb_pil.save(thumb_temp.name)
                        thumbnail_path = thumb_temp.name
                        temp_files.append(thumbnail_path)
                except Exception as e:
                    print(f"Video error: {e}, fallback to image")
                    media_type = "image"
            if media_type == "image":
                img_path = media_url
                if isinstance(img_path, str) and img_path.startswith("http"):
                    img_path = safe_download_and_convert_image(img_path, temp_files)
                    if img_path is None:
                        continue  # skip broken image
                if vertical_export:
                    img_clip = ImageClip(img_path)
                    img_clip = resize_to_vertical(img_clip, target_w=720, target_h=1280)
                else:
                    img_clip = ImageClip(img_path)
                    img_clip = resize_and_letterbox(img_clip, target_w=1280, target_h=720)
                img_clip = img_clip.set_duration(duration)
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
                if idx == 0 and not thumbnail_path:
                    thumb_pil = Image.open(img_path)
                    thumb_temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    thumb_pil.save(thumb_temp.name)
                    thumbnail_path = thumb_temp.name
                    temp_files.append(thumbnail_path)
        if not clips or not audio_clips:
            st.error("Could not build the final video.")
            return None, None, None
        final_audio = concatenate_audioclips(audio_clips)
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip = final_clip.set_audio(final_audio)
        final_clip = final_clip.subclip(0, final_audio.duration)
        if youtube_export:
            final_clip = final_clip.resize(height=720)
            if final_clip.w != 1280:
                final_clip = final_clip.crop(x_center=final_clip.w/2, width=1280, height=720)
        if vertical_export:
            final_clip = final_clip.resize(height=1280)
            if final_clip.w != 720:
                final_clip = final_clip.crop(x_center=final_clip.w/2, width=720, height=1280)
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
        final_clip = final_clip.fadein(1).fadeout(1)
        final_clip.write_videofile(out_path, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, fps=15)
        for c in clips:
            try: c.close()
            except: pass
        for a in audio_clips:
            try: a.close()
            except: pass
        final_clip.close()
        return out_path, final_audio.duration, thumbnail_path
    finally:
        for f in temp_files:
            try:
                os.remove(f)
            except Exception:
                pass

# باقي الكود (واجهة ستريملت، توزيع المشاهد، بناء الفيديو، عرض الصورة المصغرة ...) كما هو في سكريبتك الأصلي،
# فقط عند عرض الصورة المصغرة استخدم الـ thumbnail_path الناتج من assemble_video:
#    if thumbnail_path and os.path.exists(thumbnail_path):
#        st.image(thumbnail_path, caption="Thumbnail (AI Generated)")
#        with open(thumbnail_path, "rb") as fthumb:
#            st.download_button("Download Thumbnail", data=fthumb, file_name="thumbnail.jpg", mime="image/jpeg")
