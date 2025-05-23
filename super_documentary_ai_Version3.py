import streamlit as st
import requests
import tempfile
import os
import json
import numpy as np
from moviepy.editor import concatenate_videoclips, ImageClip, CompositeVideoClip, AudioFileClip, TextClip, concatenate_audioclips, VideoFileClip
from PIL import Image
from gtts import gTTS
import nltk
import re
import random
import wave
import struct

import moviepy.config as mpy_conf
mpy_conf.change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick\\magick.exe"})

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

PEXELS_API_KEY = "pLcIoo3oNdhqna28AfdaBYhkE3SFps9oRGuOsxY3JTe92GcVDZpwZE9i"
UNSPLASH_ACCESS_KEY = "SDK5avSHNm9lcNhhLhT_SzUdzd98hYX0BVjswi3ZHzU"
PIXABAY_API_KEY = "50380897-76243eaec536038f687ff8e15"
COHERE_API_KEY = "K1GW0y2wWiwW7xlK7db7zZnqX7sxfRVGiWopVfCD"

GTTS_VOICES = [
    {"name": "عربي (سعودي) - أنثى", "lang": "ar", "tld": "com.sa"},
    {"name": "عربي (مصر) - أنثى", "lang": "ar", "tld": "com.eg"},
    {"name": "English (US) - Female", "lang": "en", "tld": "com"},
    {"name": "English (UK) - Female", "lang": "en", "tld": "co.uk"},
    {"name": "French (France) - Female", "lang": "fr", "tld": "fr"},
    {"name": "Spanish (Spain) - Female", "lang": "es", "tld": "es"},
    {"name": "German (Germany) - Female", "lang": "de", "tld": "de"},
]

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
        st.error(f"خطأ من Cohere API: {response.status_code}\n{response.text}")
        return ""

def generate_script_from_media_cohere(media_list, topic, lang="ar", max_tokens=1000, temperature=0.4):
    prompt = f"""لدي مجموعة صور وفيديوهات حول موضوع "{topic}":\n"""
    for i, (_, url, desc) in enumerate(media_list, 1):
        prompt += f"{i}. {desc.strip()}\n"
    prompt += f"""
اكتب نصًا وثائقيًا قصيرًا مترابطًا ومتسلسلًا يغطي جميع هذه الصور والفيديوهات بالترتيب، بحيث تكمّل كل جملة الجملة التي قبلها، ويبدو النص كقصة أو شرح واحد متسق، دون ذكر كلمة "صورة" أو "مشهد" أو أرقام، ودون تكرار أو انقطاع.
"""
    return generate_script_with_cohere(prompt, max_tokens=max_tokens, temperature=temperature)

def filter_script_sentences(raw_text, num_media):
    sentences = re.split(r'[.!؟\n]', raw_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    ignore = [
        "الصورة", "المشهد", "في هذا المشهد", "في هذه الصورة", "رقم", "media", "picture", "image", "scene", "مشهد:", "صورة:"
    ]
    filtered = []
    for s in sentences:
        if not any(kw in s.lower() for kw in ignore) and len(s) > 5:
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

def animated_text_clip(img_clip, text, duration, lang="ar", mode="word", group_size=3, font_size=40, color="white", text_pos="bottom"):
    words = text.split()
    if mode == "group":
        items = [' '.join(words[i:i+group_size]) for i in range(0, len(words), group_size)]
    else:
        items = words
    item_dur = duration / max(len(items), 1)
    txt_clips = []
    for i, item in enumerate(items):
        font = "Arial"
        if lang == "ar":
            font = "Cairo"
        txt = TextClip(
            item, fontsize=font_size, color=color, font=font,
            size=img_clip.size, method='caption', align='center'
        )
        txt = txt.set_duration(item_dur).set_start(i * item_dur).set_position(("center", text_pos))
        txt_clips.append(txt)
    return CompositeVideoClip([img_clip] + txt_clips).set_duration(duration)

def resize_and_letterbox(img_clip, target_w=1280, target_h=720):
    img_clip = img_clip.resize(height=target_h)
    if img_clip.w > target_w:
        img_clip = img_clip.crop(x_center=img_clip.w/2, width=target_w)
    elif img_clip.w < target_w:
        img_clip = img_clip.margin(left=(target_w-img_clip.w)//2, right=(target_w-img_clip.w)//2, color=(0,0,0))
    return img_clip

def random_watermark_positions(duration, w, h, txt_w=200, txt_h=30, step=2):
    positions = []
    for t in range(0, int(duration), step):
        x = random.randint(0, max(0, w-txt_w))
        y = random.randint(0, max(0, h-txt_h))
        positions.append((t, (x, y)))
    return positions

def choose_music_for_topic(topic):
    topic = topic.lower()
    if "طبيعة" in topic or "nature" in topic:
        return "music/nature.mp3"
    elif "سيارة" in topic or "car" in topic:
        return "music/cars.mp3"
    elif "فضاء" in topic or "space" in topic:
        return "music/space.mp3"
    elif "تاريخ" in topic or "history" in topic:
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
    montage, out_path="final_montage.mp4", color="#FFFFFF", text_size=32, text_pos="bottom",
    logo_path=None, music_path=None, watermark_text="", gif_export=False, square_export=False, youtube_export=False,
    text_anim_mode="word", text_anim_group_size=3, text_anim_lang="ar"
):
    clips = []
    audio_clips = []
    for media_type, media_url, audio_path, sent in montage:
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
                    mode=text_anim_mode,
                    group_size=text_anim_group_size,
                    font_size=text_size,
                    color=color,
                    text_pos=text_pos
                )
                clips.append(anim_txt)
            except Exception as e:
                print(f"Video error: {e}, fallback to image")
                media_type = "image"
        if media_type == "image":
            img_path = media_url
            if isinstance(img_path, str) and img_path.startswith("http"):
                img_path_local = os.path.join(st.session_state['save_dir'], f"img_{random.randint(1000,9999)}.jpg")
                try:
                    img_data = requests.get(img_path, timeout=10).content
                    with open(img_path_local, "wb") as f:
                        f.write(img_data)
                    img_path = img_path_local
                except Exception as e:
                    print(f"Image download error: {e}")
                    continue
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
    if not clips or not audio_clips:
        st.error("لم يتم بناء الفيديو النهائي بسبب مشاكل في التحويل أو الدمج.")
        return None, None
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
                watermark_text, fontsize=24, color='white', font='Arial-Bold', bg_color='black',
                size=(200, 30)
            ).set_duration(final_clip.duration).set_opacity(0.4)
            positions = random_watermark_positions(final_clip.duration, final_clip.w, final_clip.h, 200, 30, step=2)
            def moving_position(t):
                idx = int(t // 2)
                return positions[idx][1] if idx < len(positions) else positions[-1][1]
            txt_clip = txt_clip.set_position(moving_position)
            final_clip = CompositeVideoClip([final_clip, txt_clip])
        except Exception as e:
            print(f"تعذر إنشاء العلامة المائية المتحركة: {e}")
            st.warning("تعذر إضافة العلامة المائية بسبب مشكلة في ImageMagick أو MoviePy.")
    final_clip = final_clip.fadein(1).fadeout(1)
    final_clip.write_videofile(out_path, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, fps=15)
    for c in clips:
        c.close()
    for a in audio_clips:
        a.close()
    final_clip.close()
    return out_path, final_audio.duration

st.set_page_config(page_title="وثائقي Cohere (صور وفيديو) | متطور", layout="wide")
st.title("🎬 وثائقي Cohere (صور وفيديو) | متطور مع نص متحرك وWatermark متحركة وموسيقى تلقائية")

save_dir = st.text_input("مكان حفظ الملفات (مجلد):", value=r"C:\Users\Computec\Desktop\OUTPUTS")
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
st.session_state['save_dir'] = save_dir

mode = st.radio("نوع المشروع", ["ابدأ مشروع جديد", "استرجع مشروع من ملف"])

if mode == "استرجع مشروع من ملف":
    uploaded_project = st.file_uploader("ارفع ملف المشروع (json):", type="json")
    if uploaded_project:
        project_data = json.load(uploaded_project)
        st.success("تم استرجاع المشروع!")
        st.json(project_data)
else:
    st.markdown("**اكتب موضوع الفيديو أو السكربت، وحدد عدد المشاهد، واختر مصادر الصور والفيديو، ودع الذكاء الاصطناعي يصنع لك فيلم وثائقي تلقائياً!**")
    topic = st.text_input("موضوع الفيديو (مثال: السيارات الذكية)")
    st.session_state["topic"] = topic
    num_media = st.slider("عدد الصور/المشاهد:", min_value=2, max_value=100, value=5)
    lang_option = st.selectbox("لغة السكربت:", ["ar", "en", "fr", "es", "de", "ru"], index=0)
    script_mode = st.radio(
        "طريقة الحصول على السكربت:",
        ["إنشاء السكربت تلقائيًا (Cohere)", "إنشاء السكربت بناءً على الوسائط (Cohere)", "أكتب السكربت بنفسي"], index=0)
    script_text = ""
    cohere_tokens = st.slider("عدد الكلمات التقريبي للسكريبت:", 100, 4000, 1000, step=50)
    cohere_temp = st.slider("درجة الإبداع:", 0.1, 1.0, 0.4, step=0.05)
    if script_mode == "أكتب السكربت بنفسي":
        script_text = st.text_area("اكتب السكربت هنا:", height=300)
    sources_selected = st.multiselect(
        "مصادر الصور والفيديو:",
        options=["Pexels", "Unsplash", "Pixabay", "Wikimedia"],
        default=["Pexels", "Unsplash", "Pixabay", "Wikimedia"]
    )
    logo_file = st.file_uploader("شعار الفيديو (اختياري):", type=["png", "jpg", "jpeg"])
    music_file = st.file_uploader("موسيقى مجانية (اختياري):", type=["mp3", "wav"])
    youtube_export = st.checkbox("تصدير نسخة يوتيوب (16:9)", value=True)
    watermark = st.text_input("نص العلامة المائية (اختياري):", value="@SuperAI")
    color = st.color_picker("اختر لون النص", "#ffffff")
    text_size = st.slider("حجم النص", 14, 60, 28)
    text_pos = st.radio("مكان النص", options=["top", "center", "bottom"], index=2)
    gif_export = st.checkbox("تصدير كـ GIF", value=False)
    square_export = st.checkbox("تصدير نسخة مربعة للفيديو (انستجرام)", value=False)
    voice_choice = st.selectbox("اختر صوت التعليق الصوتي:", [v["name"] for v in GTTS_VOICES])
    voice_data = next(v for v in GTTS_VOICES if v["name"] == voice_choice)
    text_anim_mode = st.radio("طريقة ظهور الكلمات:", ["كلمة كلمة", "كل ثلاث كلمات"], index=0)
    text_anim_mode_val = "word" if text_anim_mode == "كلمة كلمة" else "group"
    text_anim_group_size = 1 if text_anim_mode == "كلمة كلمة" else 3
    text_anim_lang = st.selectbox("لغة النص المتحرك:", ["ar", "en", "fr", "es", "de"], index=0)

    # متغيرات لحفظ السكربت والوسائط في الجلسة
    if "editable_script" not in st.session_state:
        st.session_state["editable_script"] = ""
    if "media_list" not in st.session_state:
        st.session_state["media_list"] = []
    if "last_sentences" not in st.session_state:
        st.session_state["last_sentences"] = []
    if "last_script_mode" not in st.session_state:
        st.session_state["last_script_mode"] = ""
    if "last_num_media" not in st.session_state:
        st.session_state["last_num_media"] = 0

    if st.button("ابدأ الإبداع!"):
        progress_bar = st.progress(0, text="جاري بدء العملية ...")
        completed = 0

        if script_mode == "أكتب السكربت بنفسي" and not script_text.strip():
            st.error("يرجى كتابة نص السكربت بالكامل.")
        elif script_mode != "أكتب السكربت بنفسي" and not topic.strip():
            st.error("يرجى كتابة موضوع الفيديو.")
        elif not COHERE_API_KEY:
            st.error("مفتاح Cohere API غير موجود. يرجى إضافته في الأعلى.")
        else:
            with st.spinner("جاري الإبداع ..."):
                progress_bar.progress(5, text="جاري توليد السكربت ...")
                media_list = []

                if script_mode == "إنشاء السكربت بناءً على الوسائط (Cohere)":
                    all_media = []
                    n_each = max(1, num_media // (2 * len(sources_selected)))
                    for src in sources_selected:
                        if src == "Pexels":
                            all_media += search_pexels_photos_with_desc(topic, per_page=n_each)
                            all_media += search_pexels_videos_with_desc(topic, per_page=n_each)
                        if src == "Unsplash":
                            all_media += search_unsplash_photos_with_desc(topic, per_page=n_each*2)
                        if src == "Pixabay":
                            all_media += search_pixabay_photos_with_desc(topic, per_page=n_each)
                            all_media += search_pixabay_videos_with_desc(topic, per_page=n_each)
                        if src == "Wikimedia":
                            all_media += search_wikimedia_photos_with_desc(topic, limit=n_each*2)
                    photos = [m for m in all_media if m[0] == "image"]
                    videos = [m for m in all_media if m[0] == "video"]
                    media_list = []
                    i = j = 0
                    for k in range(num_media):
                        if k % 2 == 0 and i < len(photos):
                            media_list.append(photos[i])
                            i += 1
                        elif j < len(videos):
                            media_list.append(videos[j])
                            j += 1
                        elif i < len(photos):
                            media_list.append(photos[i])
                            i += 1
                        elif j < len(videos):
                            media_list.append(videos[j])
                            j += 1
                    media_list = media_list[:num_media]
                    if not media_list:
                        st.error("لم يتم العثور على وسائط كافية (صور/فيديو). جرب تقليل العدد أو فعّل مصادر أكثر.")
                        st.stop()
                    for i, (media_type, url, desc) in enumerate(media_list):
                        if media_type == "image":
                            st.image(url, caption=f"{i+1}. {desc}")
                        elif media_type == "video":
                            st.video(url, format="video/mp4", start_time=0)
                    script_text_out = generate_script_from_media_cohere(
                        media_list, topic, lang=lang_option, max_tokens=cohere_tokens, temperature=cohere_temp
                    )
                    final_text = script_text_out.strip()
                elif script_mode == "إنشاء السكربت تلقائيًا (Cohere)":
                    cohere_prompt = f"""اكتب نصًا وثائقيًا متسلسلًا ومترابطًا عن "{topic}" مكوّن من {num_media} جمل، بحيث تكمّل كل جملة ما قبلها، وكأن المشاهد يتابع قصة أو شرح متدرج."""
                    script_text_out = generate_script_with_cohere(cohere_prompt, max_tokens=cohere_tokens, temperature=cohere_temp)
                    final_text = script_text_out.strip()
                    # media_list سيُبنى لاحقاً بعد التعديل
                else:
                    final_text = script_text.strip()
                    # media_list سيُبنى لاحقاً بعد التعديل

                st.session_state["editable_script"] = final_text
                st.session_state["media_list"] = media_list
                st.session_state["last_sentences"] = filter_script_sentences(final_text, num_media)
                st.session_state["last_script_mode"] = script_mode
                st.session_state["last_num_media"] = num_media

    # عرض السكربت القابل للتعديل
    if st.session_state.get("editable_script", ""):
        st.markdown("### ✏️ يمكنك تعديل السكربت ثم الضغط على زر بناء الفيديو:")
        script_edit = st.text_area("السكربت (يمكنك التعديل عليه قبل إنتاج الفيديو):",
                                   value=st.session_state["editable_script"], height=250, key="script_editbox")
        if st.button("بناء الفيديو/إعادة البناء بعد التعديل"):
            # بناء الفيديو بناءً على النص المعدل
            sentences = filter_script_sentences(script_edit, st.session_state["last_num_media"])
            st.session_state["editable_script"] = script_edit  # احفظ آخر نسخة
            logo_path = None
            if logo_file:
                logo_path = os.path.join(save_dir, "logo.png")
                image = Image.open(logo_file)
                image.save(logo_path)
            music_path = None
            if music_file:
                music_path = os.path.join(save_dir, "music.mp3")
                music_file.seek(0)
                with open(music_path, "wb") as f:
                    f.write(music_file.read())
            montage = []
            not_found_report = []
            media_list = st.session_state["media_list"]
            script_mode = st.session_state["last_script_mode"]
            if script_mode == "إنشاء السكربت بناءً على الوسائط (Cohere)":
                pair_count = min(len(sentences), len(media_list))
                for idx in range(pair_count):
                    sent = sentences[idx]
                    media_type, media_url, media_desc = media_list[idx]
                    if media_type == "image":
                        img_path = os.path.join(save_dir, f"img_{idx}.jpg")
                        try:
                            img_data = requests.get(media_url, timeout=10).content
                            with open(img_path, "wb") as f:
                                f.write(img_data)
                            media_url_local = img_path
                        except Exception as e:
                            not_found_report.append(f"فشل تحميل الصورة: {media_url}")
                            continue
                    else:
                        media_url_local = media_url
                    mp3_path = os.path.join(save_dir, f"audio_{idx}.mp3")
                    safe_tts_save(sent, mp3_path, voice_data["lang"], voice_data["tld"])
                    montage.append((media_type, media_url_local, mp3_path, sent))
            else:
                # توليد تلقائي للصور حسب كل جملة
                for idx in range(min(len(sentences), st.session_state["last_num_media"])):
                    sent = sentences[idx]
                    found = False
                    media_type = "image"
                    media_url = None
                    for src in sources_selected:
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
                    if found and media_type == "image":
                        img_path = os.path.join(save_dir, f"img_{idx}.jpg")
                        try:
                            img_data = requests.get(media_url, timeout=10).content
                            with open(img_path, "wb") as f:
                                f.write(img_data)
                            media_url = img_path
                        except Exception as e:
                            not_found_report.append(f"فشل تحميل الصورة: {media_url}")
                            continue
                    mp3_path = os.path.join(save_dir, f"audio_{idx}.mp3")
                    safe_tts_save(sent, mp3_path, voice_data["lang"], voice_data["tld"])
                    montage.append((media_type, media_url, mp3_path, sent))
            if not montage:
                st.error("لم يتم إنشاء أي مشهد صالح للفيديو.")
                st.stop()

            out_video_path = os.path.join(save_dir, "documentary_video.mp4")
            final_video, video_duration_sec = assemble_video(
                montage, out_path=out_video_path, color=color, text_size=text_size, text_pos=text_pos,
                logo_path=logo_path, music_path=music_path, watermark_text=watermark,
                gif_export=gif_export, square_export=square_export, youtube_export=youtube_export,
                text_anim_mode=text_anim_mode_val, text_anim_group_size=text_anim_group_size, text_anim_lang=text_anim_lang
            )
            st.success("تم الإنشاء! شاهد نتيجتك 👇")
            st.video(final_video)
            st.info(f"مدة الفيديو النهائي: {video_duration_sec/60:.2f} دقيقة ({video_duration_sec:.1f} ثانية)")
            if not_found_report:
                st.warning("المشاهد التي لم يتم العثور لها على صور أو حدث بها خطأ:")
                st.markdown("\n".join(not_found_report))
            with open(final_video, "rb") as f:
                st.download_button(label="تحميل الفيديو", data=f, file_name="documentary_video.mp4",
                                   mime="video/mp4")
