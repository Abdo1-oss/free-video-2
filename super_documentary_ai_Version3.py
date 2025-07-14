import streamlit as st
import requests
import tempfile
import random
import os
from pydub import AudioSegment, silence
from moviepy.editor import (
    VideoFileClip, AudioFileClip, concatenate_videoclips, vfx, CompositeVideoClip, ImageClip
)
import cv2
import cohere
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display

# ------------ إعدادات API KEYS ------------
COHERE_API_KEY = "K1GW0y2wWiwW7xlK7db7zZnqX7sxfRVGiWopVfCD"
PEXELS_API_KEY = "pLcIoo3oNdhqna28AfdaBYhkE3SFps9oRGuOsxY3JTe92GcVDZpwZE9i"
PIXABAY_API_KEY = "50380897-76243eaec536038f687ff8e15"

QURAA = [
    {"name": "الحصري مرتل", "id": "Husary_64kbps"},
    {"name": "المنشاوي مرتل", "id": "Minshawy_Murattal_128kbps"},
    {"name": "عبد الباسط مرتل", "id": "Abdul_Basit_Murattal_64kbps"},
    {"name": "الغامدي", "id": "Ghamadi_40kbps"},
    {"name": "المعيقلي", "id": "MaherAlMuaiqly_64kbps"},
    {"name": "العفاسي", "id": "Alafasy_64kbps"},
    {"name": "الشريم", "id": "Shuraym_64kbps"},
    {"name": "فارس عباد", "id": "Fares_Abbad_64kbps"},
    {"name": "الشاطري", "id": "Shatri_64kbps"},
    {"name": "أبو بكر الشاطري", "id": "Abu_Bakr_Ash-Shaatree_128kbps"},
    {"name": "محمد أيوب", "id": "Muhammad_Ayyoub_128kbps"},
    {"name": "ياسر الدوسري", "id": "Yasser_Ad-Dussary_128kbps"},
    {"name": "أحمد العجمي", "id": "Ajamy_64kbps"},
    {"name": "إدريس أبكر", "id": "Idrees_Abkar_48kbps"},
    {"name": "خليفة الطنيجي", "id": "Tunaiji_64kbps"},
    {"name": "عبد الله بصفر", "id": "Basfar_48kbps"},
    {"name": "الحصري مجود", "id": "Husary_Mujawwad_128kbps"},
    {"name": "محمد جبريل", "id": "Jibreel_64kbps"},
    {"name": "علي جابر", "id": "Ali_Jaber_64kbps"},
    {"name": "المنشاوي مجود", "id": "Minshawy_Mujawwad_128kbps"}
]

SURA_NAMES = [
    "الفاتحة", "البقرة", "آل عمران", "النساء", "المائدة", "الأنعام", "الأعراف", "الأنفال", "التوبة", "يونس", "هود", "يوسف", "الرعد",
    "إبراهيم", "الحجر", "النحل", "الإسراء", "الكهف", "مريم", "طه", "الأنبياء", "الحج", "المؤمنون", "النور", "الفرقان", "الشعراء",
    "النمل", "القصص", "العنكبوت", "الروم", "لقمان", "السجدة", "الأحزاب", "سبأ", "فاطر", "يس", "الصافات", "ص", "الزمر", "غافر",
    "فصلت", "الشورى", "الزخرف", "الدخان", "الجاثية", "الأحقاف", "محمد", "الفتح", "الحجرات", "ق", "الذاريات", "الطور", "النجم",
    "القمر", "الرحمن", "الواقعة", "الحديد", "المجادلة", "الحشر", "الممتحنة", "الصف", "الجمعة", "المنافقون", "التغابن", "الطلاق",
    "التحريم", "الملك", "القلم", "الحاقة", "المعارج", "نوح", "الجن", "المزمل", "المدثر", "القيامة", "الإنسان", "المرسلات", "النبأ",
    "النازعات", "عبس", "التكوير", "الانفطار", "المطففين", "الانشقاق", "البروج", "الطارق", "الأعلى", "الغاشية", "الفجر", "البلد",
    "الشمس", "الليل", "الضحى", "الشرح", "التين", "العلق", "القدر", "البينة", "الزلزلة", "العاديات", "القارعة", "التكاثر", "العصر",
    "الهمزة", "الفيل", "قريش", "الماعون", "الكوثر", "الكافرون", "النصر", "المسد", "الإخلاص", "الفلق", "الناس"
]

SURA_AYAHS = [
    7, 286, 200, 176, 120, 165, 206, 75, 129, 109, 123, 111, 43, 52, 99, 128, 111, 110, 98, 135, 112, 78, 118, 64, 77, 227, 93, 88, 69, 60, 34, 30, 73, 54, 45, 83, 182, 88, 75, 85, 54, 53, 89, 59, 37, 35, 38, 29, 18, 45, 60, 49, 62, 55, 78, 96, 29, 22, 24, 13, 14, 11, 11, 18, 12, 12, 30, 52, 52, 44, 28, 28, 20, 56, 40, 31, 50, 40, 46, 42, 29, 19, 36, 25, 22, 17, 19, 26, 30, 20, 15, 21, 11, 8, 8, 19, 5, 8, 8, 11, 11, 8, 3, 9, 5, 4, 7, 3, 6, 3, 5, 4, 5, 6
]

def contains_people(text: str):
    text = text.lower()
    people_keywords = [
        "person", "people", "man", "woman", "women", "men", "boy", "girl", "child", "children", "kids",
        "kid", "human", "face", "portrait", "selfie", "friends", "couple", "wedding", "bride", "groom",
        "student", "students", "body", "guy", "lady", "adult", "teen", "smile", "posing", "model",
        "family", "father", "mother", "son", "daughter", "group", "crowd", "head", "eyes", "mouth",
        "nose", "skin", "baby", "babies", "teacher", "worker", "doctor", "nurse"
    ]
    arabic_people = [
        "شخص", "اشخاص", "وجوه", "انسان", "بشر", "رجل", "امرأة", "نساء", "رجال", "طفل", "اطفال",
        "فتاة", "شباب", "صور شخصية", "عائلة", "مجموعة", "زفاف", "عرس", "صورة", "وجه", "أم", "أب",
        "أصدقاء", "طالب", "طلاب", "طالبة", "معلم", "معلمة", "موظف", "طبيب"
    ]
    all_keywords = people_keywords + arabic_people
    return any(word in text for word in all_keywords)

def is_shorts(width, height, duration, min_duration=7, max_duration=120):
    ratio = width / height if height > 0 else 1
    return (ratio < 0.7) and (min_duration <= duration <= max_duration)

def get_best_video_file(video_files):
    best = None
    for f in sorted(video_files, key=lambda vf: vf['height']):
        if f['height'] >= 360:
            if not best or f['height'] < best['height']:
                best = f
    return best

def get_pexels_shorts_videos(api_key, needed_duration, keywords):
    headers = {"Authorization": api_key}
    shorts = []
    for keyword in keywords:
        params = {"query": keyword, "per_page": 30}
        resp = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params)
        try:
            videos = resp.json().get('videos', [])
        except Exception:
            continue
        for v in videos:
            desc = (v.get("description") or "")
            user_name = (v.get("user", {}).get("name") or "")
            tags = [t for t in v.get("tags",[])]
            title = (v.get("title") or "")
            text = " ".join(tags) + " " + desc + " " + user_name + " " + title
            if contains_people(text):
                continue
            best_file = get_best_video_file(v["video_files"])
            if best_file and is_shorts(best_file["width"], best_file["height"], v["duration"]):
                shorts.append({"link": best_file["link"], "duration": v["duration"], "title": v.get("description",'')})
    return shorts

def get_pixabay_shorts_videos(api_key, needed_duration, keywords):
    shorts = []
    for keyword in keywords:
        params = {
            "key": api_key,
            "q": keyword,
            "per_page": 30,
            "video_type": "film",
            "safesearch": "true"
        }
        resp = requests.get("https://pixabay.com/api/videos/", params=params)
        try:
            videos = resp.json().get("hits", [])
        except Exception:
            continue
        for v in videos:
            tags = (v.get("tags") or "")
            user = (v.get("user") or "")
            title = (v.get("title") or "")
            text = tags + " " + user + " " + title
            if contains_people(text):
                continue
            best_file = None
            for quality, vid in v["videos"].items():
                if vid["height"] >= 360 and (not best_file or vid["height"] < best_file["height"]):
                    best_file = vid
            if best_file and is_shorts(best_file["width"], best_file["height"], v["duration"]):
                shorts.append({"link": best_file["url"], "duration": v["duration"], "title": v.get("tags",'')})
    return shorts

def get_mixkit_shorts_videos(needed_duration, keywords):
    mixkit_links = [
        "https://assets.mixkit.co/videos/download/mixkit-clouds-in-the-sky-123.mp4",
        "https://assets.mixkit.co/videos/download/mixkit-mountain-landscape-1233.mp4",
        "https://assets.mixkit.co/videos/download/mixkit-sunrise-in-the-mountains-1176.mp4",
        "https://assets.mixkit.co/videos/download/mixkit-forest-trees-1234.mp4",
        "https://assets.mixkit.co/videos/download/mixkit-stars-in-night-sky-1186.mp4",
        "https://assets.mixkit.co/videos/download/mixkit-moon-in-the-night-sky-1354.mp4",
        "https://assets.mixkit.co/videos/download/mixkit-rain-clouds-mountain-1173.mp4",
        "https://assets.mixkit.co/videos/download/mixkit-starry-night-sky-1214.mp4"
    ]
    shorts = []
    for link in mixkit_links:
        shorts.append({"link": link, "duration": 15, "title": "Mixkit Nature"})
    return shorts

def get_coverr_shorts_videos(needed_duration, keywords):
    coverr_links = [
        "https://www.coverr.co/s3/mp4/river.mp4",
        "https://www.coverr.co/s3/mp4/forest.mp4",
        "https://www.coverr.co/s3/mp4/space.mp4"
    ]
    shorts = []
    for link in coverr_links:
        shorts.append({"link": link, "duration": 20, "title": "Coverr Nature"})
    return shorts

def download_and_get_clip(url, used_links, resize=(1080,1920)):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, stream=True, headers=headers)
    if resp.status_code != 200:
        return None, None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as file:
        for chunk in resp.iter_content(chunk_size=1024*1024):
            if chunk:
                file.write(chunk)
        file.flush()
        clip = VideoFileClip(file.name).resize(newsize=resize)
        if clip.duration < 2:
            return None, file.name
        if url in used_links:
            return None, file.name
        used_links.add(url)
        return clip, file.name

def trim_silence(audio_segment, silence_thresh=-40, chunk_size=10):
    start_trim = silence.detect_leading_silence(audio_segment, silence_thresh, chunk_size)
    end_trim = silence.detect_leading_silence(audio_segment.reverse(), silence_thresh, chunk_size)
    duration = len(audio_segment)
    return audio_segment[start_trim:duration-end_trim]

def add_echo(sound, delay=250, attenuation=0.6):
    echo = sound - 20
    for i in range(1, 5):
        echo = echo.overlay(sound - int(20 + i*10), position=i*delay)
    return sound.overlay(echo)

def blur_frame(img, ksize=15):
    return cv2.GaussianBlur(img, (ksize|1, ksize|1), 0)

def ken_burns_effect(clip, zoom=1.1, pan_direction='left'):
    w, h = clip.size
    if pan_direction == 'left':
        return clip.fx(vfx.crop, x1=0, y1=0, x2=int(w*(1-1/zoom)), y2=0).fx(vfx.resize, width=w)
    elif pan_direction == 'right':
        return clip.fx(vfx.crop, x1=int(w*(1-1/zoom)), y1=0, x2=0, y2=0).fx(vfx.resize, width=w)
    else:
        return clip.fx(vfx.resize, lambda t: 1 + (zoom-1)*t/clip.duration)

def add_vignette(clip, strength=0.6):
    def vignette(image):
        rows, cols = image.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, 200)
        kernel_y = cv2.getGaussianKernel(rows, 200)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        for i in range(3):
            image[:,:,i] = image[:,:,i] * (1-strength) + mask * strength
        return image
    return clip.fl_image(vignette)

def montage_effects(clip, do_bw, do_vignette, do_zoom, do_blur, vignette_strength, blur_strength):
    if do_blur:
        clip = clip.fl_image(lambda img: blur_frame(img, ksize=int(blur_strength)))
    if do_zoom:
        clip = ken_burns_effect(clip, zoom=1.13, pan_direction=random.choice(['left','right','center']))
    if do_vignette:
        clip = add_vignette(clip, strength=vignette_strength)
    if do_bw:
        clip = clip.fx(vfx.blackwhite)
    return clip

# ----------- الكود المصحح لرسم النص العربي بشكل مشكّل، متصل، RTL -----------

def create_text_image(text, size, font_path="Amiri-Regular.ttf", fontsize=50):
    # ✅ إعادة تشكيل النص العربي ليظهر بالحروف المتصلة والتشكيل
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    img = Image.new("RGBA", size, (0,0,0,0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, fontsize)
    except:
        font = ImageFont.load_default()

    # تقسيم النص لعدة أسطر من اليمين لليسار
    lines = []
    words = bidi_text.split()
    line = ""
    for word in words:
        test_line = word if not line else word + " " + line
        try:
            bbox = draw.textbbox((0, 0), test_line, font=font)
            w = bbox[2] - bbox[0]
        except AttributeError:
            w, _ = font.getsize(test_line)
        if w <= size[0] - 40:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    total_text_height = len(lines) * fontsize + (len(lines)-1)*5
    y = (size[1] - total_text_height) // 2
    for l in reversed(lines):
        try:
            bbox = draw.textbbox((0, 0), l, font=font)
            w = bbox[2] - bbox[0]
        except AttributeError:
            w, _ = font.getsize(l)
        x = (size[0] - w) // 2
        draw.text((x, y), l, font=font, fill="white")
        y += fontsize + 5

    return np.array(img)

def split_text_for_timings(full_text, words_per_clip=4):
    words = full_text.split()
    clips = []
    i = 0
    while i < len(words):
        clips.append(" ".join(words[i:i+words_per_clip]))
        i += words_per_clip
    return clips

def prepare_ayah_texts(ayat_texts):
    bsm = "بسم الله الرحمن الرحيم"
    output = []
    joined = " ".join(ayat_texts)
    if joined.startswith(bsm):
        output.append(bsm)
        rest = joined[len(bsm):].strip()
        if rest:
            output.extend(split_text_for_timings(rest))
    else:
        output.extend(split_text_for_timings(joined))
    return output

def get_clip_position(position, video_size, text_img_height):
    w, h = video_size
    if position == "bottom":
        return ('center', h - text_img_height//2)
    elif position == "top":
        return ('center', 0)
    else: # center
        return ('center', (h - text_img_height)//2)

def get_ayah_text_and_translation(sura_idx, ayah_num):
    arabic, english = "", ""
    url_ar = f"https://api.alquran.cloud/v1/ayah/{sura_idx}:{ayah_num}/ar"
    r_ar = requests.get(url_ar)
    if r_ar.ok:
        arabic = r_ar.json().get("data", {}).get("text", "")
    url_en = f"https://api.alquran.cloud/v1/ayah/{sura_idx}:{ayah_num}/en.asad"
    r_en = requests.get(url_en)
    if r_en.ok:
        english = r_en.json().get("data", {}).get("text", "")
    return arabic, english

def get_keywords_from_cohere(arabic, english):
    co = cohere.Client(COHERE_API_KEY)
    prompt = f"""Given this Quran verse and its English translation, suggest 7-10 English visual keywords suitable for searching background videos (avoid humans, faces, people, and forbidden things).
Verse: {arabic}
Translation: {english}
List only the keywords, comma-separated:"""
    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=60,
        temperature=0.3,
        stop_sequences=["\n"]
    )
    kws = response.generations[0].text.strip()
    return [k.strip() for k in kws.replace('\n','').split(',') if k.strip()]


# ========== Streamlit App ==========
st.set_page_config(page_title="فيديو قرآن شورتس ذكي", layout="centered")
st.title("أنشئ فيديو قرآن قصير (شورتس) بخلفية ذكية وتأثيرات متقدمة")

text_position_choice = st.selectbox(
    "اختر مكان عرض النص على الفيديو",
    [("أسفل", 'bottom'), ("وسط", 'center'), ("أعلى", 'top')],
    format_func=lambda x: x[0]
)
text_position = text_position_choice[1]

qari_names = [q["name"] for q in QURAA]
selected_qari_idx = st.selectbox("اختر القارئ:", options=range(len(qari_names)), format_func=lambda i: qari_names[i])
selected_qari = QURAA[selected_qari_idx]["id"]

sura_idx = st.selectbox("اختر السورة:", options=range(1, 115), format_func=lambda i: f"{i}. {SURA_NAMES[i-1]}")
ayah_count = SURA_AYAHS[sura_idx-1]

col1, col2 = st.columns(2)
with col1:
    from_ayah = st.number_input("من الآية رقم:", min_value=1, max_value=ayah_count, value=1)
with col2:
    to_ayah = st.number_input("إلى الآية رقم:", min_value=from_ayah, max_value=ayah_count, value=from_ayah)

video_sources = st.multiselect(
    "اختر مصادر الفيديو (يمكنك تحديد أكثر من مصدر):",
    options=["Pexels", "Pixabay", "Mixkit", "Coverr"],
    default=["Pexels", "Pixabay", "Mixkit", "Coverr"]
)

st.markdown("**خيارات تخصيص المؤثرات**:")

col3, col4, col5 = st.columns(3)
with col3:
    video_speed = st.slider("سرعة الفيديو", 0.5, 2.0, 1.0, 0.05)
with col4:
    blur_strength = st.slider("شدة الضباب (Blur)", 0, 50, 15)
with col5:
    vignette_strength = st.slider("قوة التدرج اللوني (Vignette)", 0.0, 1.0, 0.3)

do_bw = st.checkbox("أبيض وأسود", False)
do_vignette = st.checkbox("تدرج لوني حول الحواف", True)
do_zoom = st.checkbox("تأثير زوم متحرك (Ken Burns)", True)
do_blur = st.checkbox("ضباب (Blur)", True)
do_echo = st.checkbox("صدى على الصوت", True)

aspect = st.selectbox("أبعاد الفيديو:", ["عمودي (9:16)", "أفقي (16:9)", "مربع (1:1)"])
aspect_map = {"عمودي (9:16)": (1080,1920), "أفقي (16:9)": (1920,1080), "مربع (1:1)": (1080,1080)}

uploaded_file = st.file_uploader("ارفع فيديو/صورة خلفية (اختياري)")

st.markdown("الخلفية يمكن أن تكون أي موضوع (مدينة، هندسة، فضاء... إلخ)، الشرط الوحيد ألا تحتوي على أشخاص أو وجوه.")

if st.button("إنشاء الفيديو"):
    try:
        st.info("تحليل الآيات وجلب الكلمات المفتاحية الذكية...")
        ayah_ar, ayah_en = get_ayah_text_and_translation(sura_idx, from_ayah)
        keywords = get_keywords_from_cohere(ayah_ar, ayah_en)
        st.caption("الكلمات المفتاحية الذكية: " + ', '.join(keywords))

        with st.spinner("جاري تحميل ودمج الصوت..."):
            merged = None
            missing_ayahs = []
            default_silence_ms = 3000
            ayat_texts = []
            for ayah in range(int(from_ayah), int(to_ayah)+1):
                mp3_url = f"https://everyayah.com/data/{selected_qari}/{sura_idx:03d}{ayah:03d}.mp3"
                r = requests.get(mp3_url)
                ar, _ = get_ayah_text_and_translation(sura_idx, ayah)
                ayat_texts.append(ar)
                if r.status_code != 200:
                    missing_ayahs.append(ayah)
                    segment = AudioSegment.silent(duration=default_silence_ms)
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_ayah_file:
                        temp_ayah_file.write(r.content)
                        temp_ayah_file.flush()
                        segment = AudioSegment.from_mp3(temp_ayah_file.name)
                        segment = trim_silence(segment)
                merged = segment if merged is None else merged + segment
            if not merged:
                st.error("لم يتم العثور على أي آية صوتية متاحة لهذا القارئ في هذا المقطع.")
                st.stop()
            if do_echo:
                merged = add_echo(merged)
            fade_in_duration_ms = 2000
            fade_out_duration_ms = min(2000, int(len(merged) * 0.1))
            merged = merged.fade_in(fade_in_duration_ms).fade_out(fade_out_duration_ms)
            if missing_ayahs:
                st.warning(f"بعض الآيات غير متوفرة وتم وضع صمت مكانها: {', '.join(str(a) for a in missing_ayahs)}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as merged_file:
                merged.export(merged_file.name, format="mp3")
                audio_path = merged_file.name

        audio_clip = AudioFileClip(audio_path).fx(vfx.speedx, video_speed)
        duration = audio_clip.duration

        resize = aspect_map[aspect]
        video_clips = []
        used_links = set()
        if uploaded_file:
            file_ext = os.path.splitext(uploaded_file.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as f:
                f.write(uploaded_file.read())
                f.flush()
                if file_ext.lower() in ['.mp4', '.mov', '.avi']:
                    user_clip = VideoFileClip(f.name).resize(newsize=resize).subclip(0, min(duration, VideoFileClip(f.name).duration))
                else:
                    user_clip = ImageClip(f.name, duration=duration).resize(newsize=resize)
                video_clips.append(user_clip)
        else:
            with st.spinner("جاري تحميل وتجهيز فيديوهات الخلفية..."):
                shorts = []
                if "Pexels" in video_sources:
                    shorts += get_pexels_shorts_videos(PEXELS_API_KEY, duration, keywords)
                if "Pixabay" in video_sources:
                    shorts += get_pixabay_shorts_videos(PIXABAY_API_KEY, duration, keywords)
                if "Mixkit" in video_sources:
                    shorts += get_mixkit_shorts_videos(duration, keywords)
                if "Coverr" in video_sources:
                    shorts += get_coverr_shorts_videos(duration, keywords)
                random.shuffle(shorts)
                downloaded_duration = 0.0
                shorts_index = 0
                while downloaded_duration < duration and shorts_index < len(shorts):
                    video_obj = shorts[shorts_index]
                    link = video_obj["link"]
                    shorts_index += 1
                    if link in used_links:
                        continue
                    clip, fname = download_and_get_clip(link, used_links, resize=resize)
                    if not clip:
                        continue
                    clip = montage_effects(
                        clip,
                        do_bw=do_bw,
                        do_vignette=do_vignette,
                        do_zoom=do_zoom,
                        do_blur=do_blur,
                        vignette_strength=vignette_strength,
                        blur_strength=blur_strength
                    )
                    if downloaded_duration + clip.duration > duration:
                        clip = clip.subclip(0, duration-downloaded_duration)
                    video_clips.append(clip)
                    downloaded_duration += clip.duration
                if not video_clips or downloaded_duration < duration:
                    st.error("لم يتم العثور على فيديوهات كافية أو مناسبة لتغطية مدة الصوت. حاول مجددًا أو قلل المقطع الصوتي.")
                    st.stop()

        final_video = concatenate_videoclips(video_clips).subclip(0, duration).fx(vfx.speedx, video_speed)
        final = final_video.set_audio(audio_clip).set_duration(duration)

        text_chunks = prepare_ayah_texts(ayat_texts)
        chunk_count = len(text_chunks)
        chunk_duration = duration / chunk_count if chunk_count else duration

        font_path = "Amiri-Regular.ttf"
        if not os.path.exists(font_path):
            font_path = "Arial"
        img_height = 200
        video_w, video_h = resize

        text_clips = []
        for i, chunk in enumerate(text_chunks):
            text_img = create_text_image(chunk, (video_w, img_height), font_path, 50)
            start_time = i * chunk_duration
            end_time = min((i+1) * chunk_duration, duration)
            text_clip = (
                ImageClip(text_img, duration=end_time-start_time)
                .set_start(start_time)
                .set_position(get_clip_position(text_position, resize, img_height))
            )
            text_clips.append(text_clip)

        final_with_text = CompositeVideoClip([final] + text_clips)

        output_path = "quran_shorts.mp4"
        with st.spinner("جاري تصدير الفيديو النهائي..."):
            final_with_text.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
        st.success("تم إنشاء الفيديو بنجاح!")
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("تحميل الفيديو", f, file_name="quran_shorts.mp4", mime="video/mp4")

        # تنظيف الملفات المؤقتة
        try:
            os.remove(audio_path)
            for clip in video_clips:
                if hasattr(clip, 'filename') and os.path.exists(clip.filename):
                    os.remove(clip.filename)
        except Exception:
            pass

    except Exception as e:
        st.error(f"حدث خطأ: {e}")
