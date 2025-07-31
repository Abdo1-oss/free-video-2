import streamlit as st
import os
import requests
import shutil
import tempfile
from gtts import gTTS
from textblob import TextBlob
from moviepy.editor import (
    ImageClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip,
    TextClip, VideoFileClip
)
from PIL import Image

# إعداد صفحة ستريملت
st.set_page_config(page_title="منشئ فيديو الذكاء الاصطناعي", layout="wide")
st.title("🎬 منشئ فيديو احترافي تلقائي بالذكاء الاصطناعي")
st.markdown(
    "أدخل عنوانًا فقط وسيتم إنشاء فيديو كامل تلقائيًا مع تعليق صوتي وصور مشاهد مناسبة وأصوات طبيعة مجانية."
)

# واجهة المستخدم
video_title = st.text_input("🔹 أدخل عنوان الفيديو أو الكلمة المفتاحية", "")
video_type = st.selectbox(
    "🔸 نوع الفيديو",
    ["تحفيزي", "تعليمي", "وثائقي", "ديني", "تقني", "رياضي", "أخرى"]
)
video_quality = st.selectbox("🔸 جودة الفيديو", ["720p", "1080p"])
add_logo = st.file_uploader("لوغو القناة (اختياري)", type=["png", "jpg", "jpeg"])
add_intro = st.checkbox("إضافة مقدمة تلقائية", value=True)
add_outro = st.checkbox("إضافة خاتمة تلقائية", value=True)
shorts_mode = st.checkbox("توليد فيديو شورتس تلقائيًا")
st.markdown("---")

# وظائف الذكاء الاصطناعي والميديا

def generate_script(title, video_type):
    # استخدام Cohere لتوليد السكربت
    api_key = os.getenv("K1GW0y2wWiwW7xlK7db7zZnqX7sxfRVGiWopVfCD", "")
    if not api_key:
        # سكربت افتراضي إذا لم يتوفر مفتاح
        return (
            f"Welcome to our {video_type} video about {title}.\n"
            "Let's begin the journey...\n"
            "Thank you for watching!"
        )
    prompt = (
        f"Write a {video_type} video script in English based on the title: '{title}'. "
        "Make it concise and divide it into clear scenes with newlines for each scene."
    )
    try:
        import cohere
        co = cohere.Client(api_key)
        response = co.generate(
            model="command",
            prompt=prompt,
            max_tokens=700,
            temperature=0.7,
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return (
            f"Welcome to our {video_type} video about {title}.\n"
            "Let's begin the journey...\n"
            "Thank you for watching!"
        )

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

def split_script_to_scenes(script):
    scenes = [s.strip() for s in script.split('\n') if len(s.strip()) > 20]
    if not scenes:
        scenes = [sent.strip() for sent in script.split('.') if len(sent.strip()) > 20]
    return scenes

def search_image_unsplash(query):
    try:
        url = f"https://source.unsplash.com/1280x720/?{query.replace(' ','%20')}"
        response = requests.get(url, stream=True, timeout=8)
        if response.status_code == 200 and response.headers.get("Content-Type", "").startswith("image"):
            temp_path = tempfile.mktemp(suffix=".jpg")
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(response.raw, f)
            return temp_path
    except Exception:
        return None
    return None

def search_image_pexels(query):
    api_key = os.getenv("pLcIoo3oNdhqna28AfdaBYhkE3SFps9oRGuOsxY3JTe92GcVDZpwZE9i", "")
    if not api_key:
        return None
    headers = {"Authorization": api_key}
    url = f"https://api.pexels.com/v1/search?query={query}&per_page=1"
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        data = resp.json()
        if data.get("photos"):
            img_url = data["photos"][0]["src"]["large"]
            r = requests.get(img_url, stream=True, timeout=8)
            temp_path = tempfile.mktemp(suffix=".jpg")
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
            return temp_path
    except Exception:
        return None
    return None

def search_image_pixabay(query):
    api_key = os.getenv("50380897-76243eaec536038f687ff8e15", "")
    if not api_key:
        return None
    url = f"https://pixabay.com/api/?key={api_key}&q={query}&image_type=photo"
    try:
        resp = requests.get(url, timeout=8)
        data = resp.json()
        if data.get("hits"):
            img_url = data["hits"][0]["largeImageURL"]
            r = requests.get(img_url, stream=True, timeout=8)
            temp_path = tempfile.mktemp(suffix=".jpg")
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
            return temp_path
    except Exception:
        return None
    return None

def search_image(query):
    # الأولوية: بيكسيلز > بيكسابي > Unsplash > صورة افتراضية
    img = search_image_pexels(query)
    if img:
        return img
    img = search_image_pixabay(query)
    if img:
        return img
    img = search_image_unsplash(query)
    if img:
        return img
    # صورة افتراضية عند الفشل
    temp_path = tempfile.mktemp(suffix=".jpg")
    r = requests.get("https://placehold.co/1280x720?text=No+Image", stream=True)
    with open(temp_path, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    return temp_path

def get_images_for_scenes(scenes, video_type):
    images = []
    for i, scene in enumerate(scenes):
        query = " ".join(scene.split(' ')[:5]) + f" {video_type}"
        img_path = search_image(query)
        images.append(img_path)
    return images

def generate_voiceover(script):
    tts = gTTS(text=script, lang='en')
    out_path = "voiceover.mp3"
    tts.save(out_path)
    return out_path

def download_nature_sound():
    # أصوات طبيعة مجانية من Mixkit
    nature_links = [
        "https://assets.mixkit.co/sfx/preview/mixkit-forest-stream-water-1226.mp3",
        "https://assets.mixkit.co/sfx/preview/mixkit-small-bird-in-the-forest-17.mp3",
        "https://assets.mixkit.co/sfx/preview/mixkit-ocean-waves-ambience-1183.mp3",
        "https://assets.mixkit.co/sfx/preview/mixkit-calm-water-small-river-1172.mp3",
        "https://assets.mixkit.co/sfx/preview/mixkit-rain-loop-2395.mp3",
    ]
    url = nature_links[0]  # يمكنك التبديل عشوائيًا إذا رغبت
    local_path = "nature.mp3"
    if not os.path.exists(local_path):
        r = requests.get(url, stream=True, timeout=15)
        with open(local_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return local_path

def create_video(scenes, images, voiceover_path, nature_path, quality, logo_file=None):
    clips = []
    duration_per_scene = max(4, int(15 / max(1, len(scenes))))
    for img_path in images:
        clip = ImageClip(img_path).set_duration(duration_per_scene)
        clips.append(clip)
    video = concatenate_videoclips(clips, method="compose")
    # الصوت
    audio_voice = AudioFileClip(voiceover_path)
    audio_nature = AudioFileClip(nature_path).volumex(0.22)
    final_audio = CompositeAudioClip([audio_voice, audio_nature])
    video = video.set_audio(final_audio)
    # جودة الفيديو
    res_map = {"720p": (1280, 720), "1080p": (1920, 1080)}
    size = res_map.get(quality, (1280, 720))
    video = video.resize(newsize=size)
    # اللوجو
    if logo_file is not None:
        logo_img = Image.open(logo_file).convert("RGBA")
        logo_img = logo_img.resize((110, 110))
        temp_logo_path = tempfile.mktemp(suffix=".png")
        logo_img.save(temp_logo_path)
        logo_clip = (
            ImageClip(temp_logo_path)
            .set_duration(video.duration)
            .set_position(("right", "top"))
            .resize(height=90)
            .margin(right=10, top=10, opacity=0)
        )
        video = video.set_audio(final_audio)
        video = concatenate_videoclips([video])
    out_path = "final_video.mp4"
    video.write_videofile(out_path, fps=24, codec='libx264', audio_codec="aac", verbose=False, logger=None)
    return out_path

def add_intro_outro(video_path, intro_text="مرحبًا بكم!", outro_text="شكرًا للمشاهدة!"):
    main_clip = VideoFileClip(video_path)
    size = main_clip.size
    intro = TextClip(intro_text, fontsize=72, color='white', bg_color='green', size=size, font="Arial-Bold").set_duration(2.5)
    outro = TextClip(outro_text, fontsize=55, color='white', bg_color='black', size=size, font="Arial-Bold").set_duration(2.3)
    final = concatenate_videoclips([intro, main_clip, outro])
    out_path = "video_with_intro_outro.mp4"
    final.write_videofile(out_path, fps=24, codec='libx264', audio_codec="aac", verbose=False, logger=None)
    return out_path

def create_shorts_version(video_path):
    clip = VideoFileClip(video_path).subclip(0, min(60, int(VideoFileClip(video_path).duration)))
    w, h = clip.size
    target_h = 1280
    target_w = 720
    # crop to center and resize to 9:16
    if w/h > 9/16:
        new_w = int(h * 9 / 16)
        x1 = (w - new_w) // 2
        shorts = clip.crop(x1=x1, y1=0, x2=x1+new_w, y2=h)
    else:
        shorts = clip
    shorts = shorts.resize((target_w, target_h))
    out_path = "shorts.mp4"
    shorts.write_videofile(out_path, fps=24, codec='libx264', audio_codec="aac", verbose=False, logger=None)
    return out_path

# زر التشغيل
if st.button("🚀 ابدأ"):
    if not video_title.strip():
        st.warning("الرجاء إدخال عنوان الفيديو.")
        st.stop()
    # تنظيف الملفات المؤقتة
    for f in ["voiceover.mp3", "nature.mp3", "final_video.mp4", "video_with_intro_outro.mp4", "shorts.mp4"]:
        try:
            os.remove(f)
        except Exception:
            pass
    st.info("جاري توليد السكربت...")
    script = generate_script(video_title, video_type)
    sentiment = analyze_sentiment(script)
    scenes = split_script_to_scenes(script)
    st.info("جاري توليد الصور...")
    images = get_images_for_scenes(scenes, video_type)
    st.info("جاري توليد التعليق الصوتي...")
    voiceover_path = generate_voiceover(script)
    st.info("جاري تحميل صوت الطبيعة...")
    nature_path = download_nature_sound()
    st.info("جاري إنشاء الفيديو...")
    video_path = create_video(
        scenes=scenes,
        images=images,
        voiceover_path=voiceover_path,
        nature_path=nature_path,
        quality=video_quality,
        logo_file=add_logo,
    )
    if add_intro or add_outro:
        intro_text = "مرحبًا بكم!" if add_intro else ""
        outro_text = "شكرًا للمشاهدة!" if add_outro else ""
        video_path = add_intro_outro(video_path, intro_text, outro_text)
    st.success("✅ تم إنشاء الفيديو بنجاح!")
    st.video(video_path)
    with open(video_path, "rb") as f:
        st.download_button("⬇️ تحميل الفيديو", f, file_name="video.mp4")
    if shorts_mode:
        st.info("جاري إنشاء نسخة شورتس...")
        shorts_path = create_shorts_version(video_path)
        with open(shorts_path, "rb") as f:
            st.download_button("⬇️ تحميل نسخة شورتس", f, file_name="shorts.mp4")
        st.video(shorts_path)

st.caption("© جميع الحقوق محفوظة – برمجة: Abdo1-oss")
