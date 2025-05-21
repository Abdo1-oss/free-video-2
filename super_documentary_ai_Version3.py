import streamlit as st
import requests
import tempfile
import os
import time
import json
import traceback

from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageClip, CompositeVideoClip, AudioFileClip, TextClip
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
from googletrans import Translator
import wikipedia
from transformers import pipeline

# ==============================
# أداة كشف الأخطاء البرمجية
# ==============================
def run_with_debug(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error("حذراً، لم نتمكن من إنتاج الفيديو.")
        st.write("تفاصيل الخطأ البرمجي:")
        st.write(traceback.format_exc())
        st.exception(e)  # عرض الخطأ البرمجي بشكل واضح
        print(traceback.format_exc())
        return None

PEXELS_API_KEY = st.secrets.get("PEXELS_API_KEY", "")
UNSPLASH_ACCESS_KEY = st.secrets.get("UNSPLASH_ACCESS_KEY", "")
PIXABAY_API_KEY = st.secrets.get("PIXABAY_API_KEY", "")

@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summarizer = get_summarizer()
    return summarizer(text, max_length=180, min_length=60, do_sample=False)[0]['summary_text']

def suggest_text_from_keyword(keyword, lang="en"):
    wikipedia.set_lang(lang)
    try:
        return wikipedia.summary(keyword, sentences=4)
    except:
        return f"No summary found for {keyword}."

def translate_text(text, dest_lang):
    try:
        translator = Translator()
        return translator.translate(text, dest=dest_lang).text
    except:
        return text

def search_pexels_videos(query, per_page=2):
    if not PEXELS_API_KEY: return []
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/videos/search?query={query}&per_page={per_page}"
    data = requests.get(url, headers=headers).json()
    return [v["video_files"][0]["link"] for v in data.get("videos", []) if v.get("video_files")]

def search_pexels_photos(query, per_page=3):
    if not PEXELS_API_KEY: return []
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/v1/search?query={query}&per_page={per_page}"
    data = requests.get(url, headers=headers).json()
    return [photo["src"]["large"] for photo in data.get("photos", [])]

def search_unsplash_photos(query, per_page=3):
    if not UNSPLASH_ACCESS_KEY: return []
    url = f"https://api.unsplash.com/search/photos?query={query}&per_page={per_page}&client_id={UNSPLASH_ACCESS_KEY}"
    data = requests.get(url).json()
    return [photo["urls"]["regular"] for photo in data.get("results", [])]

def search_pixabay_photos(query, per_page=3):
    if not PIXABAY_API_KEY: return []
    url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={query}&per_page={per_page}&image_type=photo"
    data = requests.get(url).json()
    return [hit["largeImageURL"] for hit in data.get("hits", [])]

def search_wikimedia_photos(query, limit=2):
    url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={query}&gsrlimit={limit}&prop=imageinfo&iiprop=url&format=json"
    response = requests.get(url)
    pages = response.json().get("query", {}).get("pages", {})
    return [v["imageinfo"][0]["url"] for v in pages.values() if "imageinfo" in v]

def generate_voice(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

def ken_burns_effect(image_path, duration=5, zoom=1.2):
    clip = ImageClip(image_path)
    w, h = clip.size
    return (clip
            .resize(lambda t : 1 + (zoom-1)*t/duration)
            .set_position(lambda t: ('center', int(0.5*(h - h*(1 + (zoom-1)*t/duration)))))
            .set_duration(duration)
    )

def add_text_to_image(image_path, text, output_path="output_img.png", color="#FFFFFF", size=28, pos="bottom"):
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except:
        font = ImageFont.load_default()
    text_width, text_height = draw.textsize(text, font)
    width, height = image.size
    if pos == "top":
        x = (width - text_width) // 2
        y = 10
    elif pos == "center":
        x = (width - text_width) // 2
        y = (height - text_height) // 2
    else:
        x = (width - text_width) // 2
        y = height - text_height - 15
    draw.rectangle([x-10, y-5, x+text_width+10, y+text_height+5], fill=(0,0,0,128))
    draw.text((x, y), text, font=font, fill=color)
    image.save(output_path)
    return output_path

def generate_srt(text, lang, duration, output_path="output.srt"):
    lines = [l for l in text.split('\n') if l.strip()]
    per_line = max(duration // (len(lines) or 1), 2)
    srt = ""
    for i, l in enumerate(lines):
        start = i * per_line
        end = min((i+1) * per_line, duration)
        srt += f"{i+1}\n00:00:{start:02d},000 --> 00:00:{end:02d},000\n{l}\n\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt)
    return output_path

def create_final_video(video_urls, photo_urls, audio_path, logo_path=None, music_path=None,
                      output_path="final_output.mp4", video_duration=120, overlay_texts=[],
                      watermark_text="", color="#FFFFFF", text_size=28, text_pos="bottom", gif_export=False, square_export=False):
    clips = []
    for url in video_urls:
        try:
            clips.append(VideoFileClip(url).subclip(0, min(video_duration, 10)))
        except: pass
    if not clips and photo_urls:
        duration_per_image = max(video_duration // (len(photo_urls) or 1), 2)
        for i, url in enumerate(photo_urls):
            img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
            img_data = requests.get(url).content
            with open(img_path, "wb") as f: f.write(img_data)
            if overlay_texts and i < len(overlay_texts):
                img_path = add_text_to_image(img_path, overlay_texts[i], color=color, size=text_size, pos=text_pos)
            img_clip = ken_burns_effect(img_path, duration=duration_per_image)
            clips.append(img_clip)
    if not clips:
        return None
    final_clip = concatenate_videoclips(clips, method="compose")
    audio_clip = AudioFileClip(audio_path)
    final_clip = final_clip.set_audio(audio_clip).subclip(0, audio_clip.duration)
    if music_path:
        try:
            music_clip = AudioFileClip(music_path).volumex(0.15)
            final_audio = CompositeAudioClip([final_clip.audio, music_clip])
            final_clip = final_clip.set_audio(final_audio)
        except: pass
    if logo_path:
        logo = (ImageClip(logo_path)
                .set_duration(final_clip.duration)
                .resize(height=50)
                .set_pos(("right", "top")).margin(right=8, top=8, opacity=0))
        final_clip = CompositeVideoClip([final_clip, logo])
    final_clip = final_clip.fadein(1).fadeout(1)
    if watermark_text:
        txt_clip = (TextClip(watermark_text, fontsize=24, color='white', font='Arial-Bold', bg_color='black', size=(final_clip.size[0], 30))
                    .set_position(("center", "bottom")).set_duration(final_clip.duration).set_opacity(0.4))
        final_clip = CompositeVideoClip([final_clip, txt_clip])
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    if gif_export:
        final_clip.write_gif(output_path.replace(".mp4", ".gif"), fps=10)
    if square_export:
        min_dim = min(final_clip.size)
        square_clip = final_clip.crop(x_center=final_clip.size[0]/2, y_center=final_clip.size[1]/2, width=min_dim, height=min_dim)
        square_clip.write_videofile(output_path.replace(".mp4", "_square.mp4"))
        square_clip.close()
    for clip in clips: clip.close()
    audio_clip.close()
    if music_path: music_clip.close()
    final_clip.close()
    return output_path

def save_project(project_data, path="my_project.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project_data, f, ensure_ascii=False, indent=2)

def load_project(path="my_project.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

st.set_page_config(page_title="وثائقي خارق بالذكاء الاصطناعي", layout="wide")
st.title("🎬 وثائقي خارق | Super Documentary AI")

mode = st.radio("نوع المشروع", ["ابدأ مشروع جديد", "استرجع مشروع من ملف"])

if mode == "استرجع مشروع من ملف":
    uploaded_project = st.file_uploader("ارفع ملف المشروع (json):", type="json")
    if uploaded_project:
        project_data = json.load(uploaded_project)
        st.success("تم استرجاع المشروع!")
        st.json(project_data)
else:
    st.markdown("**اكتب كلمة مفتاحية أو نص وثائقي كامل وسيتم إنتاج فيديو خارق مجاناً!**")
    kw = st.text_input("الكلمة المفتاحية أو موضوع الوثائقي")
    script_text = st.text_area("أو الصق نص الوثائقي هنا (اختياري):", height=200)
    lang_option = st.selectbox("لغة التعليق الصوتي:", options=["en", "ar", "fr", "es", "de", "ru"])
    sources_selected = st.multiselect(
        "مصادر الصور والفيديو:",
        options=["Pexels", "Unsplash", "Pixabay", "Wikimedia", "Wikipedia"],
        default=["Pexels", "Wikipedia"]
    )
    logo_file = st.file_uploader("شعار الفيديو (اختياري):", type=["png", "jpg", "jpeg"])
    music_file = st.file_uploader("موسيقى مجانية (اختياري):", type=["mp3", "wav"])
    short_video = st.checkbox("إنتاج نسخة قصيرة (Short)", value=True)
    watermark = st.text_input("نص العلامة المائية (اختياري):", value="@SuperAI")
    video_length_option = st.slider("مدة الفيديو بالدقائق:", min_value=1, max_value=15, value=3)
    color = st.color_picker("اختر لون النص", "#ffffff")
    text_size = st.slider("حجم النص", 14, 60, 28)
    text_pos = st.radio("مكان النص", options=["top", "center", "bottom"], index=2)
    gif_export = st.checkbox("تصدير كـ GIF", value=False)
    square_export = st.checkbox("تصدير نسخة مربعة للفيديو (انستجرام)", value=False)
    save_proj = st.checkbox("حفظ المشروع كملف (للاستكمال لاحقاً)", value=False)

    if st.button("ابدأ الإبداع!"):
        if not kw and not script_text.strip():
            st.error("يرجى إدخال كلمة مفتاحية أو نص الوثائقي.")
        else:
            def creative_workflow():
                st.write("بدأ التنفيذ!")  # سطر التحقق
                if script_text.strip():
                    final_text = summarize_text(script_text)
                else:
                    final_text = suggest_text_from_keyword(kw, lang=lang_option)
                if lang_option != "en":
                    final_text = translate_text(final_text, lang_option)
                video_urls, photo_urls = [], []
                if "Pexels" in sources_selected:
                    video_urls += search_pexels_videos(kw)
                    photo_urls += search_pexels_photos(kw)
                if "Unsplash" in sources_selected:
                    photo_urls += search_unsplash_photos(kw)
                if "Pixabay" in sources_selected:
                    photo_urls += search_pixabay_photos(kw)
                if "Wikimedia" in sources_selected:
                    photo_urls += search_wikimedia_photos(kw)
                overlay_texts = [t.strip() for t in final_text.split('\n') if t.strip()]
                audio_path = generate_voice(final_text, lang=lang_option)
                logo_path = None
                if logo_file:
                    logo_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                    image = Image.open(logo_file)
                    image.save(logo_path)
                music_path = None
                if music_file:
                    music_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                    music_file.seek(0)
                    with open(music_path, "wb") as f:
                        f.write(music_file.read())
                output_video_path = f"output_{int(time.time())}.mp4"
                duration = video_length_option*60 if not short_video else 60
                final_video = create_final_video(
                    video_urls=video_urls,
                    photo_urls=photo_urls,
                    audio_path=audio_path,
                    logo_path=logo_path,
                    music_path=music_path,
                    output_path=output_video_path,
                    video_duration=duration,
                    overlay_texts=overlay_texts,
                    watermark_text=watermark,
                    color=color,
                    text_size=text_size,
                    text_pos=text_pos,
                    gif_export=gif_export,
                    square_export=square_export
                )
                srt_path = generate_srt(final_text, lang_option, duration)
                if final_video:
                    st.success("تم الإنشاء! شاهد نتيجتك 👇")
                    st.video(final_video)
                    with open(final_video, "rb") as f:
                        st.download_button(label="تحميل الفيديو", data=f, file_name="documentary_video.mp4", mime="video/mp4")
                    with open(srt_path, "rb") as f:
                        st.download_button(label="تحميل ملف الترجمة SRT", data=f, file_name="documentary_video.srt", mime="text/srt")
                    if gif_export:
                        gif_path = output_video_path.replace(".mp4", ".gif")
                        with open(gif_path, "rb") as f:
                            st.download_button(label="تحميل GIF", data=f, file_name="documentary_video.gif", mime="image/gif")
                    if square_export:
                        sq_path = output_video_path.replace(".mp4", "_square.mp4")
                        with open(sq_path, "rb") as f:
                            st.download_button(label="تحميل فيديو مربع", data=f, file_name="documentary_video_square.mp4", mime="video/mp4")
                    if save_proj:
                        project_data = {
                            "kw": kw, "final_text": final_text, "lang": lang_option,
                            "sources": sources_selected, "watermark": watermark,
                            "video_length": video_length_option, "color": color,
                            "text_size": text_size, "text_pos": text_pos
                        }
                        proj_path = f"project_{int(time.time())}.json"
                        save_project(project_data, proj_path)
                        with open(proj_path, "rb") as f:
                            st.download_button(label="تحميل ملف المشروع", data=f, file_name="documentary_project.json", mime="application/json")
                else:
                    st.error("عذراً، لم نتمكن من إنتاج الفيديو.")

            with st.spinner("جاري الإبداع ..."):
                run_with_debug(creative_workflow)
