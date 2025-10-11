import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import os
import sys

# إصلاح مشكلة tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from gtts import gTTS
import time

# إضافة المسار الحالي
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# استيراد الوحدات المطلوبة
try:
    from multi import MultiEraEgyptianRAG
except ImportError as e:
    st.error("❌ لم يتم العثور على ملف multi.py")
    st.info("تأكد من وجود ملف multi.py في نفس المجلد")
    st.code(str(e))
    st.stop()

# تحميل المتغيرات
load_dotenv()

# التحقق من وجود API Key
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY غير موجود في ملف .env")
    st.stop()

# إعداد الصفحة
st.set_page_config(
    page_title="🕰️ Time Travel Chatbot", 
    page_icon="✈️", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# ===== CSS للتصميم الاحترافي =====
st.markdown("""
<style>
    /* الخلفية الرئيسية */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* صندوق المحادثة */
    .main .block-container {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* العناوين */
    h1 {
        color: #667eea;
        text-align: center;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2, h3 {
        color: #764ba2;
    }
    
    /* الأزرار */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* صندوق الإجابة */
    .answer-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #000 !important; /* النصوص بالأسود */
    }
    .answer-box p, .answer-box strong, .answer-box span, .answer-box div {
        color: #000 !important;
        -webkit-text-fill-color: #000 !important;
    }
    
    /* قائمة الاختيار */
    .stSelectbox > div > div {
        background: white;
        border-radius: 10px;
        border: 2px solid #667eea;
        color: #000 !important; /* النصوص بالأسود */
    }
    
    /* حقل النص */
    .stTextArea textarea, .stTextInput input {
        border-radius: 10px;
        border: 2px solid #667eea;
        font-size: 16px;
        color: #fff !important;  /* النصوص بالأبيض */
        background: #1e1e1e;     /* خلفية داكنة عشان الأبيض يبان */
    }
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: rgba(255,255,255,0.7) !important;  /* placeholder بالأبيض */
        opacity: 1 !important;
    }
    
    /* الشريط الجانبي */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    .css-1d391kg p, .css-1d391kg li,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: rgba(255,255,255,0.9) !important;
    }
    
    /* صندوق المعلومات */
    .stInfo {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 5px solid #667eea;
    }
    
    /* شاشة البداية */
    .welcome-container {
        text-align: center;
        padding: 3rem 2rem;
    }
    
    .welcome-emoji {
        font-size: 120px;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    
    .welcome-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
    }
    
    .welcome-subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)



# ===== دالة توليد الصوت =====
def text_to_speech(text, era_name):
    """تحويل النص إلى صوت عربي"""
    try:
        # إضافة مقدمة درامية حسب العصر
        era_intros = {
            "pharaonic": "من أعماق التاريخ الفرعوني العظيم... ",
            "greek": "من زمن البطالمة والحضارة اليونانية... ",
            "roman": "من عصر الرومان العظماء... ",
            "medieval": "من العصور الوسطى المجيدة... "
        }
        
        intro = era_intros.get(era_name, "")
        # أخذ أول 400 حرف فقط
        short_text = text[:400] if len(text) > 400 else text
        full_text = intro + short_text
        
        # إنشاء مجلد مؤقت
        audio_dir = Path("temp_audio")
        audio_dir.mkdir(exist_ok=True)
        
        # اسم فريد للملف
        timestamp = int(time.time())
        audio_path = audio_dir / f"audio_{era_name}_{timestamp}.mp3"
        
        # توليد الصوت
        st.info("🎤 جاري التسجيل...")
        tts = gTTS(text=full_text, lang='ar', slow=False, tld='com')
        tts.save(str(audio_path))
        
        # التأكد من حفظ الملف
        if audio_path.exists():
            st.success("✅ تم التسجيل بنجاح!")
            return audio_path
        else:
            st.error("❌ فشل حفظ الملف الصوتي")
            return None
        
    except Exception as e:
        st.error(f"⚠️ خطأ في توليد الصوت: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ===== واجهة البداية =====
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-emoji">✈️</div>
        <h1 class="welcome-title">Time Travel Chatbot</h1>
        <p class="welcome-subtitle">سافر عبر العصور التاريخية المصرية واستمع لحكايات الأبطال 🌍</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 ابدأ الرحلة", use_container_width=True):
            st.session_state.started = True
            st.rerun()

else:
    # ===== واجهة الشات بوت =====
    st.title("🕰️ Time Travel Chatbot")
    st.markdown("<p style='text-align:center; color:#666;'>سافر عبر العصور واستمع للحكايات 🏺</p>", unsafe_allow_html=True)

    @st.cache_resource
    def init_rag():
        """تهيئة نظام RAG"""
        with st.spinner("⏳ جاري تحميل قاعدة المعرفة..."):
            try:
                chatbot = MultiEraEgyptianRAG()
                
                kb_path = "knowledge_base"
                if os.path.exists(f"{kb_path}/config.pkl"):
                    chatbot.load_all_eras(kb_path)
                    st.success("✅ تم تحميل العصور بنجاح!")
                else:
                    st.error("⚠️ قاعدة المعرفة غير موجودة!")
                    st.info("📝 شغّل `python multi.py` أولاً لبناء قاعدة المعرفة")
                    st.stop()
                
                return chatbot
                
            except Exception as e:
                st.error(f"❌ خطأ في التهيئة: {str(e)}")
                st.stop()

    try:
        chatbot = init_rag()
    except Exception as e:
        st.error(f"❌ فشل تحميل النظام: {str(e)}")
        st.stop()

    st.markdown("---")
    
    # اختيار العصر
    eras = {
        "pharaonic": "🏛️ العصر الفرعوني",
        "greek": "🏺 العصر اليوناني",
        "roman": "🏟️ العصر الروماني",
        "medieval": "🕌 العصر الوسيط"
    }
    
    loaded_eras = chatbot.get_loaded_eras()
    available_eras = {k: v for k, v in eras.items() if k in loaded_eras}
    
    if not available_eras:
        st.error("❌ لا توجد عصور محملة!")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        era_choice = st.selectbox(
            "🌍 اختر العصر:",
            list(available_eras.keys()),
            format_func=lambda e: available_eras[e]
        )
    
    with col2:
        mode = st.radio(
            "🎭 نوع المحتوى:",
            ["معلومات", "حكايات"],
            horizontal=True
        )

    # حقل السؤال
    user_query = st.text_area(
        "💬 اكتب سؤالك هنا:",
        placeholder="مثال: احكيلي عن حياة الفراعنة وانتصاراتهم العظيمة",
        height=100
    )

    # أزرار التحكم
    col1, col2 = st.columns([3, 1])
    with col1:
        ask_button = st.button("🗣️ اسأل واستمع", use_container_width=True, type="primary")
    with col2:
        clear_button = st.button("🗑️ مسح", use_container_width=True)

    if clear_button:
        st.rerun()

    # معالجة السؤال
    if ask_button:
        if not user_query.strip():
            st.warning("⚠️ من فضلك اكتب سؤال أولاً.")
        else:
            with st.spinner("🤔 جاري استحضار الحكاية من أعماق التاريخ..."):
                try:
                    if chatbot.switch_era(era_choice):
                        # الحصول على الإجابة
                        result = chatbot.ask(user_query)
                        
                        # عرض الإجابة بتصميم احترافي
                        st.markdown("### 🎭 الحكاية:")
                        st.markdown(f"<div class='answer-box'><strong>{available_eras[era_choice]}</strong><br><br>{result}</div>", unsafe_allow_html=True)
                        
                        # توليد الصوت
                        st.markdown("### 🔊 استمع للحكاية:")
                        
                        try:
                            audio_path = text_to_speech(result, era_choice)
                            
                            if audio_path and audio_path.exists():
                                # قراءة الملف
                                with open(audio_path, 'rb') as audio_file:
                                    audio_bytes = audio_file.read()
                                
                                # عرض المشغل مع autoplay و JavaScript
                                import base64
                                audio_base64 = base64.b64encode(audio_bytes).decode()
                                
                                # HTML مع JavaScript للتشغيل الإجباري
                                audio_html = f"""
                                <div style="padding: 20px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 15px; margin: 10px 0;">
                                    <p style="color: #667eea; font-weight: bold; margin-bottom: 10px;">▶️ الصوت جاهز - اضغط Play للاستماع</p>
                                    <audio id="audioPlayer" controls style="width: 100%;">
                                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                                        متصفحك لا يدعم تشغيل الصوت.
                                    </audio>
                                </div>
                                <script>
                                    // محاولة التشغيل التلقائي
                                    setTimeout(function() {{
                                        var audio = document.getElementById('audioPlayer');
                                        if (audio) {{
                                            audio.play().catch(function(error) {{
                                                console.log("التشغيل التلقائي محظور. اضغط Play يدوياً.");
                                            }});
                                        }}
                                    }}, 500);
                                </script>
                                """
                                st.markdown(audio_html, unsafe_allow_html=True)
                                
                                # معلومة مساعدة
                                st.info("💡 إذا لم يبدأ الصوت تلقائياً، اضغط على زر ▶️ Play")
                                
                                # زر تحميل احتياطي
                                st.download_button(
                                    label="📥 تحميل الصوت",
                                    data=audio_bytes,
                                    file_name=f"story_{era_choice}.mp3",
                                    mime="audio/mp3"
                                )
                            else:
                                st.warning("⚠️ لم يتمكن من توليد الصوت")
                                st.info("💡 تأكد من اتصالك بالإنترنت (gTTS يحتاج إنترنت)")
                                
                        except Exception as e:
                            st.error(f"❌ خطأ: {str(e)}")
                            st.info("💡 يمكنك قراءة النص أعلاه")
                        
                        # معلومات إضافية
                        with st.expander("ℹ️ تفاصيل الرحلة"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.info(f"🌍 **العصر:** {available_eras[era_choice]}")
                            with col2:
                                st.info(f"🎭 **النوع:** {mode}")
                        
                    else:
                        st.error(f"❌ فشل الانتقال إلى {available_eras[era_choice]}")
                        
                except Exception as e:
                    st.error(f"❌ حدث خطأ: {str(e)}")
                    with st.expander("🔍 تفاصيل الخطأ"):
                        st.code(str(e))

    # الفوتر
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔙 عودة للشاشة الرئيسية", use_container_width=True):
            st.session_state.started = False
            st.rerun()

    # ===== الشريط الجانبي =====
    with st.sidebar:
        st.title("📚 مكتبة العصور")
        st.markdown("---")
        
        st.markdown("### 🌟 العصور المتاحة:")
        for era_key, era_name in available_eras.items():
            era_data = chatbot.era_data.get(era_key)
            if era_data and era_data['loaded']:
                num_docs = len(era_data.get('child_documents', []))
                with st.container():
                    st.markdown(f"**{era_name}**")
                    st.caption(f"📄 {num_docs} مستند متاح")
                    st.markdown("")
        
        st.markdown("---")
        st.markdown("### 💡 نصائح للاستخدام:")
        st.markdown("""
        - 🎯 اسأل أسئلة محددة للحصول على حكايات مفصلة
        - 🎭 اختر نوع "حكايات" لتجربة سرد درامية
        - 🔊 فعّل الصوت لتجربة غامرة
        - 🌍 استكشف جميع العصور المختلفة
        """)
        
        st.markdown("---")
        st.markdown("### 🎨 المميزات:")
        st.markdown("""
        - ✅ صوت عربي احترافي
        - ✅ تصميم عصري وجذاب
        - ✅ معلومات تاريخية دقيقة
        - ✅ تجربة تفاعلية ممتعة
        """)
        
        st.markdown("---")
        st.caption("Made with ❤️ using Streamlit & AI")