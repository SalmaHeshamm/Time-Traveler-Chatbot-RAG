import streamlit as st
from dotenv import load_dotenv
import os
from rag_engine import RAGEngine
from utils.voice_generator import text_to_speech_ar
from utils.prompts import get_prompt_by_persona

load_dotenv()

st.set_page_config(page_title="Time Traveler Chatbot", page_icon="🕰", layout="wide")
st.title("🕰️ Time Traveler Chatbot - LangChain Demo (لهجة مصرية)")
st.write("اختار عصر، من ثم نوع المحتوى (معلومات أو حكايات)، واختر الشخصية. إذا كان OpenAI مفعلًا سيستخدم GPT لتحسين النص.")

rag = RAGEngine(data_dir="data")

st.sidebar.title("خيارات")
use_openai = os.getenv("OPENAI_API_KEY") not in (None, "", "None")
st.sidebar.write(f"OpenAI enabled: {use_openai}")

era = st.sidebar.selectbox("🌍 اختر العصر:", ["pharaohs", "roman", "abbasid", "medieval"])
mode = st.sidebar.radio("نوع المحتوى:", ("معلومات", "حكايات"))

if "conversation" not in st.session_state:
    st.session_state.conversation = []

st.header("ابدأ المحادثة")

if mode == "حكايات":
    persona = st.selectbox("🎭 اختر الشخصية:", ["ruler", "farmer", "knight", "merchant"])
    if st.button("اسمع حكاية"):
        story = rag.get_story(era, persona)
        role_prompt = get_prompt_by_persona(persona)
        reply_text = role_prompt + "\n\n" + story
        st.session_state.conversation.append(("assistant", reply_text))
        audio = text_to_speech_ar(reply_text)
        st.audio(audio, format="audio/mp3")
else:
    question = st.text_input("💬 اكتب سؤالك:", placeholder="مثال: كيف كان نظام الحكم؟")
    if st.button("اسأل"):
        if not question.strip():
            st.warning("اكتب سؤال أولًا.")
        else:
            answer = rag.get_answer(era, question)
            if answer is None or answer.strip() == "":
                st.info("❌ ليست موجودة الآن في معلوماتي عن هذا العصر.")
                st.session_state.conversation.append(("assistant", "ليست موجودة الآن في معلوماتي عن هذا العصر."))
            else:
                role_prompt = get_prompt_by_persona("scholar")
                final = role_prompt + "\n\n" + answer
                st.session_state.conversation.append(("assistant", final))
                st.write(final)
                audio = text_to_speech_ar(final)
                st.audio(audio, format="audio/mp3")

st.markdown("----")
st.subheader("سجل المحادثة")
for role, text in st.session_state.conversation[-10:]:
    if role == "assistant":
        st.markdown(f"**💬 بوت:** {text}")
    else:
        st.markdown(f"**🧑 المستخدم:** {text}")
