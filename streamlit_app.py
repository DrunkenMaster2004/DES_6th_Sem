import streamlit as st
from datetime import datetime
import sys
from pathlib import Path

# Load your backend modules
sys.path.append(str(Path(__file__).parent))
try:
    from agricultural_advisor_bot import AgriculturalAdvisorBot
except ImportError:
    st.error("⚠️ Missing dependency: agricultural_advisor_bot.py")

# --------------------------------------------------------
# 🌱 PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(
    page_title="AgriSense – Smart Farming Companion",
    page_icon="🌾",
    layout="wide"
)

# --------------------------------------------------------
# 🎨 CUSTOM STYLING — EARTHY, CALM, ORGANIC AESTHETIC
# --------------------------------------------------------
st.markdown("""
<style>
/* ---------------- GLOBAL THEME ---------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background: radial-gradient(circle at 20% 30%, #f0fdf4 0%, #fefce8 40%, #fef9c3 100%);
    font-family: 'Inter', sans-serif;
    color: #1e293b;
}

/* ---------------- CONTAINERS ---------------- */
.main-container {
    max-width: 1100px;
    margin: 2rem auto;
    padding: 2rem 3rem;
    border-radius: 30px;
    background: rgba(255,255,255,0.55);
    box-shadow: 0 20px 60px rgba(16, 185, 129, 0.15);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: all 0.4s ease;
}

.main-container:hover {
    box-shadow: 0 25px 70px rgba(22,163,74,0.25);
}

/* ---------------- HEADER ---------------- */
.header {
    text-align: center;
    margin-bottom: 2.5rem;
}

.header h1 {
    font-size: 3rem;
    font-weight: 700;
    color: #166534;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}

.header p {
    font-size: 1.1rem;
    color: #475569;
    margin-top: 0;
}

/* ---------------- FEATURE GRID ---------------- */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.feature-card {
    background: rgba(255,255,255,0.85);
    border: 1px solid #dcfce7;
    border-radius: 20px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.25s ease-in-out;
    box-shadow: 0 6px 15px rgba(0,0,0,0.05);
}

.feature-card:hover {
    transform: translateY(-6px);
    background: #ecfccb;
    box-shadow: 0 12px 20px rgba(0,0,0,0.08);
}

/* ---------------- CHAT SECTION ---------------- */
.chat-box {
    background: rgba(255,255,255,0.7);
    border-radius: 25px;
    padding: 1.5rem 2rem;
    box-shadow: 0 5px 25px rgba(0,0,0,0.1);
    margin-top: 2rem;
}

.message {
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    border-radius: 18px;
    line-height: 1.6;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    max-width: 80%;
}

.user-message {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    margin-left: auto;
    border-left: 5px solid #16a34a;
}

.bot-message {
    background: linear-gradient(135deg, #fef9c3, #fef08a);
    border-left: 5px solid #ca8a04;
}

/* ---------------- INPUT AREA ---------------- */
textarea {
    border-radius: 15px !important;
    border: 1.5px solid #d1fae5 !important;
    background: rgba(255, 255, 255, 0.7) !important;
    color: #1e293b !important;
    font-size: 1rem !important;
}

/* ---------------- BUTTONS ---------------- */
.stButton > button {
    background: linear-gradient(135deg, #16a34a 0%, #65a30d 100%);
    color: #fff;
    border: none;
    border-radius: 40px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: 0.2s ease-in-out;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #15803d, #4d7c0f);
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}

/* ---------------- STATUS BADGE ---------------- */
.status {
    padding: 0.4rem 0.9rem;
    border-radius: 15px;
    font-weight: 600;
    font-size: 0.9rem;
}

.online { background: #dcfce7; color: #166534; }
.offline { background: #fee2e2; color: #991b1b; }

/* ---------------- FOOTER ---------------- */
.footer {
    text-align: center;
    margin-top: 2.5rem;
    padding: 1rem;
    color: #475569;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# 🧠 STATE MANAGEMENT
# --------------------------------------------------------
if "bot" not in st.session_state:
    st.session_state.bot = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------------
# ⚙️ BOT SETUP
# --------------------------------------------------------
def initialize_bot():
    try:
        bot = AgriculturalAdvisorBot()
        bot.user_city = "Kanpur"
        bot.user_crop = "Wheat"
        bot.user_language = "English"
        bot.is_initialized = True
        st.session_state.bot = bot
        return True
    except Exception as e:
        st.error(f"Bot initialization failed: {e}")
        return False

def process_query(query):
    try:
        if not st.session_state.bot:
            return "⚠️ Please start the bot first."
        return st.session_state.bot.process_query(query)
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --------------------------------------------------------
# 🌿 MAIN APP
# --------------------------------------------------------
def main():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # HEADER
    st.markdown("""
    <div class="header">
        <h1>🌿 AgriSense</h1>
        <p>Smarter, Greener, and Kinder Agriculture — powered by AI.</p>
    </div>
    """, unsafe_allow_html=True)

    # FEATURES
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">💰 <b>Market Rates</b><br>Real-time mandi & wholesale price tracking.</div>
        <div class="feature-card">🌤️ <b>Weather Forecast</b><br>Local, crop-specific guidance.</div>
        <div class="feature-card">📜 <b>Policy Updates</b><br>Latest agricultural schemes explained.</div>
        <div class="feature-card">🌾 <b>Farming Insights</b><br>Tips for soil health and better yields.</div>
    </div>
    """, unsafe_allow_html=True)

    # CHAT SECTION
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("💬 Chat with AgriSense")
    with col2:
        if st.session_state.bot:
            st.markdown('<span class="status online">🟢 Online</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status offline">🔴 Offline</span>', unsafe_allow_html=True)
            if st.button("Start Bot 🌾"):
                if initialize_bot():
                    st.success("AgriSense is ready to help you 🌱")
                    st.rerun()

    query = st.text_area(
        "Ask about your crop, weather, or farming query:",
        placeholder="Example: 🌾 What is the best fertilizer for rice in humid weather?",
        height=120
    )

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("🚀 Send Query"):
            if not st.session_state.bot:
                st.error("Start the bot first!")
            elif query.strip():
                st.session_state.chat_history.append(
                    {"role": "user", "content": query, "time": datetime.now().strftime("%H:%M")}
                )
                with st.spinner("🤖 Thinking..."):
                    response = process_query(query)
                st.session_state.chat_history.append(
                    {"role": "bot", "content": response, "time": datetime.now().strftime("%H:%M")}
                )
                st.rerun()
    with colB:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    with colC:
        if st.button("💡 Suggestions"):
            st.info("Try:\n- Weather forecast for wheat in Lucknow\n- गेहूं का भाव क्या है?\n- PM Fasal Bima Yojana details\n- Best time to sow paddy")

    # DISPLAY CHAT
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='message user-message'><b>You:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='message bot-message'><b>AgriSense:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.info("Start chatting to get personalized farming insights 🌾")

    st.markdown("</div>", unsafe_allow_html=True)

    # FOOTER
    st.markdown("""
    <div class="footer">
        🌱 <b>AgriSense</b> | Empowering Sustainable Farming with AI · Bilingual (EN + HI) · Weather · Market · Policy
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
