# import streamlit as st
# from datetime import datetime
# import sys
# from pathlib import Path

# # Load your backend modules
# sys.path.append(str(Path(__file__).parent))
# try:
#     from agricultural_advisor_bot import AgriculturalAdvisorBot
# except ImportError:
#     st.error("⚠️ Missing dependency: agricultural_advisor_bot.py")

# # --------------------------------------------------------
# # 🌱 PAGE CONFIG
# # --------------------------------------------------------
# st.set_page_config(
#     page_title="AgriSense – Smart Farming Companion",
#     page_icon="🌾",
#     layout="wide"
# )

# # --------------------------------------------------------
# # 🎨 CUSTOM STYLING — EARTHY, CALM, ORGANIC AESTHETIC
# # --------------------------------------------------------
# st.markdown("""
# <style>
# /* ---------------- GLOBAL THEME ---------------- */
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

# .stApp {
#     background: radial-gradient(circle at 20% 30%, #f0fdf4 0%, #fefce8 40%, #fef9c3 100%);
#     font-family: 'Inter', sans-serif;
#     color: #1e293b;
# }

# /* ---------------- CONTAINERS ---------------- */
# .main-container {
#     max-width: 1100px;
#     margin: 2rem auto;
#     padding: 2rem 3rem;
#     border-radius: 30px;
#     background: rgba(255,255,255,0.55);
#     box-shadow: 0 20px 60px rgba(16, 185, 129, 0.15);
#     backdrop-filter: blur(20px);
#     -webkit-backdrop-filter: blur(20px);
#     transition: all 0.4s ease;
# }

# .main-container:hover {
#     box-shadow: 0 25px 70px rgba(22,163,74,0.25);
# }

# /* ---------------- HEADER ---------------- */
# .header {
#     text-align: center;
#     margin-bottom: 2.5rem;
# }

# .header h1 {
#     font-size: 3rem;
#     font-weight: 700;
#     color: #166534;
#     letter-spacing: -0.5px;
#     margin-bottom: 0.3rem;
# }

# .header p {
#     font-size: 1.1rem;
#     color: #475569;
#     margin-top: 0;
# }

# /* ---------------- FEATURE GRID ---------------- */
# .feature-grid {
#     display: grid;
#     grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
#     gap: 1rem;
#     margin: 2rem 0;
# }

# .feature-card {
#     background: rgba(255,255,255,0.85);
#     border: 1px solid #dcfce7;
#     border-radius: 20px;
#     padding: 1.2rem;
#     text-align: center;
#     transition: all 0.25s ease-in-out;
#     box-shadow: 0 6px 15px rgba(0,0,0,0.05);
# }

# .feature-card:hover {
#     transform: translateY(-6px);
#     background: #ecfccb;
#     box-shadow: 0 12px 20px rgba(0,0,0,0.08);
# }

# /* ---------------- CHAT SECTION ---------------- */
# .chat-box {
#     background: rgba(255,255,255,0.7);
#     border-radius: 25px;
#     padding: 1.5rem 2rem;
#     box-shadow: 0 5px 25px rgba(0,0,0,0.1);
#     margin-top: 2rem;
# }

# .message {
#     padding: 1rem 1.5rem;
#     margin: 1rem 0;
#     border-radius: 18px;
#     line-height: 1.6;
#     box-shadow: 0 4px 12px rgba(0,0,0,0.05);
#     max-width: 80%;
# }

# .user-message {
#     background: linear-gradient(135deg, #dcfce7, #bbf7d0);
#     margin-left: auto;
#     border-left: 5px solid #16a34a;
# }

# .bot-message {
#     background: linear-gradient(135deg, #fef9c3, #fef08a);
#     border-left: 5px solid #ca8a04;
# }

# /* ---------------- INPUT AREA ---------------- */
# textarea {
#     border-radius: 15px !important;
#     border: 1.5px solid #d1fae5 !important;
#     background: rgba(255, 255, 255, 0.7) !important;
#     color: #1e293b !important;
#     font-size: 1rem !important;
# }

# /* ---------------- BUTTONS ---------------- */
# .stButton > button {
#     background: linear-gradient(135deg, #16a34a 0%, #65a30d 100%);
#     color: #fff;
#     border: none;
#     border-radius: 40px;
#     padding: 0.6rem 2rem;
#     font-weight: 600;
#     font-size: 1rem;
#     transition: 0.2s ease-in-out;
# }

# .stButton > button:hover {
#     background: linear-gradient(135deg, #15803d, #4d7c0f);
#     transform: translateY(-2px);
#     box-shadow: 0 6px 12px rgba(0,0,0,0.15);
# }

# /* ---------------- STATUS BADGE ---------------- */
# .status {
#     padding: 0.4rem 0.9rem;
#     border-radius: 15px;
#     font-weight: 600;
#     font-size: 0.9rem;
# }

# .online { background: #dcfce7; color: #166534; }
# .offline { background: #fee2e2; color: #991b1b; }

# /* ---------------- FOOTER ---------------- */
# .footer {
#     text-align: center;
#     margin-top: 2.5rem;
#     padding: 1rem;
#     color: #475569;
#     font-size: 0.9rem;
# }
# </style>
# """, unsafe_allow_html=True)

# # --------------------------------------------------------
# # 🧠 STATE MANAGEMENT
# # --------------------------------------------------------
# if "bot" not in st.session_state:
#     st.session_state.bot = None
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # --------------------------------------------------------
# # ⚙️ BOT SETUP
# # --------------------------------------------------------
# def initialize_bot():
#     try:
#         bot = AgriculturalAdvisorBot()
#         bot.user_city = "Kanpur"
#         bot.user_crop = "Wheat"
#         bot.user_language = "English"
#         bot.is_initialized = True
#         st.session_state.bot = bot
#         return True
#     except Exception as e:
#         st.error(f"Bot initialization failed: {e}")
#         return False

# def process_query(query):
#     try:
#         if not st.session_state.bot:
#             return "⚠️ Please start the bot first."
#         return st.session_state.bot.process_query(query)
#     except Exception as e:
#         return f"⚠️ Error: {str(e)}"

# # --------------------------------------------------------
# # 🌿 MAIN APP
# # --------------------------------------------------------
# def main():
#     st.markdown('<div class="main-container">', unsafe_allow_html=True)

#     # HEADER
#     st.markdown("""
#     <div class="header">
#         <h1>🌿 AgriSense</h1>
#         <p>Smarter, Greener, and Kinder Agriculture — powered by AI.</p>
#     </div>
#     """, unsafe_allow_html=True)

#     # FEATURES
#     st.markdown("""
#     <div class="feature-grid">
#         <div class="feature-card">💰 <b>Market Rates</b><br>Real-time mandi & wholesale price tracking.</div>
#         <div class="feature-card">🌤️ <b>Weather Forecast</b><br>Local, crop-specific guidance.</div>
#         <div class="feature-card">📜 <b>Policy Updates</b><br>Latest agricultural schemes explained.</div>
#         <div class="feature-card">🌾 <b>Farming Insights</b><br>Tips for soil health and better yields.</div>
#     </div>
#     """, unsafe_allow_html=True)

#     # CHAT SECTION
#     st.markdown('<div class="chat-box">', unsafe_allow_html=True)
#     col1, col2 = st.columns([3, 1])

#     with col1:
#         st.subheader("💬 Chat with AgriSense")
#     with col2:
#         if st.session_state.bot:
#             st.markdown('<span class="status online">🟢 Online</span>', unsafe_allow_html=True)
#         else:
#             st.markdown('<span class="status offline">🔴 Offline</span>', unsafe_allow_html=True)
#             if st.button("Start Bot 🌾"):
#                 if initialize_bot():
#                     st.success("AgriSense is ready to help you 🌱")
#                     st.rerun()

#     query = st.text_area(
#         "Ask about your crop, weather, or farming query:",
#         placeholder="Example: 🌾 What is the best fertilizer for rice in humid weather?",
#         height=120
#     )

#     colA, colB, colC = st.columns(3)
#     with colA:
#         if st.button("🚀 Send Query"):
#             if not st.session_state.bot:
#                 st.error("Start the bot first!")
#             elif query.strip():
#                 st.session_state.chat_history.append(
#                     {"role": "user", "content": query, "time": datetime.now().strftime("%H:%M")}
#                 )
#                 with st.spinner("🤖 Thinking..."):
#                     response = process_query(query)
#                 st.session_state.chat_history.append(
#                     {"role": "bot", "content": response, "time": datetime.now().strftime("%H:%M")}
#                 )
#                 st.rerun()
#     with colB:
#         if st.button("🧹 Clear Chat"):
#             st.session_state.chat_history = []
#             st.rerun()
#     with colC:
#         if st.button("💡 Suggestions"):
#             st.info("Try:\n- Weather forecast for wheat in Lucknow\n- गेहूं का भाव क्या है?\n- PM Fasal Bima Yojana details\n- Best time to sow paddy")

#     # DISPLAY CHAT
#     if st.session_state.chat_history:
#         for msg in st.session_state.chat_history:
#             if msg["role"] == "user":
#                 st.markdown(f"<div class='message user-message'><b>You:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
#             else:
#                 st.markdown(f"<div class='message bot-message'><b>AgriSense:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
#     else:
#         st.info("Start chatting to get personalized farming insights 🌾")

#     st.markdown("</div>", unsafe_allow_html=True)

#     # FOOTER
#     st.markdown("""
#     <div class="footer">
#         🌱 <b>AgriSense</b> | Empowering Sustainable Farming with AI · Bilingual (EN + HI) · Weather · Market · Policy
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("</div>", unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

try:
    from agricultural_advisor_bot import AgriculturalAdvisorBot
    from weather_service import WeatherService
    from improved_policy_chatbot import ImprovedPolicyChatbot
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Please ensure all required files are in the same directory")

# Page configuration
st.set_page_config(
    page_title="AGRIBOT - Smart Farming Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS styling with beautiful aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Glassmorphism Header */
    .header-section {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 30px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        position: relative;
        overflow: hidden;
    }
    
    .header-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .header-section h1 {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .header-section p {
        font-size: 1.3rem;
        color: #555;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Feature Cards with Hover Effects */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 45px 0 rgba(102, 126, 234, 0.4);
    }
    
    .feature-card:hover::before {
        opacity: 1;
    }
    
    .feature-card h4 {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        color: #667eea;
        position: relative;
        z-index: 1;
    }
    
    .feature-card p {
        color: #666;
        font-size: 1rem;
        position: relative;
        z-index: 1;
    }
    
    /* Chat Container */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 2rem;
        min-height: 600px;
    }
    
    /* Messages with Modern Design */
    .message {
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        border-radius: 20px;
        max-width: 75%;
        animation: slideIn 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-bottom-left-radius: 5px;
    }
    
    .message strong {
        display: block;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Input Area */
    .input-area {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-size: 0.95rem;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    
    .status-online {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.4);
    }
    
    .status-offline {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(235, 51, 73, 0.4);
    }
    
    /* Text Area */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Info Section */
    .info-section {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .info-section h3 {
        color: #667eea;
        font-size: 1.8rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .info-section p {
        color: #555;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    }
    
    .footer strong {
        font-size: 1.2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .footer p {
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_bot():
    """Initialize the agricultural advisor bot with default settings"""
    try:
        if st.session_state.bot is None:
            bot = AgriculturalAdvisorBot()
            bot.user_city = "Kanpur"
            bot.user_crop = "Wheat"
            bot.user_language = "English"
            bot.is_initialized = True
            st.session_state.bot = bot
        return True
    except Exception as e:
        st.error(f"Error initializing bot: {e}")
        return False

def process_query_with_fallback(query: str) -> str:
    """Process query with fallback handling for setup issues"""
    try:
        if not st.session_state.bot:
            return "❌ Bot not initialized. Please start the bot first."
        
        response = st.session_state.bot.process_query(query)
        
        if "Setup Required" in response or "complete the initial setup" in response:
            query_lower = query.lower()
            
            cities = ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune', 'kanpur', 'lucknow', 'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik']
            for city in cities:
                if city in query_lower:
                    st.session_state.bot.user_city = city.title()
                    break
            
            crops = ['wheat', 'rice', 'corn', 'maize', 'sugarcane', 'cotton', 'pulses', 'oilseeds', 'vegetables', 'fruits', 'tomato', 'potato', 'onion', 'chilli', 'turmeric', 'ginger']
            for crop in crops:
                if crop in query_lower:
                    st.session_state.bot.user_crop = crop.title()
                    break
            
            st.session_state.bot.is_initialized = True
            response = st.session_state.bot.process_query(query)
        
        return response
        
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"

def main():
    # Header section
    st.markdown("""
    <div class="header-section">
        <h1>🌾 AGRIBOT</h1>
        <p>Your Smart Farming Partner for Quicker, Sharper, and Greener Decisions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Info section with feature cards
    st.markdown("""
    <div class="info-section">
        <h3>🤖 Intelligent Agricultural Assistant</h3>
        <p>Empowering farmers with AI-driven insights for sustainable and profitable farming</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <h4>💰 Live Mandi Prices</h4>
            <p>Real-time crop prices from mandis across India</p>
        </div>
        <div class="feature-card">
            <h4>🌤️ Weather Intelligence</h4>
            <p>Smart farming advice based on weather forecasts</p>
        </div>
        <div class="feature-card">
            <h4>📋 Policy Navigator</h4>
            <p>Complete guide to government schemes & subsidies</p>
        </div>
        <div class="feature-card">
            <h4>🌾 Expert Tips</h4>
            <p>Best practices & modern farming techniques</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-section">
        <p>💡 <strong>Multilingual Support:</strong> Ask questions in English or Hindi - the bot understands both! Simply mention your city or crop in your question.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Bot status
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown("### 💬 Chat with Your Agricultural Advisor")
    
    with col2:
        if st.session_state.bot:
            st.markdown('<span class="status-badge status-online">🟢 Online</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-offline">🔴 Offline</span>', unsafe_allow_html=True)
    
    if not st.session_state.bot:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start AGRIBOT", type="primary"):
                if initialize_bot():
                    st.success("✅ Bot started successfully!")
                    st.rerun()
    
    # Chat input area
    if st.session_state.bot:
        st.markdown('<div class="input-area">', unsafe_allow_html=True)
        
        user_input = st.text_area(
            "Type your question here:",
            placeholder="💬 Examples:\n• गेहूं का भाव क्या है?\n• What is the weather like in Mumbai?\n• PM Kisan scheme details\n• Rice prices in Kanpur\n• मौसम कैसा है?\n• Wheat prices in Delhi",
            height=120,
            key="user_input"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📤 Send Message", type="primary"):
                if user_input.strip():
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    
                    with st.spinner("🤖 Thinking..."):
                        try:
                            response = process_query_with_fallback(user_input)
                            st.session_state.chat_history.append({
                                "role": "bot",
                                "content": response,
                                "timestamp": datetime.now().strftime("%H:%M")
                            })
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            st.session_state.chat_history.append({
                                "role": "bot",
                                "content": error_msg,
                                "timestamp": datetime.now().strftime("%H:%M")
                            })
                    
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col3:
            if st.button("💡 Examples"):
                st.info("""
                **Try asking:**
                - गेहूं का भाव क्या है?
                - Weather forecast Mumbai
                - PM Kisan scheme details
                - Rice prices Kanpur
                - मौसम कैसा है?
                - Best planting season
                """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat history with modern message bubbles
    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="message user-message">
                    <strong>You • {message['timestamp']}</strong>
                    <div>{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message bot-message">
                    <strong>AGRIBOT • {message['timestamp']}</strong>
                    <div>{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        if st.session_state.bot:
            st.info("💬 Start a conversation by typing your question above!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>🌾 AGRIBOT - Smart Agricultural Advisory System</strong></p>
        <p>Powered by AI & Real-time Agricultural Data | Multilingual Support | 24/7 Assistance</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">Helping farmers make data-driven decisions for better yields and sustainable farming</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()