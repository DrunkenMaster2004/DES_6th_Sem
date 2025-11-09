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
    page_title="AGRISENSE - AI Farming Revolution",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ultra-modern CSS with stunning visuals
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1a472a, #2d5016);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Floating particles effect */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle, rgba(76, 175, 80, 0.1) 1px, transparent 1px),
            radial-gradient(circle, rgba(139, 195, 74, 0.1) 1px, transparent 1px);
        background-size: 50px 50px, 80px 80px;
        background-position: 0 0, 40px 40px;
        animation: particleFloat 20s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes particleFloat {
        0% { transform: translateY(0); }
        100% { transform: translateY(-50px); }
    }
    
    /* Ensure content is visible */
    .block-container {
        position: relative;
        z-index: 1;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hero Header with 3D effect */
    .hero-header {
        background: linear-gradient(135deg, rgba(20, 30, 48, 0.95) 0%, rgba(36, 59, 85, 0.95) 100%);
        backdrop-filter: blur(30px);
        border-radius: 40px;
        padding: 4rem 3rem;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.5),
            0 0 100px rgba(76, 175, 80, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(76, 175, 80, 0.15) 0%, transparent 60%);
        animation: heroGlow 6s ease-in-out infinite;
    }
    
    @keyframes heroGlow {
        0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.5; }
        50% { transform: translate(10%, 10%) scale(1.1); opacity: 0.8; }
    }
    
    .hero-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a8ff78 0%, #78ffd6 50%, #4ade80 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
        letter-spacing: -2px;
        animation: titleFloat 3s ease-in-out infinite;
    }
    
    @keyframes titleFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
        position: relative;
        z-index: 1;
        letter-spacing: 1px;
    }
    
    .hero-icons {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-top: 2rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-icon {
        font-size: 3rem;
        animation: iconBounce 2s ease-in-out infinite;
    }
    
    .hero-icon:nth-child(1) { animation-delay: 0s; }
    .hero-icon:nth-child(2) { animation-delay: 0.3s; }
    .hero-icon:nth-child(3) { animation-delay: 0.6s; }
    .hero-icon:nth-child(4) { animation-delay: 0.9s; }
    
    @keyframes iconBounce {
        0%, 100% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-15px) scale(1.1); }
    }
    
    /* Feature Cards with stunning hover effects */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(30, 40, 60, 0.9) 0%, rgba(40, 55, 75, 0.9) 100%);
        backdrop-filter: blur(20px);
        padding: 3rem 2rem;
        border-radius: 30px;
        text-align: center;
        transition: all 0.5s cubic-bezier(0.23, 1, 0.32, 1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(76, 175, 80, 0.2) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.5s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-20px);
        box-shadow: 
            0 30px 80px rgba(76, 175, 80, 0.4),
            0 0 60px rgba(76, 175, 80, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border-color: rgba(76, 175, 80, 0.5);
    }
    
    .feature-card:hover::before {
        opacity: 1;
    }
    
    .feature-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        filter: drop-shadow(0 0 20px rgba(76, 175, 80, 0.6));
        transition: all 0.3s ease;
    }
    
    .feature-card:hover .feature-icon {
        transform: scale(1.2);
        filter: drop-shadow(0 0 30px rgba(76, 175, 80, 1));
    }
    
    .feature-card h4 {
        font-size: 1.6rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #a8ff78 0%, #78ffd6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    .feature-card p {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Ultra-modern chat container */
    .chat-container {
        background: linear-gradient(135deg, rgba(20, 30, 48, 0.95) 0%, rgba(36, 59, 85, 0.95) 100%);
        backdrop-filter: blur(30px);
        border-radius: 40px;
        padding: 3rem;
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.5),
            0 0 100px rgba(76, 175, 80, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 3rem;
        min-height: 600px;
    }
    
    .chat-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a8ff78 0%, #78ffd6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Futuristic messages */
    .message {
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        border-radius: 25px;
        max-width: 70%;
        animation: messageSlide 0.5s cubic-bezier(0.23, 1, 0.32, 1);
        position: relative;
    }
    
    @keyframes messageSlide {
        from {
            opacity: 0;
            transform: translateY(30px);
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
        border-bottom-right-radius: 8px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-bottom-left-radius: 8px;
        box-shadow: 0 10px 30px rgba(56, 239, 125, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .message strong {
        display: block;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
        opacity: 0.95;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Futuristic input area */
    .input-area {
        background: linear-gradient(135deg, rgba(30, 40, 60, 0.8) 0%, rgba(40, 55, 75, 0.8) 100%);
        backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 30px;
        border: 2px solid rgba(76, 175, 80, 0.3);
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .input-area:hover {
        border-color: rgba(76, 175, 80, 0.6);
        box-shadow: 0 15px 50px rgba(76, 175, 80, 0.3);
    }
    
    /* Premium buttons */
    .stButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 1rem 3rem !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1) !important;
        box-shadow: 0 10px 30px rgba(56, 239, 125, 0.4) !important;
        width: 100% !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) scale(1.02) !important;
        box-shadow: 0 20px 50px rgba(56, 239, 125, 0.6) !important;
    }
    
    /* Glowing status badge */
    .status-badge {
        display: inline-block;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .status-online {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(56, 239, 125, 0.5);
        animation: statusPulse 2s ease-in-out infinite;
    }
    
    @keyframes statusPulse {
        0%, 100% { 
            box-shadow: 0 10px 30px rgba(56, 239, 125, 0.5);
        }
        50% { 
            box-shadow: 0 15px 40px rgba(56, 239, 125, 0.7);
        }
    }
    
    .status-offline {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(235, 51, 73, 0.5);
    }
    
    /* Premium text area */
    .stTextArea textarea {
        background: rgba(20, 30, 48, 0.6) !important;
        border-radius: 20px !important;
        border: 2px solid rgba(76, 175, 80, 0.3) !important;
        padding: 1.5rem !important;
        font-size: 1.1rem !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: rgba(76, 175, 80, 0.8) !important;
        box-shadow: 0 0 0 4px rgba(76, 175, 80, 0.2) !important;
        background: rgba(20, 30, 48, 0.8) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .stTextArea label {
        color: white !important;
    }
    
    /* Info section */
    .info-section {
        background: linear-gradient(135deg, rgba(30, 40, 60, 0.9) 0%, rgba(40, 55, 75, 0.9) 100%);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        padding: 2.5rem;
        margin-bottom: 3rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .info-section h3 {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, #a8ff78 0%, #78ffd6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .info-section p {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1.1rem;
        line-height: 1.8;
    }
    
    /* Stunning footer */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding: 3rem;
        background: linear-gradient(135deg, rgba(20, 30, 48, 0.95) 0%, rgba(36, 59, 85, 0.95) 100%);
        backdrop-filter: blur(30px);
        border-radius: 40px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .footer strong {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        background: linear-gradient(135deg, #a8ff78 0%, #78ffd6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    .footer p {
        color: rgba(255, 255, 255, 0.8);
        margin-top: 1rem;
        font-size: 1.05rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(20, 30, 48, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 10px;
        border: 2px solid rgba(20, 30, 48, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-header h1 {
            font-size: 3rem;
        }
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
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
    # Hero header with stunning design
    st.markdown("""
    <div class="hero-header">
        <h1>🌾 AGRISENSE</h1>
        <p class="hero-subtitle">Next-Generation AI-Powered Agricultural Intelligence Platform</p>
        <div class="hero-icons">
            <span class="hero-icon">🌱</span>
            <span class="hero-icon">🤖</span>
            <span class="hero-icon">📊</span>
            <span class="hero-icon">🌍</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Info section
    st.markdown("""
    <div class="info-section">
        <h3>🚀 Revolutionary Farming Intelligence</h3>
        <p>Harness the power of artificial intelligence to transform your agricultural operations. Get real-time insights, market intelligence, and expert recommendations at your fingertips.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards with stunning visuals
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">💰</div>
            <h4>Live Market Intelligence</h4>
            <p>Real-time mandi prices and market trends from across India with predictive analytics</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🌤️</div>
            <h4>Climate-Smart Advisory</h4>
            <p>AI-powered weather forecasts and precision farming recommendations</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📋</div>
            <h4>Policy Intelligence Hub</h4>
            <p>Comprehensive guide to government schemes, subsidies, and financial assistance</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h4>Expert Insights</h4>
            <p>Data-driven farming techniques and best practices for maximum yield</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-section">
        <p>💡 <strong>Intelligent Multilingual Support:</strong> Communicate naturally in English or Hindi. Our AI understands context and automatically adapts to your language preference.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Chat header
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="chat-title">💬 AI Assistant</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.bot:
            st.markdown('<span class="status-badge status-online">● Live</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-offline">○ Offline</span>', unsafe_allow_html=True)
    
    if not st.session_state.bot:
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("⚡ ACTIVATE AGRIBOT", type="primary"):
                if initialize_bot():
                    st.success("✅ AI System Online!")
                    st.rerun()
    
    # Chat interface
    if st.session_state.bot:
        st.markdown('<div class="input-area">', unsafe_allow_html=True)
        
        user_input = st.text_area(
            "Your Message",
            placeholder="🎤 Ask anything...\n\n• गेहूं की कीमत क्या है?\n• Weather forecast for Mumbai\n• PM Kisan scheme benefits\n• Best irrigation practices\n• Rice market trends",
            height=140,
            key="user_input"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 SEND", type="primary"):
                if user_input.strip():
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    
                    with st.spinner("🧠 AI Processing..."):
                        try:
                            response = process_query_with_fallback(user_input)
                            st.session_state.chat_history.append({
                                "role": "bot",
                                "content": response,
                                "timestamp": datetime.now().strftime("%H:%M")
                            })
                        except Exception as e:
                            error_msg = f"⚠️ System Error: {str(e)}"
                            st.session_state.chat_history.append({
                                "role": "bot",
                                "content": error_msg,
                                "timestamp": datetime.now().strftime("%H:%M")
                            })
                    
                    st.rerun()
        
        with col2:
            if st.button("🗑️ CLEAR", type="primary"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col3:
            if st.button("💡 EXAMPLES", type="primary"):
                st.info("""
                **Try these queries:**
                - गेहूं का भाव क्या है?
                - Weather forecast Mumbai
                - PM Kisan eligibility
                - Rice market analysis
                - Crop rotation tips
                - मौसम की जानकारी
                """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="message user-message">
                    <strong>YOU • {message['timestamp']}</strong>
                    <div>{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message bot-message">
                    <strong>AGRIBOT AI • {message['timestamp']}</strong>
                    <div>{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        if st.session_state.bot:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.6);">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
                <div style="font-size: 1.3rem; font-weight: 500;">AI Ready to Assist</div>
                <div style="font-size: 1rem; margin-top: 0.5rem;">Type your agricultural query above to get started</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Premium footer
    st.markdown("""
    <div class="footer">
        <p><strong>🌾 AGRIBOT</strong></p>
        <p>AI-Powered Agricultural Intelligence Platform</p>
        <p style="font-size: 0.95rem; margin-top: 1.5rem; opacity: 0.8;">
            🌍 Multilingual Support | 📊 Real-Time Data | 🤖 Advanced AI | 🔒 Secure & Private
        </p>
        <p style="font-size: 0.9rem; margin-top: 1rem; opacity: 0.7;">
            Empowering farmers with cutting-edge technology for sustainable and profitable agriculture
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()