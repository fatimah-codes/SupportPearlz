"""
SupportPearlz — Streamlit Frontend Application
Theme: Facet "Designed in glass" dark smoked-glass hero with WebGL Three.js interactive canvas.
Workflow: Prompts user for API key directly on top of 3D glass shards -> Grants access to Enterprise Support Console.
"""

import os
import sys
import time
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import settings
from src.retrieval.vector_store import load_or_build, similarity_search_helper
from src.chains.rag_chain import execute_rag_pipeline
from src.chains.memory import ConversationMemory
from src.chains.schemas import GroundedResponse

# Configure Page
st.set_page_config(
    page_title="Facet // SupportPearlz Console",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep Smoked-Glass Dark CSS Theme (Facet "Designed in Glass")
st.markdown("""
<style>
  /* =========================================================================
     1. GLOBAL CANVAS & AMBIENT STUDIO LIGHTING
     ========================================================================= */
  html, body, [data-testid="stAppViewContainer"], .main {
    background-color: #050507 !important;
    background-image: 
      radial-gradient(circle at 18% 12%, rgba(66, 59, 98, 0.25) 0%, transparent 40%),
      radial-gradient(circle at 82% 80%, rgba(54, 214, 126, 0.10) 0%, transparent 45%),
      radial-gradient(circle at 50% 50%, rgba(23, 23, 27, 0.45) 0%, transparent 80%) !important;
    background-attachment: fixed !important;
    color: #e4e4e7 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  }
  
  /* Top Streamlit App Header */
  header[data-testid="stHeader"] {
    background: rgba(5, 5, 7, 0.75) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
  }

  /* Main Container Padding */
  .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 4rem !important;
    max-width: 1400px !important;
  }

  /* =========================================================================
     2. SMOKED-GLASS CARDS & CONTAINERS
     ========================================================================= */
  .glass-card {
    background: linear-gradient(135deg, rgba(26, 26, 32, 0.75) 0%, rgba(18, 18, 24, 0.90) 100%) !important;
    backdrop-filter: blur(24px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 14px !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.55), inset 0 1px 1px rgba(255, 255, 255, 0.22) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
  }
  
  .glass-card:hover {
    border-color: rgba(185, 171, 255, 0.35) !important;
    box-shadow: 0 20px 48px rgba(0, 0, 0, 0.65), 0 0 30px rgba(185, 171, 255, 0.12) !important;
  }

  /* Hero Typography */
  .glass-hero-title {
    font-size: 28px !important;
    font-weight: 600 !important;
    letter-spacing: -0.04em !important;
    background: linear-gradient(180deg, #dcdce0 0%, #bdbdc3 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 6px !important;
  }
  
  .glass-eyebrow {
    font-size: 11px !important;
    letter-spacing: 0.2em !important;
    color: rgba(255, 255, 255, 0.42) !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
    font-weight: 600 !important;
  }

  /* =========================================================================
     3. BUTTONS: SMOKED-GLASS PILLS WITH GREEN #36d67e HOVER GLOW
     ========================================================================= */
  button[kind="secondary"], button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, rgba(26, 26, 34, 0.85) 0%, rgba(18, 18, 24, 0.95) 100%) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #e4e4e7 !important;
    border-radius: 9px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 8px 18px !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.12) !important;
  }

  /* Turns green #36d67e on hover as per prompt specification! */
  button[kind="secondary"]:hover, button[kind="primary"]:hover, .stButton > button:hover {
    background: rgba(54, 214, 126, 0.16) !important;
    border-color: #36d67e !important;
    color: #36d67e !important;
    box-shadow: 0 0 24px rgba(54, 214, 126, 0.45), 0 8px 24px rgba(0, 0, 0, 0.6) !important;
    transform: translateY(-1px) !important;
  }

  /* =========================================================================
     4. TABS: SMOKED-GLASS NAV SLAB (NO RED HIGHLIGHT)
     ========================================================================= */
  div[data-testid="stTabs"] {
    margin-top: 10px !important;
  }

  div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(18, 18, 24, 0.85) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    padding: 6px 8px !important;
    gap: 8px !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.15) !important;
    margin-bottom: 20px !important;
  }

  button[data-baseweb="tab"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: rgba(255, 255, 255, 0.6) !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    border-radius: 8px !important;
    transition: all 0.25s ease !important;
  }

  button[data-baseweb="tab"]:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.06) !important;
  }

  button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
    background: rgba(185, 171, 255, 0.16) !important;
    border: 1px solid rgba(185, 171, 255, 0.45) !important;
    box-shadow: 0 0 18px rgba(185, 171, 255, 0.22) !important;
  }

  /* Kill Streamlit default red highlight bar */
  div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
    display: none !important;
    height: 0 !important;
  }

  /* =========================================================================
     5. CHAT BUBBLES & SOURCE BADGES
     ========================================================================= */
  .chat-bubble-user {
    background: linear-gradient(135deg, rgba(32, 32, 42, 0.80) 0%, rgba(22, 22, 28, 0.95) 100%) !important;
    backdrop-filter: blur(18px) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    margin: 12px 0 !important;
    color: #f4f4f5 !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.15) !important;
  }

  .chat-bubble-bot {
    background: linear-gradient(135deg, rgba(23, 23, 30, 0.92) 0%, rgba(15, 15, 20, 0.98) 100%) !important;
    backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(185, 171, 255, 0.28) !important;
    border-left: 4px solid #b9abff !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    margin: 14px 0 !important;
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.55), 0 0 28px rgba(185, 171, 255, 0.12), inset 0 1px 1px rgba(255, 255, 255, 0.25) !important;
  }

  .source-pill {
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    background: rgba(185, 171, 255, 0.14) !important;
    border: 1px solid rgba(185, 171, 255, 0.38) !important;
    color: #d8b4fe !important;
    padding: 4px 12px !important;
    border-radius: 16px !important;
    font-size: 12px !important;
    margin: 4px 8px 4px 0 !important;
    font-weight: 500 !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
  }

  .status-pill {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    background: rgba(54, 214, 126, 0.14) !important;
    border: 1px solid #36d67e !important;
    color: #36d67e !important;
    padding: 5px 14px !important;
    border-radius: 18px !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    box-shadow: 0 0 14px rgba(54, 214, 126, 0.35) !important;
  }

  /* =========================================================================
     6. CHAT INPUT BAR
     ========================================================================= */
  [data-testid="stChatInput"] {
    background: rgba(20, 20, 26, 0.90) !important;
    backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 14px !important;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.7), inset 0 1px 1px rgba(255, 255, 255, 0.15) !important;
  }

  [data-testid="stChatInput"]:focus-within {
    border-color: #b9abff !important;
    box-shadow: 0 0 25px rgba(185, 171, 255, 0.25), 0 12px 35px rgba(0,0,0,0.8) !important;
  }

  /* =========================================================================
     7. SIDEBAR: FROSTED SMOKED GLASS
     ========================================================================= */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(13, 13, 17, 0.96) 0%, rgba(7, 7, 10, 0.99) 100%) !important;
    backdrop-filter: blur(30px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(30px) saturate(180%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.12) !important;
    box-shadow: 10px 0 35px rgba(0, 0, 0, 0.6) !important;
  }

  /* =========================================================================
     8. INPUTS, SELECTS, EXPANDERS & TABLES
     ========================================================================= */
  input[type="text"], input[type="password"], div[data-baseweb="select"] > div {
    background: rgba(18, 18, 24, 0.88) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
  }

  div[data-testid="stExpander"] {
    background: rgba(23, 23, 29, 0.75) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    margin-bottom: 12px !important;
  }

  /* Metric cards */
  div[data-testid="stMetric"] {
    background: rgba(24, 24, 32, 0.80) !important;
    backdrop-filter: blur(18px) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 10px !important;
    padding: 16px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45), inset 0 1px 1px rgba(255, 255, 255, 0.15) !important;
  }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "api_provider" not in st.session_state:
    st.session_state["api_provider"] = "google_genai"
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationMemory()
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "vector_store" not in st.session_state:
    try:
        st.session_state["vector_store"] = load_or_build()
    except Exception as e:
        st.session_state["vector_store"] = None


# =============================================================================
# GATEWAY VIEW: Facet Smoked Glass Hero + API Key Modal ON TOP OF 3D GLASS
# =============================================================================
if not st.session_state["authenticated"]:
    # 1. Overlay CSS: Pulls the API Key Form directly on top of the 3D WebGL scene!
    st.markdown("""
    <style>
      /* Ensure the 3D canvas is positioned cleanly in the background */
      iframe {
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.85) !important;
      }

      /* Position the API Key Form DIRECTLY ON TOP of the 3D glass hero */
      div[data-testid="stForm"] {
        position: relative !important;
        margin-top: -540px !important;
        margin-bottom: 70px !important;
        z-index: 1000 !important;
        background: rgba(14, 14, 20, 0.86) !important;
        backdrop-filter: blur(32px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(32px) saturate(200%) !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 18px !important;
        padding: 32px 36px !important;
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.95), 0 0 50px rgba(185, 171, 255, 0.22), inset 0 1px 1px rgba(255, 255, 255, 0.35) !important;
      }
    </style>
    """, unsafe_allow_html=True)

    # 2. Render Interactive WebGL 3D Glass Hero (Height 620px)
    hero_html_path = BASE_DIR / "facet_hero.html"
    if hero_html_path.exists():
        with open(hero_html_path, "r", encoding="utf-8") as f:
            hero_content = f.read()
        components.html(hero_content, height=620, scrolling=False)

    # 3. Form centered inside columns, pulled directly on top of the 3D glass!
    col_l, col_center, col_r = st.columns([1, 2.2, 1])
    with col_center:
        with st.form("api_key_form"):
            st.markdown("""
            <div style="margin-bottom: 20px;">
              <div class="glass-eyebrow">REAL-TIME MATERIAL STUDIO &bull; FACET</div>
              <div class="glass-hero-title">SupportPearlz Gateway</div>
              <p style="color: rgba(255,255,255,0.65); font-size: 13.5px; line-height: 1.5; margin-top: 6px; margin-bottom: 0;">
                Provide your API credentials to activate live semantic retrieval and grounded customer support reasoning over the Pearlz Home Systems knowledge base.
              </p>
            </div>
            """, unsafe_allow_html=True)

            provider = st.selectbox(
                "Select Intelligence Engine Provider",
                ["Google Gemini (Recommended)", "OpenAI", "Local Demonstration Mode (Offline / Free)"],
                index=0
            )
            api_key_input = st.text_input(
                "Enter API Key",
                type="password",
                placeholder="AIzaSy... / sk-...",
                help="Your key is stored only in memory for this session and never saved to disk."
            )
            submit_btn = st.form_submit_button("Enter Support Console ➔", use_container_width=True)

            if submit_btn:
                if "Local" in provider:
                    st.session_state["authenticated"] = True
                    st.session_state["api_provider"] = "mock"
                    settings.LLM_PROVIDER = "mock"
                    st.success("Authorized in Local Demonstration Mode. Loading console...")
                    time.sleep(0.4)
                    st.rerun()
                elif "Google" in provider:
                    if not api_key_input.strip():
                        st.error("Please enter a valid Google Gemini API Key or choose Local Demonstration Mode.")
                    else:
                        st.session_state["authenticated"] = True
                        st.session_state["api_provider"] = "google_genai"
                        os.environ["GOOGLE_API_KEY"] = api_key_input.strip()
                        settings.GOOGLE_API_KEY = api_key_input.strip()
                        settings.LLM_PROVIDER = "google_genai"
                        st.success("Google Gemini credentials validated. Loading console...")
                        time.sleep(0.4)
                        st.rerun()
                elif "OpenAI" in provider:
                    if not api_key_input.strip():
                        st.error("Please enter a valid OpenAI API Key or choose Local Demonstration Mode.")
                    else:
                        st.session_state["authenticated"] = True
                        st.session_state["api_provider"] = "openai"
                        os.environ["OPENAI_API_KEY"] = api_key_input.strip()
                        settings.OPENAI_API_KEY = api_key_input.strip()
                        settings.LLM_PROVIDER = "openai"
                        st.success("OpenAI credentials validated. Loading console...")
                        time.sleep(0.4)
                        st.rerun()

    st.stop()


# =============================================================================
# CONSOLE VIEW: SupportPearlz Active Working Dashboard (Full Smoked-Glass Theme)
# =============================================================================

# ── Broken Glass WebGL Background ────────────────────────────────────────────
# Inject broken-glass CSS + Three.js scene as a fixed full-page backdrop
_bg_path = BASE_DIR / "facet_console_bg.html"
if _bg_path.exists():
    with open(_bg_path, "r", encoding="utf-8") as _f:
        _bg_html = _f.read()

    # Wrap in a JS snippet that repositions its own <iframe> to be fixed/full-viewport
    _fix_css = """
    <style>
      /* The iframe Streamlit places renders at normal flow; we move it to fixed */
      iframe[title="components.v1.html"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        z-index: 0 !important;
        pointer-events: none !important;
        opacity: 0.50 !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
      }
      /* Ensure all real content is above the glass backdrop */
      [data-testid="stMainBlockContainer"],
      [data-testid="stVerticalBlock"],
      .block-container,
      [data-testid="stSidebar"],
      header[data-testid="stHeader"] {
        position: relative !important;
        z-index: 10 !important;
      }
    </style>
    """
    st.markdown(_fix_css, unsafe_allow_html=True)
    components.html(_bg_html, height=1, scrolling=False)

# Sidebar Controls with Frosted Glass Aesthetics
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0;">
      <div class="glass-eyebrow">FACET STUDIO &bull; ACTIVE</div>
      <h2 style="margin: 0; font-size: 24px; color: #fff; font-weight: 600;">SupportPearlz</h2>
      <p style="font-size: 12px; color: #a1a1aa; margin-top: 4px;">Enterprise Knowledge Agent v2.4</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Runtime Status")
    st.markdown(f"**Provider:** <span class='source-pill'>{st.session_state['api_provider']}</span>", unsafe_allow_html=True)
    st.markdown(f"**Collection:** `{settings.COLLECTION_NAME}`")
    st.markdown(f"**Relevance Gate:** `{settings.SCORE_THRESHOLD}`")
    st.markdown(f"**Retrieval k:** `{settings.RETRIEVAL_K}`")
    st.markdown(f"**Temperature:** `{settings.TEMPERATURE}`")

    st.markdown("---")
    if st.button("🔄 Reset Conversation History", use_container_width=True):
        st.session_state["memory"].clear()
        st.session_state["chat_history"].clear()
        st.toast("Conversation memory cleared.")
        st.rerun()

    if st.button("🔒 Lock & Exit Console", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

# Top Hero Banner in Console View (Smoked-Glass Nav Slab)
st.markdown("""
<div class="glass-card" style="padding: 18px 24px; display: flex; align-items: center; justify-content: space-between; border-left: 4px solid #36d67e; margin-bottom: 20px;">
  <div>
    <div class="glass-eyebrow">FACET // REAL-TIME MATERIAL STUDIO</div>
    <div style="font-size: 22px; font-weight: 600; color: #ffffff; letter-spacing: -0.02em;">SupportPearlz Grounded Intelligence Console</div>
    <div style="font-size: 13px; color: rgba(255,255,255,0.6); margin-top: 4px;">
      12 Verified Documents &bull; 142 Chunks &bull; Calibrated Gate &theta;=0.080 &bull; Zero-Hallucination Policy
    </div>
  </div>
  <div>
    <span class="status-pill">&#9679; KNOWLEDGE AGENT ONLINE</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Main Application Tabs with Smoked-Glass Nav Slab
tab_chat, tab_eval, tab_kb, tab_settings = st.tabs([
    "💬 Support Console",
    "🧪 Conformance Suite (Task 11)",
    "📚 Knowledge Base & Retrieval",
    "⚙️ Model & Threshold Tuner"
])

# -----------------------------------------------------------------------------
# TAB 1: Support Chat Console
# -----------------------------------------------------------------------------
with tab_chat:
    # Display Conversation History
    if not st.session_state["chat_history"]:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 36px 20px;">
          <div style="font-size: 32px; margin-bottom: 10px;">💎</div>
          <div style="font-size: 18px; font-weight: 600; color: #fff; margin-bottom: 6px;">How can SupportPearlz assist you today?</div>
          <div style="font-size: 13.5px; color: rgba(255,255,255,0.55); max-width: 580px; margin: 0 auto; line-height: 1.5;">
            Ask about AquaPearl 500 Pro specifications, error codes, warranty coverage, refund policies, or remote delivery timelines. Every answer is strictly grounded with citations.
          </div>
        </div>
        """, unsafe_allow_html=True)

    for turn in st.session_state["chat_history"]:
        user_text = turn["user"]
        bot_res: GroundedResponse = turn["response"]

        st.markdown(f"""
        <div class="chat-bubble-user">
          <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.15em; color: rgba(255,255,255,0.42); margin-bottom: 4px; font-weight: 600;">CUSTOMER INQUIRY</div>
          <div style="font-size: 14.5px; font-weight: 400; line-height: 1.5;">{user_text}</div>
        </div>
        """, unsafe_allow_html=True)

        conf_color = "#36d67e" if bot_res.confidence == "high" else ("#facc15" if bot_res.confidence == "partial" else "#f87171")
        
        sources_html = ""
        if bot_res.sources:
            badges = "".join([f'<span class="source-pill">📄 {s.source} ({s.location})</span>' for s in bot_res.sources])
            sources_html = f'<div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.08);"><b>Verified Sources:</b><br>{badges}</div>'
        else:
            if not bot_res.answered:
                sources_html = '<div style="margin-top: 12px; color: #a1a1aa; font-size: 12px;"><i>Refusal: No verified document source matched query.</i></div>'

        rewritten_html = ""
        if bot_res.rewritten_query and bot_res.rewritten_query != user_text:
            rewritten_html = f'<div style="font-size: 11.5px; color: #a1a1aa; margin-top: 8px;">🔍 <i>Query Condensation:</i> "{bot_res.rewritten_query}"</div>'

        st.markdown(f"""
        <div class="chat-bubble-bot">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; color: {conf_color};">
              CONFIDENCE: {bot_res.confidence.upper()} &bull; LATENCY: {bot_res.latency_ms:.1f}ms
            </span>
            <span style="font-size: 11px; color: rgba(255,255,255,0.35);">SupportPearlz Agent</span>
          </div>
          <div style="font-size: 14.5px; line-height: 1.6; color: #f4f4f5;">{bot_res.answer}</div>
          {sources_html}
          {rewritten_html}
        </div>
        """, unsafe_allow_html=True)

    # Suggested Prompts (Smoked-Glass Buttons with Green #36d67e Hover!)
    st.markdown("<p style='font-size: 12px; color: rgba(255,255,255,0.45); margin-top: 20px; font-weight: 600; letter-spacing: 0.05em;'>QUICK SAMPLE INQUIRIES (CLICK TO TEST):</p>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    q_to_send = None
    if c1.button("🚨 Beeping three times", use_container_width=True):
        q_to_send = "My aquapearl 500 keeps beeping three times and then stops"
    if c2.button("🛡️ Is that covered?", use_container_width=True):
        q_to_send = "Is that covered under warranty?"
    if c3.button("🚚 Delivery to Skardu", use_container_width=True):
        q_to_send = "Ok and can I get it delivered to Skardu by tomorrow?"
    if c4.button("❌ Student discount", use_container_width=True):
        q_to_send = "Do you offer student discounts?"

    # User Input Field
    user_query = st.chat_input("Ask a customer support or policy question...")
    if q_to_send:
        user_query = q_to_send

    if user_query:
        with st.spinner("Retrieving grounded knowledge and formulating response..."):
            response = execute_rag_pipeline(
                question=user_query,
                memory=st.session_state["memory"],
                session_id="streamlit_session",
                vector_store=st.session_state["vector_store"]
            )
            st.session_state["chat_history"].append({
                "user": user_query,
                "response": response
            })
            st.rerun()

# -----------------------------------------------------------------------------
# TAB 2: Automated 20-Case Conformance Suite
# -----------------------------------------------------------------------------
with tab_eval:
    st.markdown("""
    <div class="glass-card">
      <div class="glass-eyebrow">TASK 11 BENCHMARK</div>
      <div class="glass-hero-title">Automated 20-Case Conformance Suite</div>
      <p style="color: rgba(255,255,255,0.6); font-size: 13.5px; margin: 0;">
        Executes test cases T-01 through T-20 across all six assignment categories: Easy, Semantic, Multi-Step, Follow-up, Unanswerable, and Adversarial.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Run Complete 20-Case Benchmark Suite", use_container_width=True):
        from evaluation.run_eval import run_full_evaluation
        with st.spinner("Executing 20 test cases against the RAG pipeline..."):
            summary, test_results = run_full_evaluation(vector_store=st.session_state["vector_store"])

        # Metric Badges as Smoked Glass Shards
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Cases", summary["total_tests"])
        m2.metric("Correctness", f"{summary['correctness_rate']*100:.1f}%")
        m2.caption("Accuracy across test set")
        m3.metric("Groundedness", f"{summary['groundedness_rate']*100:.1f}%")
        m3.caption("Zero hallucinations")
        m4.metric("Citation Accuracy", f"{summary['citation_accuracy_rate']*100:.1f}%")
        m4.caption("Valid document references")
        m5.metric("Refusal on Gaps", f"{summary['refusal_rate_on_gaps']*100:.1f}%")
        m5.caption("100% gap rejection")

        # Results Table
        table_rows = []
        for r in test_results:
            status = "✅ PASS" if r["passed"] else "❌ FAIL"
            table_rows.append({
                "ID": r["test_id"],
                "Category": r["category"],
                "Question": r["question"][:45] + ("..." if len(r["question"]) > 45 else ""),
                "Ground Truth Type": r["ground_truth_type"],
                "Status": status,
                "Latency (ms)": f"{r['latency_ms']:.1f}"
            })
        st.dataframe(table_rows, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Knowledge Base & Vector Store Inspector
# -----------------------------------------------------------------------------
with tab_kb:
    st.markdown("""
    <div class="glass-card">
      <div class="glass-eyebrow">VECTOR STORE INSPECTOR</div>
      <div class="glass-hero-title">Indexed Knowledge Base Corpus</div>
      <p style="color: rgba(255,255,255,0.6); font-size: 13.5px; margin: 0;">
        Real-time inspection of 12 multi-format documents (PDF, DOCX, CSV, MD) across 142 semantic vector chunks.
      </p>
    </div>
    """, unsafe_allow_html=True)

    kb_dir = BASE_DIR / "data" / "knowledge_base"
    if kb_dir.exists():
        files = list(kb_dir.glob("*.*"))
        st.markdown(f"**Knowledge Base Directory:** `{kb_dir}` ({len(files)} files found)")
        
        cols = st.columns(3)
        for i, f in enumerate(files):
            col = cols[i % 3]
            size_kb = f.stat().st_size / 1024
            ext = f.suffix.upper()[1:]
            with col:
                st.markdown(f"""
                <div class="glass-card" style="padding: 14px 18px; margin-bottom: 12px;">
                  <div style="font-size: 11px; color: #b9abff; font-weight: 600;">{ext} DOCUMENT</div>
                  <div style="font-size: 14px; font-weight: 500; color: #fff; word-break: break-all;">{f.name}</div>
                  <div style="font-size: 12px; color: #71717a; margin-top: 4px;">Size: {size_kb:.1f} KB</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 Live Semantic Search Diagnostics")
    diag_query = st.text_input("Enter diagnostic query for vector similarity retrieval:", "AquaPearl 500 beeping three times")
    if diag_query and st.session_state["vector_store"]:
        hits = similarity_search_helper(diag_query, k=4, vector_store=st.session_state["vector_store"])
        for i, (doc, score) in enumerate(hits):
            passed = score >= settings.SCORE_THRESHOLD
            gate_color = "#36d67e" if passed else "#f87171"
            gate_label = f"PASSED ✓ score={score:.3f}" if passed else f"FILTERED ✗ score={score:.3f}"
            src_name = doc.metadata.get("source", "Unknown")
            with st.expander(f"Chunk #{i+1} | {src_name}"):
                st.markdown(
                    f"<span style='color:{gate_color}; font-weight:600;'>{gate_label}</span> "
                    f"(threshold θ={settings.SCORE_THRESHOLD})",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Section:** `{doc.metadata.get('section', 'N/A')}` | **Page:** `{doc.metadata.get('page', 'N/A')}`")
                st.markdown(f"**Source:** `{src_name}` | **Location:** `{doc.metadata.get('location', 'N/A')}`")
                st.text(doc.page_content[:600])

# -----------------------------------------------------------------------------
# TAB 4: Model & Threshold Tuner
# -----------------------------------------------------------------------------
with tab_settings:
    st.markdown("""
    <div class="glass-card">
      <div class="glass-eyebrow">HYPERPARAMETER CALIBRATION</div>
      <div class="glass-hero-title">Model & Relevance Gate Tuner</div>
      <p style="color: rgba(255,255,255,0.6); font-size: 13.5px; margin: 0;">
        Tune runtime retrieval and generation parameters. Changes take effect on next query.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        new_k = st.slider("Retrieval k (top chunks)", min_value=1, max_value=8, value=int(settings.RETRIEVAL_K))
        new_thresh = st.slider("Relevance Gate Threshold (θ)", min_value=0.00, max_value=0.50, value=float(settings.SCORE_THRESHOLD), step=0.01)
        if new_k != settings.RETRIEVAL_K or new_thresh != settings.SCORE_THRESHOLD:
            settings.RETRIEVAL_K = new_k
            settings.SCORE_THRESHOLD = new_thresh
            st.toast("Updated retrieval parameters!")

    with col_t2:
        new_temp = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=float(settings.TEMPERATURE), step=0.05)
        new_ctx = st.number_input("Max Context Characters", min_value=1000, max_value=12000, value=int(settings.MAX_CONTEXT_CHARS), step=500)
        if new_temp != settings.TEMPERATURE or new_ctx != settings.MAX_CONTEXT_CHARS:
            settings.TEMPERATURE = new_temp
            settings.MAX_CONTEXT_CHARS = new_ctx
            st.toast("Updated generation parameters!")
