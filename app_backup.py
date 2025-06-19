"""
Mental Health Support Chatbot - Compassionate AI Assistant for Mental Wellness

Run with: streamlit run app.py --server.fileWatcherType none
"""

import streamlit as st
import os
import warnings
from dotenv import load_dotenv
import logging
import requests
import json
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Optimized compatibility setup
os.environ.update({
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1"
})
warnings.filterwarnings("ignore")

# Import the mental health agent
try:
    from tutor_agent import (
        setup_mental_health_agent, 
        setup_web_search_tool, 
        detect_crisis_situation, 
        CRISIS_RESOURCES, 
        get_mental_health_disclaimer
    )
    REAL_AGENT_AVAILABLE = True
    logger.info("✅ Mental health agent available")
except ImportError as e:
    REAL_AGENT_AVAILABLE = False
    logger.warning(f"⚠️ Real agent not available, using mock: {e}")

# Fallback Mock Mental Health Agent
class MockMentalHealthAgent:
    """Fallback mock mental health agent that provides basic mental health support"""
    
    def __init__(self):
        self.knowledge_base = {
            "anxiety": {
                "answer": "Anxiety is very common and manageable. Some effective strategies include: deep breathing exercises (try the 4-7-8 technique), mindfulness meditation, regular physical activity, maintaining a consistent sleep schedule, limiting caffeine, and practicing grounding techniques like the 5-4-3-2-1 method. If anxiety persists or interferes with daily life, consider speaking with a mental health professional.",
                "sources": ["Anxiety Management Guidelines", "Mental Health Resources"]
            },
            "depression": {
                "answer": "Depression is a serious but treatable condition. Some strategies that may help include: maintaining a routine, staying connected with supportive people, engaging in gentle physical activity, practicing good sleep hygiene, eating nutritious meals, and avoiding alcohol/drugs. It's important to seek professional help, especially if you experience persistent sadness, loss of interest, or thoughts of self-harm.",
                "sources": ["Depression Support Resources", "Mental Health Treatment Guidelines"]
            },
            "stress": {
                "answer": "Stress management is crucial for mental wellness. Effective techniques include: breaking tasks into smaller manageable steps, practicing time management, using relaxation techniques (progressive muscle relaxation, deep breathing), regular exercise, maintaining social connections, and setting healthy boundaries. Don't hesitate to seek support from friends, family, or mental health professionals.",
                "sources": ["Stress Management Techniques", "Wellness Resources"]
            },
            "sleep": {
                "answer": "Sleep and mental health are closely connected. Good sleep hygiene includes: keeping a consistent sleep schedule, creating a relaxing bedtime routine, avoiding screens 1 hour before bed, keeping your bedroom cool and dark, avoiding caffeine late in the day, and limiting naps. If sleep problems persist, consider speaking with a healthcare provider.",
                "sources": ["Sleep Hygiene Guidelines", "Mental Health and Sleep Resources"]
            }
        }
    
    def __call__(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        question = inputs.get("question", "").lower()
        
        # Search for relevant topics in the knowledge base
        for topic, data in self.knowledge_base.items():
            if topic in question:
                return {
                    "answer": data["answer"],
                    "source_documents": [
                        {"metadata": {"source": source, "topic": "Mental Health"}}
                        for source in data["sources"]
                    ]
                }
        
        # Generate a supportive response for general mental health questions
        if any(word in question for word in ["mental health", "feeling", "emotions", "mood", "sad", "happy", "angry", "worried"]):
            return {
                "answer": "Thank you for sharing your feelings with me. Your mental health and emotional well-being are important. While I can provide general guidance and support, I encourage you to reach out to mental health professionals for personalized care. What you're experiencing is valid, and support is available.",
                "source_documents": [{"metadata": {"source": "General Mental Health Support", "topic": "Mental Health"}}]
            }
        
        return {
            "answer": "I'm here to provide mental health support and guidance. Please feel free to share what's on your mind, and I'll do my best to offer helpful resources and coping strategies. Remember, seeking support is a sign of strength.",
            "source_documents": []
        }

class FallbackWebSearchTool:
    """Fallback web search tool for mental health resources using free APIs"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key  # Not used but kept for compatibility
        self.search_engines = {
            "duckduckgo": "https://api.duckduckgo.com/",
            "wikipedia": "https://en.wikipedia.org/api/rest_v1/"
        }
    
    def _run(self, query: str) -> str:
        """Run web search for mental health resources and return formatted results"""
        try:
            # Enhance query for mental health context
            mental_health_query = f"mental health {query} coping strategies support"
            
            # Use DuckDuckGo Instant Answer API (no key required)
            results = self._search_duckduckgo(mental_health_query)
            if results:
                return results
            
            # Fallback to Wikipedia search
            results = self._search_wikipedia(query)
            if results:
                return results
            
            return "No web search results found for this mental health query. Please consider reaching out to a mental health professional for personalized support."
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Web search temporarily unavailable. Please consider contacting mental health resources directly: {str(e)}"
    
    def _search_duckduckgo(self, query: str) -> str:
        """Search using DuckDuckGo Instant Answer API for mental health resources"""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Format the response for mental health context
                result_parts = []
                
                if data.get("Abstract"):
                    result_parts.append(f"**{data.get('Heading', 'Mental Health Information')}**\n{data['Abstract']}")
                    if data.get("AbstractURL"):
                        result_parts.append(f"Source: {data['AbstractURL']}")
                
                if data.get("Definition"):
                    result_parts.append(f"**Definition**\n{data['Definition']}")
                    if data.get("DefinitionURL"):
                        result_parts.append(f"Source: {data['DefinitionURL']}")
                
                # Add related mental health topics
                if data.get("RelatedTopics"):
                    for topic in data["RelatedTopics"][:2]:  # Limit to 2 topics
                        if isinstance(topic, dict) and topic.get("Text"):
                            result_parts.append(f"**Related Mental Health Information**\n{topic['Text']}")
                            if topic.get("FirstURL"):
                                result_parts.append(f"Source: {topic['FirstURL']}")
                
                if result_parts:
                    return "\n\n".join(result_parts)
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
        
        return ""
    
    def _search_wikipedia(self, query: str) -> str:
        """Search Wikipedia for mental health information"""
        try:
            # Search for pages
            search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
            # Clean query for Wikipedia
            clean_query = query.replace(" ", "_")
            
            response = requests.get(f"{search_url}{clean_query}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("extract"):
                    result = f"**{data.get('title', 'Mental Health Information')}**\n{data['extract']}"
                    if data.get("content_urls", {}).get("desktop", {}).get("page"):
                        result += f"\n\nSource: {data['content_urls']['desktop']['page']}"
                    return result
            
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
        
        return ""

# This function is now imported from tutor_agent.py, but we keep this as a fallback
def setup_fallback_agent():
    """Initialize the fallback mental health agent"""
    return MockMentalHealthAgent()

# Page Configuration
st.set_page_config(
    page_title="Mental Health Support Chatbot",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optimized CSS with calming mental health theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Mental health themed colors - calming and supportive */
    .stApp {
        background: linear-gradient(135deg, #f0f8f0 0%, #e8f5e8 100%);
    }
    
    /* Sidebar styling with calming colors */
    .css-1d391kg {
        background: linear-gradient(180deg, #d4f4dd 0%, #a8e6cf 100%);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #7fb069 0%, #5a9c4a 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(95, 156, 74, 0.2);
        text-align: center;
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .header-subtitle {
        color: #f0fff0;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Chat container styling */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border: 1px solid #e8f5e8;
    }
    
    /* User message styling */
    .user-message {
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8a3 100%);
        color: #2d5a3d;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 5px 18px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(168, 230, 207, 0.3);
        position: relative;
        max-width: 80%;
        margin-left: auto;
    }
    
    /* AI response styling */
    .ai-response {
        background: linear-gradient(135deg, #ffffff 0%, #f8fffe 100%);
        color: #2d3436;
        padding: 1.5rem;
        border-radius: 18px 18px 18px 5px;
        margin: 1rem 0;
        border-left: 4px solid #7fb069;
        box-shadow: 0 2px 15px rgba(0,0,0,0.08);
        position: relative;
        max-width: 85%;
    }
    
    /* Crisis alert styling */
    .crisis-alert {
        background: linear-gradient(135deg, #ffe6e6 0%, #ffcccc 100%);
        border: 2px solid #ff6b6b;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(255, 107, 107, 0.2);
    }
    
    .crisis-title {
        color: #d63031;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    /* Disclaimer styling */
    .disclaimer {
        background: linear-gradient(135deg, #fff5e6 0%, #ffeaa7 100%);
        border-left: 4px solid #fdcb6e;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #6c5ce7;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #7fb069 0%, #5a9c4a 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(127, 176, 105, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(127, 176, 105, 0.4);
        background: linear-gradient(135deg, #5a9c4a 0%, #4a7c3a 100%);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #a8e6cf;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.9);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #7fb069;
        box-shadow: 0 0 0 0.2rem rgba(127, 176, 105, 0.25);
    }
    
    /* Sidebar enhancements */
    .sidebar-title {
        color: #2d5a3d;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Metrics styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 3px solid #7fb069;
    }
    
    /* Source documents styling */
    .source-doc {
        background: #f8fffe;
        border: 1px solid #a8e6cf;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .source-title {
        color: #5a9c4a;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .user-message, .ai-response {
            max-width: 95%;
            padding: 1rem;
        }
        
        .chat-container {
            padding: 1rem;
        }
    }
    
    /* Animation for messages */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .user-message, .ai-response {
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f3f4;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #a8e6cf;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #7fb069;
    }
    
    /* Remove Streamlit branding */
    .css-1rs6os.edgvbvh3, .css-10trblm.e16nr0p30 {
        display: none;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Display header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">💚 Mental Health Support Chatbot</h1>
    <p class="header-subtitle">Your compassionate AI companion for mental wellness and emotional support</p>
</div>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables for mental health chatbot."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False
    
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "web_search_tool" not in st.session_state:
        st.session_state.web_search_tool = None
    
    if "use_web_search" not in st.session_state:
        st.session_state.use_web_search = True
    
    if "total_conversations" not in st.session_state:
        st.session_state.total_conversations = 0
    
    if "user_first_visit" not in st.session_state:
        st.session_state.user_first_visit = True
        # Add welcome message for first-time users
        welcome_message = """
        Welcome to your Mental Health Support Chatbot! 💚
        
        I'm here to provide compassionate support, practical coping strategies, and mental health resources. 
        
        **How I can help:**
        - Provide emotional support and validation
        - Share coping strategies for anxiety, stress, and difficult emotions
        - Offer resources for mental wellness
        - Guide you toward professional help when needed
        
        **Please remember:**
        - I'm not a substitute for professional mental health care
        - If you're in crisis, please reach out to emergency services or crisis hotlines
        - Your well-being is important, and seeking help is a sign of strength
        
        What's on your mind today? I'm here to listen and support you. 🤗
        """
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_message,
            "sources": [],
            "is_welcome": True
        })

def setup_sidebar():
    """Setup sidebar with mental health resources and configuration."""
    with st.sidebar:
        st.markdown('<p class="sidebar-title">💚 Mental Health Resources</p>', unsafe_allow_html=True)
        
        # Crisis resources section
        with st.expander("🚨 Crisis Resources", expanded=False):
            st.markdown("""
            **If you're in immediate danger, call 911 or go to your nearest emergency room.**
            
            **United States:**
            - 988 Suicide & Crisis Lifeline: 988
            - Crisis Text Line: Text HOME to 741741
            - National Domestic Violence Hotline: 1-800-799-7233
            
            **Canada:**
            - Talk Suicide Canada: 1-833-456-4566
            - Kids Help Phone: 1-800-668-6868
            
            **United Kingdom:**
            - Samaritans: 116 123
            - Crisis Text Line UK: Text SHOUT to 85258
            
            **International:**
            - Befrienders Worldwide: befrienders.org
            """)
        
        # Mental wellness tips
        with st.expander("🌱 Daily Wellness Tips", expanded=False):
            st.markdown("""
            **Daily Mental Health Practices:**
            - Practice deep breathing for 5 minutes
            - Write down 3 things you're grateful for
            - Take a short walk outside
            - Connect with a supportive friend or family member
            - Practice self-compassion and kindness
            - Maintain a regular sleep schedule
            - Limit social media if it affects your mood
            - Engage in activities that bring you joy
            """)
        
        # Professional help resources
        with st.expander("🔍 Finding Professional Help", expanded=False):
            st.markdown("""
            **How to Find Mental Health Support:**
            - Contact your primary care doctor for referrals
            - Check with your insurance for covered providers
            - Use Psychology Today's therapist finder
            - Contact your employee assistance program (EAP)
            - Reach out to community mental health centers
            - Consider telehealth options for convenience
            
            **Types of Mental Health Professionals:**
            - Psychologists - therapy and assessment
            - Psychiatrists - medication and therapy
            - Licensed Clinical Social Workers - therapy
            - Licensed Professional Counselors - therapy
            """)
        
        st.divider()
        
        # Configuration section
        st.markdown("⚙️ **Chatbot Settings**")
        
        # API Configuration
        tavily_key = st.text_input(
            "Tavily API Key (Optional)",
            type="password",
            help="Enter your Tavily API key for enhanced web search capabilities",
            placeholder="tvly-..."
        )
        
        exa_key = st.text_input(
            "Exa API Key (Optional)", 
            type="password",
            help="Enter your Exa API key for additional search capabilities",
            placeholder="..."
        )
        
        # Web search toggle
        use_web_search = st.checkbox(
            "🔍 Enable Web Search", 
            value=st.session_state.use_web_search,
            help="Enable web search for additional mental health resources"
        )
        st.session_state.use_web_search = use_web_search
        
        # Display API status
        st.markdown("📊 **API Status**")
        col1, col2 = st.columns(2)
        with col1:
            tavily_status = "🟢 Active" if tavily_key else "🔴 Not Set"
            st.markdown(f"**Tavily:** {tavily_status}")
        with col2:
            exa_status = "🟢 Active" if exa_key else "🔴 Not Set"
            st.markdown(f"**Exa:** {exa_status}")
        
        # Update environment variables
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key
        if exa_key:
            os.environ["EXA_API_KEY"] = exa_key
        
        st.divider()
        
        # Statistics
        st.markdown("📈 **Session Statistics**")
        
        with st.container():
            st.markdown(f"""
            <div class="metric-container">
                <strong>Total Conversations:</strong> {st.session_state.total_conversations}
            </div>
            """, unsafe_allow_html=True)
        
        # Agent status
        agent_status = "🟢 Active" if st.session_state.agent_initialized else "🔴 Initializing"
        st.markdown(f"""
        <div class="metric-container">
            <strong>Agent Status:</strong> {agent_status}
        </div>
        """, unsafe_allow_html=True)
        
        # Reset conversation button
        if st.button("🔄 New Conversation", help="Start a fresh conversation"):
            st.session_state.messages = []
            st.session_state.total_conversations = 0
            st.session_state.user_first_visit = True
            st.rerun()
        
        st.divider()
        
        # Support information
        st.markdown("💙 **About This Chatbot**")
        st.markdown("""
        This AI chatbot provides general mental health support and information. 
        It uses advanced language models to offer compassionate responses and practical guidance.
        
        **Remember:** This is not a substitute for professional mental health care.
        """)

@st.cache_resource
def get_qa_chain():
    """Initialize the mental health support agent with caching."""
    try:
        if REAL_AGENT_AVAILABLE:
            logger.info("🔄 Initializing real mental health agent...")
            agent = setup_mental_health_agent()
            logger.info("✅ Real mental health agent initialized successfully")
            return agent
        else:
            logger.info("🔄 Initializing fallback mental health agent...")
            agent = setup_fallback_agent()
            logger.info("✅ Fallback mental health agent initialized")
            return agent
    except Exception as e:
        logger.error(f"❌ Failed to initialize mental health agent: {e}")
        logger.info("🔄 Falling back to mock agent...")
        return setup_fallback_agent()

def initialize_agent():
    """Initialize the mental health agent and web search tool."""
    if not st.session_state.agent_initialized:
        with st.spinner("🔄 Initializing Mental Health Support Agent..."):
            try:
                # Initialize the mental health agent
                st.session_state.agent = get_qa_chain()
                
                # Initialize web search tool if available
                if st.session_state.use_web_search:
                    try:
                        if REAL_AGENT_AVAILABLE:
                            st.session_state.web_search_tool = setup_web_search_tool()
                        else:
                            st.session_state.web_search_tool = FallbackWebSearchTool()
                        logger.info("✅ Web search tool initialized")
                    except Exception as e:
                        logger.warning(f"⚠️ Web search tool failed to initialize: {e}")
                        st.session_state.web_search_tool = None
                
                st.session_state.agent_initialized = True
                logger.info("✅ Mental health agent initialization complete")
                
            except Exception as e:
                logger.error(f"❌ Agent initialization failed: {e}")
                st.error(f"Failed to initialize mental health agent: {str(e)}")
                # Use fallback agent
                st.session_state.agent = setup_fallback_agent()
                st.session_state.agent_initialized = True

def is_concerning_response(answer: str, question: str) -> tuple:
    """Check if the response indicates serious mental health concerns."""
    concerning_indicators = [
        "emergency", "crisis", "professional help immediately", 
        "cannot provide", "beyond my capabilities", "serious condition"
    ]
    
    answer_lower = answer.lower()
    is_concerning = any(indicator in answer_lower for indicator in concerning_indicators)
    
    # Check if question indicates crisis
    crisis_detected = False
    if REAL_AGENT_AVAILABLE:
        try:
            crisis_detected = detect_crisis_situation(question)
        except:
            crisis_detected = any(word in question.lower() for word in ["suicide", "kill myself", "hurt myself", "can't go on"])
    
    return is_concerning or crisis_detected, crisis_detected

def format_web_results(web_results: str):
    """Format web search results for mental health context."""
    if not web_results:
        return ""
    
    formatted_results = f"""
    
    🔍 **Additional Mental Health Resources from Web Search:**
    
    {web_results}
    
    *Please verify information with qualified mental health professionals.*
    """
    return formatted_results

def display_chat_history():
    """Display chat history with mental health appropriate styling."""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Check if this is a crisis-related response
            is_crisis = message.get("is_crisis", False)
            
            if is_crisis:
                st.markdown(f"""
                <div class="crisis-alert">
                    <div class="crisis-title">🚨 Important Mental Health Notice</div>
                    <div>{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ai-response">
                    <strong>💚 Mental Health Assistant:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            
            # Display sources if available
            if message.get("sources") and len(message["sources"]) > 0:
                with st.expander(f"📚 Sources ({len(message['sources'])} found)", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        source_info = source.get("metadata", {})
                        topic = source_info.get("topic", "Mental Health")
                        source_name = source_info.get("source", f"Resource {i}")
                        
                        st.markdown(f"""
                        <div class="source-doc">
                            <div class="source-title">{i}. {source_name}</div>
                            <div><strong>Topic:</strong> {topic}</div>
                        </div>
                        """, unsafe_allow_html=True)

def main():
    """Main application function for mental health chatbot."""
    # Initialize session state
    init_session_state()
    
    # Setup sidebar
    setup_sidebar()
    
    # Initialize agent
    initialize_agent()
    
    # Display disclaimer
    if REAL_AGENT_AVAILABLE:
        try:
            disclaimer = get_mental_health_disclaimer()
            st.markdown(f"""
            <div class="disclaimer">
                {disclaimer}
            </div>
            """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div class="disclaimer">
                ⚠️ <strong>Important:</strong> This AI provides general mental health information and support, 
                but is not a substitute for professional medical advice, diagnosis, or treatment. 
                In crisis situations, please contact emergency services or crisis hotlines immediately.
            </div>
            """, unsafe_allow_html=True)
    
    # Main chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Share what's on your mind... I'm here to listen and support you. 💚"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.total_conversations += 1
        
        # Check for crisis situation
        crisis_detected = False
        if REAL_AGENT_AVAILABLE:
            try:
                crisis_detected = detect_crisis_situation(prompt)
            except:
                crisis_detected = any(word in prompt.lower() for word in ["suicide", "kill myself", "hurt myself"])
        
        with st.spinner("💭 Thinking and preparing a supportive response..."):
            try:
                # Get response from mental health agent
                if st.session_state.agent:
                    result = st.session_state.agent({"question": prompt})
                    answer = result.get("answer", "I'm here to support you. Could you tell me more about what you're experiencing?")
                    sources = result.get("source_documents", [])
                else:
                    answer = "I'm experiencing some technical difficulties, but I want you to know that your feelings are valid and support is available. Please consider reaching out to a mental health professional."
                    sources = []
                
                # Check if response indicates serious concerns
                is_concerning, crisis_from_response = is_concerning_response(answer, prompt)
                crisis_detected = crisis_detected or crisis_from_response
                
                # Add crisis resources if needed
                if crisis_detected:
                    if REAL_AGENT_AVAILABLE:
                        try:
                            crisis_info = CRISIS_RESOURCES
                            answer = f"{answer}\n\n{crisis_info}"
                        except:
                            answer = f"{answer}\n\n🚨 If you're having thoughts of self-harm or suicide, please reach out for immediate help: Call 988 (US), 116 123 (UK), or contact emergency services."
                    else:
                        answer = f"{answer}\n\n🚨 If you're having thoughts of self-harm or suicide, please reach out for immediate help: Call 988 (US), 116 123 (UK), or contact emergency services."
                
                # Try web search for additional resources if enabled and not crisis
                web_results = ""
                if st.session_state.use_web_search and st.session_state.web_search_tool and not crisis_detected:
                    try:
                        web_search_query = f"mental health support {prompt}"
                        web_results = st.session_state.web_search_tool._run(web_search_query)
                        if web_results:
                            answer += format_web_results(web_results)
                    except Exception as e:
                        logger.warning(f"⚠️ Web search failed: {e}")
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "is_crisis": crisis_detected
                })
                
                logger.info(f"✅ Mental health response generated successfully")
                
            except Exception as e:
                logger.error(f"❌ Error generating response: {e}")
                error_message = """
                I apologize, but I'm experiencing some technical difficulties right now. 
                
                Your mental health and well-being are important. Please consider:
                - Reaching out to a trusted friend, family member, or mental health professional
                - Contacting a crisis helpline if you're in distress
                - Visiting a mental health resource website for immediate support
                
                Remember, seeking help is a sign of strength, and you don't have to go through difficult times alone. 💚
                """
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                    "is_crisis": False
                })
        
        # Rerun to display new messages
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with additional resources
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
        💚 Remember: You matter, your feelings are valid, and help is always available. 💚<br>
        <em>This chatbot provides general support and is not a substitute for professional mental health care.</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
""", unsafe_allow_html=True)