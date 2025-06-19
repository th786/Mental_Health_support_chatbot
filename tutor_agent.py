"""
Mental Health Support Chatbot - Compassionate AI Assistant for Mental Wellness
"""

import os
import warnings
import logging
from typing import Optional, List, Dict, Any
from functools import lru_cache

# Load environment variables explicitly
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Compatibility setup (once)
os.environ.update({
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "0" if os.getenv("CUDA_VISIBLE_DEVICES") else ""
})
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Core imports with error handling
try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    
    # Use new HuggingFace imports
    try:
        from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    except ImportError:
        # Fallback to old imports
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.llms import HuggingFacePipeline
        except ImportError:
            from langchain.embeddings import HuggingFaceEmbeddings
            from langchain.llms import HuggingFacePipeline
    
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.tools import BaseTool
    from langchain.callbacks.manager import CallbackManagerForToolRun
    from langchain.prompts import PromptTemplate
    
    import torch
    from datasets import load_dataset
    from transformers import pipeline
    
    logger.info("✅ Core dependencies loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import core dependencies: {e}")
    raise ImportError(f"Missing required dependencies: {e}")

# Web search imports
try:
    from web_search import create_web_search_tool
    WEB_SEARCH_AVAILABLE = True
    logger.info("✅ Enhanced web search available")
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    logger.warning("⚠️ Web search module not available")

# Optimized configuration class
class Config:
    """Configuration settings for the mental health chatbot."""
    
    # Model settings - using more reliable models
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "google/flan-t5-base"  # Good for supportive conversations
    
    # Processing settings
    CHUNK_SIZE = 1000  # Good for mental health articles
    CHUNK_OVERLAP = 200  # Increased overlap
    TOP_K_RETRIEVAL = 4  # Increased for better coverage
    MAX_DATASET_SIZE = 500  # Mental health resources
    
    # Generation settings - tuned for empathetic responses
    MAX_NEW_TOKENS = 300  # Longer for supportive responses
    MIN_LENGTH = 50  # Ensure substantial responses
    TEMPERATURE = 0.3  # Slightly higher for more human-like warmth
    DO_SAMPLE = True  # Enable sampling for natural responses
    NUM_BEAMS = 2  # Beam search for better quality
    
    # API settings
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Device settings
    FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"

config = Config()

# Configure torch with better error handling
try:
    torch.set_num_threads(1)
    if hasattr(torch, 'set_grad_enabled'):
        torch.set_grad_enabled(False)
    logger.info("✅ PyTorch configured")
except Exception as e:
    logger.warning(f"⚠️ PyTorch configuration warning: {e}")

@lru_cache(maxsize=1)
def get_device() -> str:
    """Get device configuration with caching and better logic."""
    if config.FORCE_CPU:
        logger.info("🖥️ Forcing CPU usage")
        return 'cpu'
    
    if torch.cuda.is_available():
        logger.info("🚀 CUDA available - using GPU")
        return 'cuda'
    else:
        logger.info("🖥️ CUDA not available - using CPU")
        return 'cpu'

def load_mental_health_dataset(max_size: int = None) -> Any:
    """Load mental health dataset with better error handling."""
    if max_size is None:
        max_size = config.MAX_DATASET_SIZE
    
    try:
        logger.info(f"🔄 Loading mental health dataset (max {max_size} examples)...")
        
        # Try loading mental health datasets
        try:
            # Try loading mental health counseling dataset
            dataset = load_dataset("Amod/mental_health_counseling_conversations", split=f"train[:{max_size}]")
        except Exception:
            try:
                # Fallback to other mental health datasets
                logger.warning("Trying alternative mental health dataset...")
                dataset = load_dataset("heliosbrahma/mental_health_chatbot_dataset", split=f"train[:{max_size}]")
            except Exception:
                # Final fallback
                logger.warning("Using fallback dataset creation...")
                return create_mental_health_fallback_dataset()
        
        logger.info(f"✅ Loaded {len(dataset)} mental health examples")
        return dataset
        
    except Exception as e:
        logger.error(f"❌ Failed to load mental health dataset: {e}")
        # Create a mental health fallback dataset
        logger.info("Creating mental health fallback dataset...")
        return create_mental_health_fallback_dataset()

def create_mental_health_fallback_dataset() -> List[Dict]:
    """Create a mental health focused fallback dataset."""
    fallback_data = [
        {
            "question": "I'm feeling really anxious lately. What can I do?",
            "response": "Anxiety is very common and there are many effective strategies to help manage it. Some techniques include deep breathing exercises, mindfulness meditation, regular physical activity, and maintaining a consistent sleep schedule. It's also important to limit caffeine, practice grounding techniques like the 5-4-3-2-1 method (5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste), and consider talking to a mental health professional if symptoms persist.",
            "topic": "Anxiety Management",
            "support_level": "general_guidance"
        },
        {
            "question": "I'm having trouble sleeping and it's affecting my mood.",
            "response": "Sleep and mental health are closely connected. Poor sleep can significantly impact mood, stress levels, and overall well-being. Some sleep hygiene tips include: keeping a consistent sleep schedule, creating a relaxing bedtime routine, avoiding screens 1 hour before bed, keeping your bedroom cool and dark, avoiding caffeine late in the day, and limiting naps. If sleep problems persist, it may be helpful to speak with a healthcare provider about potential underlying causes.",
            "topic": "Sleep and Mental Health",
            "support_level": "general_guidance"
        },
        {
            "question": "How do I know if I need professional help?",
            "response": "It's important to seek professional help if you're experiencing: persistent feelings of sadness, hopelessness, or emptiness that last more than two weeks; significant changes in appetite or sleep patterns; loss of interest in activities you once enjoyed; difficulty concentrating or making decisions; thoughts of self-harm or suicide; substance use to cope; or if mental health symptoms are interfering with work, relationships, or daily activities. Remember, seeking help is a sign of strength, not weakness.",
            "topic": "When to Seek Help",
            "support_level": "guidance_and_referral"
        },
        {
            "question": "I'm feeling overwhelmed with stress. Any suggestions?",
            "response": "Feeling overwhelmed is a common experience, and there are many ways to manage stress effectively. Try breaking large tasks into smaller, manageable steps. Practice time management and prioritization techniques. Use stress-reduction methods like deep breathing, progressive muscle relaxation, or brief mindfulness exercises. Regular physical activity, even just a short walk, can significantly reduce stress hormones. Don't hesitate to reach out to friends, family, or mental health professionals for support.",
            "topic": "Stress Management",
            "support_level": "general_guidance"
        },
        {
            "question": "What are some healthy coping strategies?",
            "response": "Healthy coping strategies include: mindfulness and meditation, regular exercise, maintaining social connections, creative activities (art, music, writing), spending time in nature, practicing gratitude, journaling, deep breathing exercises, progressive muscle relaxation, maintaining a routine, getting adequate sleep, eating nutritious foods, setting boundaries, and seeking support from friends, family, or professionals. Remember, what works for one person may not work for another, so it's important to find strategies that feel right for you.",
            "topic": "Healthy Coping Strategies",
            "support_level": "general_guidance"
        }
    ]
    logger.info(f"✅ Created mental health fallback dataset with {len(fallback_data)} examples")
    return fallback_data

def prepare_mental_health_documents(data) -> List[Document]:
    """Convert mental health dataset to documents with improved processing."""
    documents = []
    
    try:
        # Handle both dataset objects and lists
        items = data if isinstance(data, list) else list(data)
        
        for i, item in enumerate(items):
            try:
                # Extract question and response for mental health conversations
                if isinstance(item, dict):
                    question = str(item.get("question", "")).strip()
                    response = str(item.get("response", "")).strip()
                    topic = str(item.get("topic", "Mental Health")).strip()
                    support_level = str(item.get("support_level", "general_guidance")).strip()
                else:
                    # Handle different dataset formats
                    question = str(getattr(item, "question", getattr(item, "input", ""))).strip()
                    response = str(getattr(item, "response", getattr(item, "output", ""))).strip()
                    topic = str(getattr(item, "topic", "Mental Health")).strip()
                    support_level = "general_guidance"
                
                if not question or not response:
                    continue
                
                # Create comprehensive content for mental health support
                content = f"""
Mental Health Query: {question}

Supportive Response: {response}

Topic: {topic}
Support Level: {support_level}

This information provides guidance for mental health and emotional well-being. 
Always remember that this is general information and not a substitute for professional mental health care.
"""
                
                # Create document with comprehensive metadata
                doc = Document(
                    page_content=content.strip(),
                    metadata={
                        "source": f"Mental Health Resource {i+1}",
                        "topic": topic,
                        "support_level": support_level,
                        "type": "mental_health_guidance",
                        "question": question,
                        "response": response
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"⚠️ Skipping item {i}: {e}")
                continue
        
        logger.info(f"✅ Prepared {len(documents)} mental health documents")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Error preparing mental health documents: {e}")
        return []

@lru_cache(maxsize=1)
def initialize_embeddings():
    """Initialize embeddings model with caching."""
    try:
        logger.info(f"🔄 Loading embeddings model: {config.EMBEDDING_MODEL}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': get_device()},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        logger.info("✅ Embeddings model loaded successfully")
        return embeddings
    except Exception as e:
        logger.error(f"❌ Failed to load embeddings: {e}")
        raise

def create_vector_store(documents: List[Document], embeddings):
    """Create vector store from documents."""
    try:
        if not documents:
            logger.warning("⚠️ No documents provided for vector store")
            return None
        
        logger.info(f"🔄 Creating vector store from {len(documents)} mental health documents...")
        
        # Split documents for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"📄 Split into {len(split_docs)} chunks")
        
        # Create FAISS vector store
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        logger.info("✅ Vector store created successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"❌ Failed to create vector store: {e}")
        raise

@lru_cache(maxsize=1)
def initialize_llm():
    """Initialize language model with caching and better error handling."""
    try:
        logger.info(f"🔄 Loading LLM: {config.LLM_MODEL}")
        
        # Initialize the pipeline with mental health-appropriate settings
        pipe = pipeline(
            "text2text-generation",
            model=config.LLM_MODEL,
            device=0 if get_device() == 'cuda' else -1,
            max_new_tokens=config.MAX_NEW_TOKENS,
            min_length=config.MIN_LENGTH,
            temperature=config.TEMPERATURE,
            do_sample=config.DO_SAMPLE,
            num_beams=config.NUM_BEAMS,
            pad_token_id=50256
        )
        
        # Wrap in LangChain
        llm = HuggingFacePipeline(pipeline=pipe)
        
        logger.info("✅ LLM loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"❌ Failed to load LLM: {e}")
        raise

def setup_web_search_tool():
    """Setup web search tool for mental health resources."""
    try:
        if WEB_SEARCH_AVAILABLE:
            logger.info("🔄 Setting up enhanced web search for mental health resources...")
            # Explicitly pass the Tavily API key
            tavily_key = os.getenv("TAVILY_API_KEY")
            if tavily_key:
                logger.info("✅ Tavily API key found in environment")
            else:
                logger.warning("⚠️ Tavily API key not found in environment")
            
            web_search_tool = create_web_search_tool(tavily_api_key=tavily_key)
            logger.info("✅ Enhanced web search tool initialized")
            return web_search_tool
        else:
            logger.warning("⚠️ Web search not available")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to setup web search: {e}")
        return None

def setup_mental_health_agent():
    """Setup the main mental health support agent."""
    try:
        logger.info("🔄 Setting up Mental Health Support Agent...")
        
        # Load mental health data and create knowledge base
        data = load_mental_health_dataset()
        documents = prepare_mental_health_documents(data)
        
        if not documents:
            logger.error("❌ No mental health documents available")
            return None
        
        # Initialize components
        embeddings = initialize_embeddings()
        vectorstore = create_vector_store(documents, embeddings)
        llm = initialize_llm()
        
        # Setup conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create mental health-focused prompt template
        mental_health_prompt = PromptTemplate(
            template="""You are a compassionate mental health support chatbot. Your role is to provide empathetic, supportive, and helpful responses while maintaining appropriate boundaries.

IMPORTANT GUIDELINES:
- Always be empathetic, non-judgmental, and supportive
- Provide practical coping strategies and resources when appropriate
- Encourage professional help when needed, especially for serious mental health concerns
- Never provide medical diagnoses or replace professional treatment
- If someone mentions self-harm or suicide, provide crisis resources immediately
- Maintain confidentiality and respect privacy
- Use warm, understanding language

Context from mental health resources:
{context}

Previous conversation:
{chat_history}

User's query: {question}

AI Assistant:""",
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": mental_health_prompt}
        )
        
        logger.info("✅ Mental Health Support Agent setup complete")
        return qa_chain
    except Exception as e:
        logger.error(f"❌ Failed to setup Mental Health Support Agent: {e}")
        return None

# Main execution guard
if __name__ == "__main__":
    try:
        logger.info("🧪 Testing Mental Health Support Agent...")
        agent = setup_mental_health_agent()
        
        # Test query
        test_query = "I'm feeling really anxious lately. What can I do?"
        result = agent({"question": test_query})
        print(f"\nTest Query: {test_query}")
        print(f"Answer: {result['answer']}")
        print("✅ Agent test successful!")
        
    except Exception as e:
        logger.error(f"❌ Agent test failed: {e}")
        print(f"Error: {e}")

# Crisis resources and safety information
CRISIS_RESOURCES = """
🚨 CRISIS RESOURCES - Please reach out for immediate help:

🇺🇸 United States:
• National Suicide Prevention Lifeline: 988 or 1-800-273-8255
• Crisis Text Line: Text HOME to 741741
• National Domestic Violence Hotline: 1-800-799-7233

🇨🇦 Canada:
• Talk Suicide Canada: 1-833-456-4566
• Kids Help Phone: 1-800-668-6868

🇬🇧 United Kingdom:
• Samaritans: 116 123
• Crisis Text Line UK: Text SHOUT to 85258

🌍 International:
• Befrienders Worldwide: befrienders.org
• International Association for Suicide Prevention: iasp.info/resources/Crisis_Centres

If you're in immediate danger, please contact emergency services (911, 999, 112) or go to your nearest emergency room.
"""

def detect_crisis_situation(text: str) -> bool:
    """Detect if the user might be in a crisis situation requiring immediate help."""
    crisis_keywords = [
        "suicide", "kill myself", "end it all", "don't want to live", "hurt myself",
        "self-harm", "cutting", "overdose", "die", "death", "can't go on",
        "no point", "hopeless", "worthless", "better off dead", "harm myself"
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in crisis_keywords)

def get_mental_health_disclaimer() -> str:
    """Get the mental health disclaimer text."""
    return """
⚠️ **Important Disclaimer**: 
This AI chatbot provides general mental health information and support, but it is not a substitute for professional medical advice, diagnosis, or treatment. If you're experiencing a mental health crisis, please contact emergency services or a crisis hotline immediately. Always consult with qualified healthcare professionals for personalized mental health care.
"""