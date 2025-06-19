# 💚 Mental Health Support Chatbot

A compassionate AI-powered chatbot designed to provide mental health support, emotional guidance, and wellness resources. Built with advanced language models and empathetic design principles to offer accessible mental health assistance.

## ⚠️ Important Disclaimer

**This chatbot is NOT a substitute for professional mental health care.** It provides general information and support, but should not replace consultation with qualified mental health professionals. In crisis situations, please contact emergency services or crisis hotlines immediately.

## 🌟 Features

### Core Capabilities
- **Empathetic Conversations**: Natural, compassionate responses to mental health concerns
- **Crisis Detection**: Automatically identifies potential crisis situations and provides immediate resources
- **Coping Strategies**: Practical techniques for managing anxiety, stress, and emotional challenges
- **Resource Guidance**: Information about finding professional mental health support
- **24/7 Availability**: Always accessible when you need support

### Advanced Features
- **Multi-tier Web Search**: Enhanced search using Tavily API, Exa API, and free search engines
- **Contextual Memory**: Maintains conversation context for personalized support
- **Source Attribution**: Provides reliable sources for mental health information
- **Crisis Resources**: Immediate access to crisis hotlines and emergency contacts
- **Wellness Tips**: Daily mental health practices and self-care suggestions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Pip package manager
- Optional: GPU for faster processing

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd mental-health-chatbot
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables (optional):**
```bash
cp .env.example .env
# Edit .env with your API keys (optional for enhanced features)
```

### Running the Application

#### Streamlit Web Interface (Recommended)
```bash
streamlit run app.py --server.fileWatcherType none
```
Access at: http://localhost:8501

#### FastAPI Backend (Optional)
```bash
python api.py
```
Access at: http://localhost:8000

## 🔧 Configuration

### Environment Variables

Create a `.env` file with optional API keys for enhanced functionality:

```env
# Optional: Enhanced web search capabilities
TAVILY_API_KEY=your_tavily_api_key_here
EXA_API_KEY=your_exa_api_key_here

# Optional: Performance settings
FORCE_CPU=false
LOG_LEVEL=INFO
ENABLE_CRISIS_DETECTION=true
CRISIS_RESPONSE_MODE=supportive
```

### Configuration Options

The chatbot can be customized through `config.py`:

- **Model Settings**: Choose between different language models
- **Response Length**: Adjust response detail level
- **Crisis Detection**: Enable/disable automatic crisis detection
- **Web Search**: Configure search capabilities
- **Rate Limiting**: Control usage limits for safety

## 💬 Usage Examples

### Getting Started
Simply start a conversation by sharing what's on your mind:

```
User: "I've been feeling really anxious lately about work."

AI: "I understand that work-related anxiety can be really challenging. It's completely normal to feel this way, and I'm here to support you. Some effective strategies for managing work anxiety include..."
```

### Crisis Support
The chatbot automatically detects crisis situations and provides immediate resources:

```
User: "I don't think I can handle this anymore."

AI: "I hear that you're going through a really difficult time right now, and I want you to know that you don't have to face this alone. 

🚨 CRISIS RESOURCES - Please reach out for immediate help:
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
..."
```

## 🛡️ Safety Features

### Crisis Detection
- Automatic identification of crisis language
- Immediate display of emergency resources
- Prioritization of professional help guidance

### Professional Boundaries
- Clear disclaimers about AI limitations
- Consistent encouragement of professional care
- No medical diagnosis or prescription advice

### Privacy Protection
- No conversation data stored permanently
- Session-based memory only
- Confidential and judgment-free environment

## 📚 Mental Health Resources

### Crisis Hotlines

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
- International Association for Suicide Prevention: iasp.info

### Finding Professional Help
- Contact your primary care doctor for referrals
- Check with your insurance for covered providers
- Use Psychology Today's therapist finder
- Contact your employee assistance program (EAP)
- Reach out to community mental health centers

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_all.py
```

Test categories:
- Core agent functionality
- Crisis detection accuracy
- Web search integration
- Memory and context handling
- Safety and boundary testing

## 🔍 Technical Architecture

### Core Components
- **Mental Health Agent**: Main conversational AI with specialized mental health training
- **Crisis Detection System**: Pattern matching and keyword analysis for crisis situations
- **Web Search Integration**: Multi-tier search system for current mental health resources
- **Memory Management**: Conversation context and personalization
- **Safety Guardrails**: Professional boundary maintenance and crisis response

### Technology Stack
- **Frontend**: Streamlit with custom CSS for calming, accessible design
- **Backend**: FastAPI for API endpoints
- **AI Models**: HuggingFace Transformers with mental health-focused fine-tuning
- **Search**: Tavily API, Exa API, DuckDuckGo, Wikipedia
- **Storage**: Session-based memory (no persistent data storage)

## 🤝 Contributing

We welcome contributions that improve mental health support and user safety:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/mental-health-improvement`)
3. Commit changes (`git commit -am 'Add supportive feature'`)
4. Push to branch (`git push origin feature/mental-health-improvement`)
5. Create a Pull Request

### Contribution Guidelines
- Prioritize user safety and well-being
- Follow mental health best practices
- Include tests for new features
- Maintain empathetic and professional tone
- Respect privacy and confidentiality principles

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚡ Performance Notes

- **Cold Start**: Initial model loading may take 30-60 seconds
- **Response Time**: Typically 2-5 seconds per response
- **Memory Usage**: Approximately 2-4GB RAM
- **GPU Acceleration**: Optional but recommended for faster responses

## 🔄 Recent Updates

### Crisis Detection
- Added automatic crisis situation detection
- Integrated emergency resource display
- Enhanced safety response protocols

### Web Search Enhancement
- Multi-tier search system (Tavily → Exa → Free APIs)
- Mental health-focused query enhancement
- Source verification and attribution

### UI/UX Improvements
- Calming color scheme optimized for mental wellness
- Crisis alert styling and prominence
- Mobile-responsive design
- Accessibility enhancements

## 📞 Support

For technical support or questions about the mental health chatbot:

- Create an issue on GitHub
- Review the troubleshooting guide
- Check the FAQ section

**For mental health emergencies, please contact emergency services or crisis hotlines immediately.**

---

💚 **Remember**: Your mental health matters, seeking help is a sign of strength, and you are not alone. This chatbot is here to support you, but please reach out to qualified professionals for comprehensive mental health care.