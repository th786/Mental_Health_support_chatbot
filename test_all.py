#!/usr/bin/env python3
"""
Comprehensive test script for Mental Health Support Chatbot
"""

import sys
import traceback
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test all critical imports."""
    print("🧪 Testing imports...")
    
    try:
        # Core Python modules
        import os
        import warnings
        import json
        print("  ✅ Core Python modules")
        
        # Web framework modules
        import streamlit as st
        import fastapi
        import uvicorn
        print("  ✅ Web framework modules")
        
        # AI/ML modules
        import torch
        import transformers
        import datasets
        print("  ✅ AI/ML modules")
        
        # LangChain modules (with new imports)
        from langchain.schema import Document
        from langchain_community.vectorstores import FAISS
        try:
            from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
            print("  ✅ LangChain modules (new imports)")
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.llms import HuggingFacePipeline
            print("  ✅ LangChain modules (fallback imports)")
        
        # Optional modules
        try:
            import tavily
            print("  ✅ Tavily library available")
        except ImportError:
            print("  ⚠️ Tavily library not available (optional)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_project_modules():
    """Test project-specific modules."""
    print("\n🔧 Testing project modules...")
    
    try:
        # Test config
        from config import config
        print("  ✅ Config module")
        
        # Test utils
        from utils import setup_production_logging, cache
        print("  ✅ Utils module")
        
        # Test mental health agent
        from tutor_agent import setup_mental_health_agent, setup_web_search_tool
        print("  ✅ Mental health agent module")
        
        # Test web search
        from web_search import create_web_search_tool
        print("  ✅ Web search module")
        
        # Test API
        import api
        print("  ✅ API module")
        
        # Test main app (this might generate warnings but shouldn't fail)
        import app
        print("  ✅ Main app module")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Project module error: {e}")
        traceback.print_exc()
        return False

def test_functionality():
    """Test basic functionality."""
    print("\n⚙️ Testing functionality...")
    
    try:
        # Test agent initialization (lightweight test)
        from tutor_agent import Config, get_device
        
        # Test device detection
        device = get_device()
        print(f"  ✅ Device detection: {device}")
        
        # Test configuration
        config = Config()
        print(f"  ✅ Configuration loaded: {config.EMBEDDING_MODEL}")
        
        # Test logging setup
        from utils import setup_production_logging
        logger_test = setup_production_logging("INFO", "logs/test.log")
        logger_test.info("Test log message")
        print("  ✅ Logging system")
        
        # Test cache
        from utils import cache
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        assert value == "test_value"
        print("  ✅ Cache system")
        
        # Test web search
        from web_search import create_web_search_tool
        search_tool = create_web_search_tool()
        search_result = search_tool._run("test query")
        assert isinstance(search_result, str)
        print("  ✅ Web search system")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality error: {e}")
        traceback.print_exc()
        return False

def test_streamlit_compatibility():
    """Test Streamlit compatibility."""
    print("\n🎨 Testing Streamlit compatibility...")
    
    try:
        import streamlit as st
        
        # Test basic streamlit functions
        print("  ✅ Streamlit imported successfully")
        
        # Test if we can access streamlit functions without context errors
        # (We can't fully test these without a running streamlit app, but we can import)
        from streamlit import session_state, cache_resource, chat_message
        print("  ✅ Streamlit functions accessible")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Streamlit compatibility error: {e}")
        traceback.print_exc()
        return False

def test_api_compatibility():
    """Test FastAPI compatibility."""
    print("\n🚀 Testing FastAPI compatibility...")
    
    try:
        from fastapi import FastAPI
        from api import app as api_app
        
        print("  ✅ FastAPI app created successfully")
        
        # Test if we can access the app routes
        routes = [route.path for route in api_app.routes]
        expected_routes = ["/", "/health", "/query", "/stats"]
        for route in expected_routes:
            if route in routes:
                print(f"  ✅ Route {route} available")
            else:
                print(f"  ⚠️ Route {route} not found")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FastAPI compatibility error: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("💚 Mental Health Support Chatbot - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    tests = [
        ("Import Tests", test_imports),
        ("Project Module Tests", test_project_modules),
        ("Functionality Tests", test_functionality),
        ("Streamlit Compatibility", test_streamlit_compatibility),
        ("FastAPI Compatibility", test_api_compatibility)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! The Educational Tutor Agent is ready to use.")
        print("\nTo run the application:")
        print("  Streamlit: streamlit run app.py")
        print("  FastAPI:   uvicorn api:app --reload")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed. Please check the errors above.")
    
    print(f"\nTest completed at: {datetime.now()}")
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 