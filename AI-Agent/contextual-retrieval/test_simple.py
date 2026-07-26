#!/usr/bin/env python3
"""Simple test script for Agentic RAG system"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_basic_functionality():
    """Test basic functionality of the system"""
    print("🧪 Testing Agentic RAG System")
    print("="*60)
    
    # Import modules
    try:
        from config import Config, KnowledgeBaseType
        from agent import AgenticRAG
        from tools import KnowledgeBaseTools
        from chunking import DocumentChunker, DocumentIndexer
        
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test configuration
    print("\n📋 Testing Configuration...")
    try:
        config = Config.from_env()
        print(f"  Model: {config.llm.model}")
        print(f"  KB Type: {config.knowledge_base.type}")
        print(f"  Chunk Size: {config.chunking.chunk_size}")
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False
    
    # Test document chunking
    print("\n📄 Testing Document Chunking...")
    try:
        chunker = DocumentChunker(config.chunking)
        sample_text = """Intentional homicide is the intentional and unlawful deprivation of
        another person's life.

        Under Article 232 of the Criminal Law of the PRC, whoever intentionally kills another
        shall be sentenced to death, life imprisonment or fixed-term imprisonment of ten years
        or more; where the circumstances are relatively minor, to three to ten years.

        Sentencing considers the motive, the means used and the consequences."""
        
        chunks = chunker.chunk_text(sample_text, "test_doc")
        print(f"  Created {len(chunks)} chunks")
        print(f"  First chunk: {chunks[0]['text'][:100]}...")
        print("✅ Chunking works")
    except Exception as e:
        print(f"❌ Chunking error: {e}")
        return False
    
    # Test knowledge base tools
    print("\n🔧 Testing Knowledge Base Tools...")
    try:
        kb_tools = KnowledgeBaseTools(config.knowledge_base)
        
        # Add test document to store
        kb_tools.add_document(
            "test_doc_1",
            "Intentional homicide is punishable by death, life imprisonment, or "
            "fixed-term imprisonment of ten years or more.",
            {"source": "test"}
        )
        
        # Test document retrieval
        doc = kb_tools.get_document("test_doc_1")
        if "error" not in doc:
            print(f"  Retrieved document: {doc['doc_id']}")
            print("✅ Document storage works")
        else:
            print(f"⚠️  Document retrieval returned: {doc}")
    except Exception as e:
        print(f"❌ KB Tools error: {e}")
        return False
    
    # Test agent initialization
    print("\n🤖 Testing Agent Initialization...")
    try:
        agent = AgenticRAG(config)
        print(f"  Model: {agent.model}")
        print("✅ Agent initialized")
    except Exception as e:
        print(f"❌ Agent initialization error: {e}")
        print("  Make sure you have set the appropriate API key in .env")
        return False
    
    # Test simple query (if API key is available)
    if os.getenv("OPENAI_API_KEY") and os.getenv("RUN_LIVE_SMOKE_TESTS") == "1":
        print("\n💬 Testing Simple Query...")
        try:
            # Add some test data
            kb_tools.add_document(
                "criminal_law_test",
                """Thresholds for filing a theft case:
                1. Relatively large amount: generally 1,000 to 3,000 yuan or more
                2. Repeated theft: three or more thefts within two years
                3. Home-invasion theft, theft while carrying a weapon and pickpocketing count
                   regardless of the amount""",
                {"type": "law"}
            )
            
            # Test non-agentic query (simpler, less likely to fail)
            response = agent.query_non_agentic("threshold for filing a theft case", stream=False)
            
            if response and len(response) > 10:
                print(f"  Response: {response[:200]}...")
                print("✅ Query processing works")
            else:
                print(f"⚠️  Response was empty or too short: {response}")
        except Exception as e:
            print(f"⚠️  Query error: {e}")
            print("  This might be due to retrieval pipeline not running")
    else:
        print("\n⚠️  Skipping query test (no API key found)")
    
    print("\n" + "="*60)
    print("🎉 Basic functionality test complete!")
    return True


def test_evaluation_dataset():
    """Test evaluation dataset generation"""
    print("\n📊 Testing Evaluation Dataset...")
    
    try:
        # Import dataset builder
        import sys
        sys.path.append("evaluation")
        from dataset_builder import LegalDatasetBuilder, create_legal_documents
        
        # Build dataset
        builder = LegalDatasetBuilder()
        simple_cases = builder.create_simple_cases()
        complex_cases = builder.create_complex_cases()
        
        print(f"  Simple cases: {len(simple_cases)}")
        print(f"  Complex cases: {len(complex_cases)}")
        print(f"  First simple case: {simple_cases[0]['question']}")
        
        # Create documents
        documents = create_legal_documents()
        print(f"  Legal documents: {len(documents)}")
        
        print("✅ Evaluation dataset works")
        return True
        
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Agentic RAG System - Test Suite")
    print("="*60)
    
    # Run tests
    success = test_basic_functionality()
    
    if success:
        test_evaluation_dataset()
    
    print("\n" + "="*60)
    if success:
        print("✅ All basic tests passed!")
        print("\nNext steps:")
        print("1. Make sure retrieval pipeline is running:")
        print("   cd ../retrieval-pipeline && python main.py")
        print("\n2. Run the quickstart:")
        print("   python quickstart.py")
        print("\n3. Or start interactive mode:")
        print("   python main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("1. Missing API keys in .env file")
        print("2. Retrieval pipeline not running")
        print("3. Missing dependencies (run: pip install -r requirements.txt)")
