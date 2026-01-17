# docubot/13. test_llm_fix.py
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_engine.llm_client import LLMClient

def main():
    print("Testing LLM Client...")
    
    try:
        client = LLMClient()
        print("✅ LLMClient initialized")
        
        # Test supported models
        models = client.list_available_models()
        print(f"✅ {len(models)} supported models")
        
        # Test model switching
        if client.switch_model("mistral:7b"):
            print("✅ Can switch to Mistral 7B")
            client.switch_model("llama2:7b")  # Switch back
        else:
            print("⚠️ Cannot switch to Mistral (may not be installed)")
        
        # Test health check
        health = client.health_check()
        if health['success']:
            print(f"✅ Ollama is healthy ({health['available_models']} models)")
        else:
            print(f"⚠️ Ollama not running: {health.get('error', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nTest {'passed' if success else 'failed'}")