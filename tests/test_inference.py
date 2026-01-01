import os
import sys
# Add project root to sys.path
sys.path.append(os.getcwd())

from src.inference import load_artifacts, preprocess_and_predict
import os

def test_inference():
    print("Testing Baseline Model...")
    os.environ["MODEL_TYPE"] = "baseline"
    load_artifacts()
    
    title = "Breaking: Local Cat Becomes Mayor"
    text = "In a surprising turn of events, a local tabby cat has been elected mayor of the small town."
    
    result = preprocess_and_predict(title, text)
    print(f"Result (Baseline): {result}")
    assert "prediction" in result
    assert "probability" in result

    if os.path.exists("models/distilbert_model"):
        print("\nTesting BERT Model...")
        os.environ["MODEL_TYPE"] = "bert"
        load_artifacts()
        
        result_bert = preprocess_and_predict(title, text)
        print(f"Result (BERT): {result_bert}")
        assert "prediction" in result_bert
        assert "probability" in result_bert
    else:
        print("\nSkipping BERT test (model not found in models/distilbert_model)")

if __name__ == "__main__":
    try:
        test_inference()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
