"""
Test script to verify models can be loaded correctly
Run this before using the app to ensure everything is set up properly
"""

import os
import tensorflow as tf
import numpy as np

# Model paths
MODEL_PATHS = {
    'MobileNetV2': 'MobileNetV2/best_Colorectal_Classifier_Balanced.keras',
    'ResNet50': 'ResNet50/best_Colorectal_Classifier_ResNet50.keras'
}

CLASS_LABELS = ['Adenocarcinoma', 'High-grade IN', 'Low-grade IN', 'Normal', 'Polyp', 'Serrated adenoma']
IMG_SIZE = (128, 128)

def test_model_loading():
    """Test if models can be loaded"""
    print("=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    results = {}
    
    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n📦 Testing {model_name}...")
        print(f"   Path: {model_path}")
        
        # Check if file exists
        if not os.path.exists(model_path):
            print(f"   ❌ ERROR: Model file not found!")
            results[model_name] = False
            continue
        
        # Try to load model
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"   ✅ Model loaded successfully")
            print(f"   📊 Input shape: {model.input_shape}")
            print(f"   📊 Output shape: {model.output_shape}")
            
            # Test prediction with dummy image
            dummy_img = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
            try:
                prediction = model.predict(dummy_img, verbose=0)
                pred_class = CLASS_LABELS[np.argmax(prediction[0])]
                confidence = np.max(prediction[0])
                print(f"   ✅ Test prediction successful")
                print(f"   📈 Sample prediction: {pred_class} ({confidence:.2%})")
                results[model_name] = True
            except Exception as e:
                print(f"   ⚠️  Model loaded but prediction failed: {str(e)}")
                results[model_name] = False
                
        except Exception as e:
            print(f"   ❌ ERROR loading model: {str(e)}")
            results[model_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for model_name, success in results.items():
        status = "✅ READY" if success else "❌ FAILED"
        print(f"{model_name}: {status}")
    
    all_ready = all(results.values())
    if all_ready:
        print("\n🎉 All models are ready! You can run the app now.")
        print("   Run: streamlit run app.py")
    else:
        print("\n⚠️  Some models failed to load. Please check the error messages above.")
    
    return all_ready

if __name__ == "__main__":
    test_model_loading()

