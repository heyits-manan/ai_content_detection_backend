"""
Test script for AI-generated image detection using SDXL detector
Updated to handle multiple label formats (artificial/human, ai-generated/real, etc.)
"""

# Step 1: Import required libraries
import os
import time
from PIL import Image
import requests
from transformers import pipeline
from pathlib import Path

# Step 2: Create a directory for test images if it doesn't exist
TEST_DIR = Path("test_samples/images")
TEST_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("AI IMAGE DETECTION TEST")
print("=" * 60)

# Step 3: Load the model (this will download on first run)
print("\n📥 Loading SDXL image detector model...")
print("   (This will download ~87MB on first run)")
start_time = time.time()

try:
    # Using the SDXL detector from Hugging Face
    # This pipeline handles all the complex preprocessing automatically
    detector = pipeline(
        "image-classification", 
        model="Organika/sdxl-detector",
        device="mps"  # Use "cpu" for Intel Macs, "cuda" for NVIDIA, "mps" for Apple Silicon
    )
    
    load_time = time.time() - start_time
    print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
    print(f"   Device being used: {detector.device}")
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)


def detect_image(image_path):
    """
    Takes an image path, runs detection, returns results
    Updated to handle multiple label formats intelligently
    """
    try:
        # Load the image
        image = Image.open(image_path)
        print(f"   📸 Image size: {image.size}, Format: {image.format}")
        
        # Run detection
        start_time = time.time()
        results = detector(image)
        inference_time = time.time() - start_time
        
        # Print raw results for debugging
        print(f"   📋 Raw model output: {results}")
        
        # --- IMPROVED LABEL DETECTION LOGIC ---
        
        # Define possible label patterns for AI and real
        ai_keywords = ['ai', 'artificial', 'fake', 'generated', 'synthetic', 'LABEL_1']
        real_keywords = ['human', 'real', 'natural', 'authentic', 'original', 'LABEL_0']
        
        ai_score = 0
        real_score = 0
        
        for result in results:
            label = result['label'].lower()
            score = result['score']
            
            # Check if this label matches any AI keywords
            if any(keyword in label for keyword in ai_keywords):
                ai_score = score
                print(f"   🔍 Found AI label: '{result['label']}' with score {score:.4f}")
            
            # Check if this label matches any real keywords
            elif any(keyword in label for keyword in real_keywords):
                real_score = score
                print(f"   🔍 Found Real label: '{result['label']}' with score {score:.4f}")
        
        # If we still couldn't determine, use a heuristic
        if ai_score == 0 and real_score == 0:
            print("   ⚠️ Could not identify labels by keywords, using position heuristic")
            # Assume first result is AI, second is human (common pattern)
            ai_score = results[0]['score']
            real_score = results[1]['score'] if len(results) > 1 else 1 - ai_score
        
        # If one score is 0, calculate it from the other
        if ai_score == 0 and real_score > 0:
            ai_score = 1 - real_score
        elif real_score == 0 and ai_score > 0:
            real_score = 1 - ai_score
        
        # Determine verdict
        is_ai_generated = ai_score > 0.5
        
        return {
            'ai_probability': ai_score,
            'real_probability': real_score,
            'is_ai_generated': is_ai_generated,
            'inference_time_ms': inference_time * 1000,
            'raw_results': results
        }
        
    except Exception as e:
        print(f"   ❌ Error processing image: {e}")
        return None

# Step 6: Gather test images
print("\n📂 Preparing test images...")

test_images = []

# Option A: Use your own images
# Add paths to your own images here
your_images = [
    "test_samples/images/ai-generated1.jpg",
    "test_samples/images/ai-generated2.jpg",
    "test_samples/images/ai-generated3.png",
    "test_samples/images/ai-generated4.webp",
    "test_samples/images/real1.jpg",


    # "test_samples/images/my_ai_image.png",
]

if your_images and any(os.path.exists(img) for img in your_images if img):
    test_images.extend([img for img in your_images if os.path.exists(img)])
    print(f"   ✅ Found {len(test_images)} local images")
else:
    print("   No local images found, downloading sample images...")

# Step 7: Run detection on all test images
print("\n🔍 Running detection on test images...")
print("-" * 60)

# Track statistics
correct_predictions = 0
total_tests = 0

for i, image_path in enumerate(test_images, 1):
    print(f"\n📷 Test Image {i}: {os.path.basename(image_path)}")
    
    result = detect_image(image_path)
    
    if result:
        total_tests += 1
        
        # Print results in a nice format
        print(f"   🤖 AI Probability: {result['ai_probability']:.2%}")
        print(f"   👤 Real Probability: {result['real_probability']:.2%}")
        print(f"   📊 Verdict: {'🤖 AI GENERATED' if result['is_ai_generated'] else '📸 REAL PHOTO'}")
        print(f"   ⚡ Inference time: {result['inference_time_ms']:.2f} ms")
        
        # Show confidence level with emoji
        confidence = result['ai_probability'] if result['is_ai_generated'] else result['real_probability']
        if confidence > 0.95:
            confidence_emoji = "🔥🔥🔥"
            confidence_text = "Extremely High"
        elif confidence > 0.9:
            confidence_emoji = "🔥🔥"
            confidence_text = "Very High"
        elif confidence > 0.8:
            confidence_emoji = "🔥"
            confidence_text = "High"
        elif confidence > 0.7:
            confidence_emoji = "👍"
            confidence_text = "Good"
        elif confidence > 0.6:
            confidence_emoji = "⚖️"
            confidence_text = "Moderate"
        else:
            confidence_emoji = "🤔"
            confidence_text = "Low - Model Unsure"
        
        print(f"   {confidence_emoji} Confidence: {confidence_text} ({confidence:.2%})")
        
        # For known images, we could check accuracy
        # This is just for learning - we're assuming the downloaded samples are correctly labeled
        if "real" in str(image_path).lower():
            expected_ai = False
            is_correct = not result['is_ai_generated']
            if is_correct:
                correct_predictions += 1
                print(f"   ✅ Correct! This should be a REAL image")
            else:
                print(f"   ❌ Incorrect! This should be a REAL image but model says AI")
        elif "ai" in str(image_path).lower():
            expected_ai = True
            is_correct = result['is_ai_generated']
            if is_correct:
                correct_predictions += 1
                print(f"   ✅ Correct! This should be an AI image")
            else:
                print(f"   ❌ Incorrect! This should be an AI image but model says REAL")
        
    else:
        print(f"   ❌ Detection failed for {image_path}")

print("\n" + "=" * 60)
print("✅ TEST COMPLETE")
print("=" * 60)

# Step 8: Summary
print("\n📊 SUMMARY STATISTICS:")       
if total_tests > 0:
    accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
    print(f"   📈 Accuracy on known samples: {accuracy:.1f}% ({correct_predictions}/{total_tests})")
