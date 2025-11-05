#!/usr/bin/env python3
"""
MINIMAL EXAMPLE: Use the trained grocery model in another project

Just copy this file and best.pt to your other project!
"""

from ultralytics import YOLO
from pathlib import Path

class GroceryClassifier:
    """Simple wrapper for grocery classification"""
    
    def __init__(self, model_path='best.pt'):
        """Load the trained model"""
        self.model = YOLO(model_path)
        print(f"✅ Model loaded from {model_path}")
        print(f"   Classes: {len(self.model.names)} items")
    
    def predict(self, image_path):
        """
        Classify a grocery item
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict with prediction results
        """
        results = self.model(image_path, verbose=False)
        result = results[0]
        
        # Get top 5 predictions
        top5_classes = []
        for idx in result.probs.top5:
            class_name = result.names[idx]
            confidence = result.probs.data[idx].item()
            top5_classes.append({
                'class': class_name,
                'confidence': confidence
            })
        
        return {
            'image': Path(image_path).name,
            'predicted_class': result.names[result.probs.top1],
            'confidence': result.probs.top1conf.item(),
            'top5': top5_classes
        }
    
    def get_all_classes(self):
        """Get list of all grocery items the model can recognize"""
        return list(self.model.names.values())


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = GroceryClassifier('runs/classify/grocery-classify6/weights/best.pt')
    
    # Show what items it can recognize
    print("\n📦 Can recognize these items:")
    for item in classifier.get_all_classes():
        print(f"   - {item}")
    
    # Classify an image
    print("\n🔍 Testing on sample image...")
    result = classifier.predict('dataset/GroceryStoreDataset/dataset_flat/test/Banana/Banana_001.jpg')
    
    print(f"\n✅ Results for {result['image']}:")
    print(f"   Predicted: {result['predicted_class']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"\n   Top 5 predictions:")
    for pred in result['top5']:
        print(f"      {pred['class']}: {pred['confidence']:.1%}")

