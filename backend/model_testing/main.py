from predict import GroceryClassifier

# Load model
classifier = GroceryClassifier('best.pt')

# Classify images
result = classifier.predict('Banana_029.jpg')
print(f"This is a {result['predicted_class']}")