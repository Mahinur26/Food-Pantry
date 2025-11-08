from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import firebase_admin
from firebase_admin import credentials, firestore, auth
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from datetime import datetime
import os
from dotenv import load_dotenv
from google.oauth2 import service_account


from transformers import AutoModelForImageClassification
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
from difflib import get_close_matches


# Load environment variables
load_dotenv()




# Get configuration from .env
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("VERTEX_AI_LOCATION")
MODEL_NAME = os.getenv("VERTEX_AI_MODEL")
FRONTEND_URL = os.getenv("FRONTEND_URL")
BACKEND_PORT = int(os.getenv("BACKEND_PORT"))




# Initialize FastAPI
app = FastAPI()


# CORS - Cross-Origin Resource Sharing
# Allows the frontend(from a different port) to access the backend (which is also in a different port)
# These requests are usually blocked by browsers for security reasons, but this overrides that
# This block is in place to prevent other malicious domains from accessing your backend and taking precious data muhhaha
app.add_middleware(
   CORSMiddleware,
   allow_origins=[FRONTEND_URL],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)


# Initialize Firebase creds and gets a database client
cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()


# Initialize Vertex AI with explicit credentials
# Reusing the Firebase service account for Vertix AI, just using one service account with both Firebase and Vertex AI enabled
vertex_credentials = service_account.Credentials.from_service_account_file(
   FIREBASE_CREDENTIALS_PATH
)
vertexai.init(
   project=PROJECT_ID,
   location=LOCATION,
   credentials=vertex_credentials
)
model = GenerativeModel(MODEL_NAME)


# Loads the grocery classification model at startup - uses images to classify mostly fruits and veggies atm
grocery_model = None
preprocess_transform = None




@app.on_event("startup")
async def load_grocery_model():
   """Load the grocery classification model when server starts"""
   global grocery_model, preprocess_transform
   try:
       model_name = "jazzmacedo/fruits-and-vegetables-detector-36"
       print(
           f"🔄 Loading grocery classification model from Hugging Face: {model_name}...")
       print("   (First run will download ~23.6MB model, this may take a moment...)")


       # Load the model from Hugging Face
       grocery_model = AutoModelForImageClassification.from_pretrained(
           model_name)
       grocery_model.eval()  # Set to evaluation mode


       # Create preprocessing pipeline matching ImageNet normalization
       preprocess_transform = transforms.Compose([
           transforms.Resize((224, 224)),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                0.229, 0.224, 0.225])
       ])


       # Extract labels from model config
       labels = list(grocery_model.config.id2label.values())


       print("✅ Grocery classification model loaded successfully")
       print(f"   Can recognize {len(labels)} items")
       print(f"   Model type: {type(grocery_model).__name__}")
       # Print first few class names as verification
       if labels:
           sample_classes = labels[:5]
           print(f"   Sample classes: {', '.join(sample_classes)}...")
   except Exception as e:
       print(f"⚠️  Warning: Could not load grocery model: {e}")
       print(f"   Error type: {type(e).__name__}")
       print("   Image classification will not be available")
       print("   Note: Model requires internet connection for first download")
       grocery_model = None
       preprocess_transform = None


# Category mapping dictionary
CATEGORY_MAPPING = {
   "fruits": [
       "apple", "apples", "banana", "bananas", "orange", "oranges", "strawberry", "strawberries",
       "grape", "grapes", "watermelon", "melon", "pineapple", "mango", "mangos", "kiwi", "kiwis",
       "peach", "peaches", "pear", "pears", "plum", "plums", "cherry", "cherries", "blueberry",
       "blueberries", "raspberry", "raspberries", "blackberry", "blackberries", "cranberry",
       "cranberries", "lemon", "lemons", "lime", "limes", "grapefruit", "avocado", "avocados",
       "papaya", "coconut", "coconuts", "pomegranate", "fig", "figs", "date", "dates",
       "apricot", "apricots", "nectarine", "nectarines", "tangerine", "tangerines", "clementine",
       "clementines", "mandarin", "mandarins", "dragon fruit", "passion fruit", "lychee",
       "guava", "star fruit", "persimmon", "persimmons"
   ],
   "vegetables": [
       "tomato", "tomatoes", "lettuce", "spinach", "cucumber", "cucumbers", "carrot", "carrots",
       "broccoli", "cauliflower", "bell pepper", "bell peppers", "pepper", "peppers", "onion",
       "onions", "garlic", "potato", "potatoes", "sweet potato", "sweet potatoes", "corn",
       "peas", "green beans", "beans", "zucchini", "eggplant", "eggplants", "squash",
       "mushroom", "mushrooms", "cabbage", "kale", "celery", "asparagus", "brussels sprouts",
       "brussel sprouts", "radish", "radishes", "beet", "beets", "turnip", "turnips",
       "artichoke", "artichokes", "leek", "leeks", "scallion", "scallions", "green onion",
       "green onions", "shallot", "shallots", "bok choy", "chard", "arugula", "romaine",
       "iceberg", "butter lettuce", "red cabbage", "purple cabbage", "savoy cabbage",
       "yellow squash", "butternut squash", "acorn squash", "pumpkin", "pumpkins", "okra",
       "collard greens", "mustard greens", "endive", "fennel", "jicama"
   ],
   "dairy": [
       "milk", "cheese", "yogurt", "yoghurt", "butter", "cream", "sour cream", "cottage cheese",
       "cream cheese", "mozzarella", "cheddar", "swiss cheese", "parmesan", "feta", "ricotta",
       "gouda", "brie", "camembert", "blue cheese", "goat cheese", "provolone", "pepper jack",
       "monterey jack", "american cheese", "heavy cream", "half and half", "whipping cream",
       "buttermilk", "greek yogurt", "greek yoghurt", "greek style yogurt", "skim milk",
       "whole milk", "2% milk", "1% milk", "almond milk", "soy milk", "oat milk", "coconut milk",
       "ice cream", "gelato", "sorbet", "frozen yogurt"
   ],
   "meat": [
       "chicken", "beef", "pork", "turkey", "lamb", "duck", "ham", "bacon", "sausage", "sausages",
       "ground beef", "ground turkey", "ground pork", "ground chicken", "chicken breast",
       "chicken thighs", "chicken wings", "chicken drumsticks", "steak", "ribs", "pork chops",
       "pork tenderloin", "roast beef", "pot roast", "meatballs", "meatball", "hot dog",
       "hot dogs", "hamburger", "hamburgers", "burger", "burgers", "pepperoni", "salami",
       "prosciutto", "chorizo", "bratwurst", "kielbasa", "pastrami", "corned beef",
       "roast chicken", "roasted chicken", "chicken leg", "chicken legs", "chicken wing",
       "drumstick", "drumsticks", "ribeye", "sirloin", "tenderloin", "filet mignon", "brisket", "short ribs"
   ],
   "grains": [
       "rice", "bread", "pasta", "spaghetti", "noodles", "flour", "oats", "oatmeal", "quinoa",
       "barley", "wheat", "rye", "cornmeal", "couscous", "bulgur", "buckwheat", "millet",
       "farro", "wild rice", "brown rice", "white rice", "jasmine rice", "basmati rice",
       "sushi rice", "arborio rice", "bread loaf", "sandwich bread", "whole wheat bread",
       "white bread", "sourdough", "baguette", "rolls", "buns", "pita", "tortilla", "tortillas",
       "naan", "crackers", "cereal", "granola", "cereal bars", "rice cakes", "pretzels",
       "macaroni", "penne", "fettuccine", "linguine", "rigatoni", "fusilli", "ravioli",
       "lasagna", "lasagne", "ramen", "udon", "soba", "rice noodles", "egg noodles"
   ]
}


# Valid categories
VALID_CATEGORIES = ["Fruits", "Vegetables", "Dairy", "Meat", "Grains", "Other"]




def detect_category(item_name: str) -> str:
    # docstring explaining the function
   """
   Detect category of an item based on its name using predefined mapping.
   Returns category name (capitalized) or "Other" if no match found.
   """
   if not item_name:
       return "Other"


   item_lower = item_name.lower().strip()
   
   #handle plural words
   if item_lower.endswith('es'):
        item_lower = item_lower[:-2]
   elif item_lower.endswith('s'):
        item_lower = item_lower[:-1]


   # Check each category's keywords
   for category, keywords in CATEGORY_MAPPING.items():
       for keyword in keywords:
           if keyword in item_lower:
               # Capitalize first letter of category
               return category.capitalize()
           # Fuzzy match (handles typos like "strawbery" -> "strawberry")
           close = get_close_matches(item_lower, keywords, n=1, cutoff=0.8)
           if close:
                return category.capitalize() 
   

   return "Other"


# The 3 models use pydantic to validate data requests for each endpoint
# FastAPI is used to automatically checks data requests and matches it to the right model



# base model means json file, with the category field below
class AuthRequest(BaseModel):
   email: str
   password: str




class InventoryItem(BaseModel):
   user_id: str
   name: str
   quantity: int
   expiration_date: Optional[str] = None
   category: Optional[str] = None




class ChatRequest(BaseModel):
   user_id: str
   message: str




class DeleteItemRequest(BaseModel):
   item_id: str



class UpdateCategoryRequest(BaseModel):
   category: str


# Model for adding item with image




class ClassifyAndAddRequest(BaseModel):
   user_id: str
   quantity: int = 1
   expiration_date: Optional[str] = None


# NEW ENDPOINTS FOR GROCERY CLASSIFICATION




@app.post("/classify/image")
async def classify_grocery_image(file: UploadFile = File(...)):
   """
   Classify a grocery item from an uploaded image
   Returns the predicted item name and confidence
   """
   if not grocery_model or not preprocess_transform:
       raise HTTPException(
           status_code=503, detail="Grocery classification model not available")


   # Validate file type
   if not file.content_type.startswith('image/'):
       raise HTTPException(status_code=400, detail="File must be an image")


   try:
       # Read and preprocess image
       contents = await file.read()
       image = Image.open(io.BytesIO(contents))


       # Convert to RGB if necessary (handles RGBA, L, etc.)
       if image.mode != 'RGB':
           image = image.convert('RGB')


       # Preprocess image for ResNet-50
       input_tensor = preprocess_transform(
           image).unsqueeze(0)  # Add batch dimension


       # Run inference with the transformers model
       with torch.no_grad():
           outputs = grocery_model(input_tensor)


       # Apply softmax to get probabilities
       probs = F.softmax(outputs.logits, dim=1)


       # Get top prediction
       predicted_idx = torch.argmax(probs, dim=1).item()
       predicted_item = grocery_model.config.id2label[predicted_idx]
       confidence = float(probs[0][predicted_idx].item())


       # Get top 3 predictions for alternatives
       top3_indices = torch.topk(probs, k=3, dim=1).indices[0]
       top3 = []
       for idx in top3_indices:
           idx_item = idx.item()
           top3.append({
               "name": grocery_model.config.id2label[idx_item],
               "confidence": float(probs[0][idx_item].item())
           })


       print(
           f"✅ Classification result: {predicted_item} (confidence: {confidence:.2%})")


       return {
           "success": True,
           "predicted_item": predicted_item,
           "confidence": confidence,
           "alternatives": top3
       }


   except Exception as e:
       print(f"❌ Classification error: {str(e)}")
       print(f"   Error type: {type(e)}")
       raise HTTPException(
           status_code=500, detail=f"Classification failed: {str(e)}")




@app.post("/inventory/add-from-image")
async def add_inventory_from_image(
   file: UploadFile = File(...),
   user_id: str = Form(...),
   quantity: int = Form(1),
   expiration_date: Optional[str] = Form(None)
):
   """
   Classify a grocery image AND automatically add it to inventory
   This is the magic endpoint that combines classification + inventory addition!
   """
   if not grocery_model or not preprocess_transform:
       raise HTTPException(
           status_code=503, detail="Grocery classification model not available")


   try:
       # Validate inputs
       if not user_id or user_id.strip() == "":
           raise HTTPException(status_code=400, detail="user_id is required")


       # Ensure quantity is an integer
       try:
           quantity_int = int(quantity)
           if quantity_int < 1:
               raise ValueError("Quantity must be at least 1")
       except (ValueError, TypeError):
           raise HTTPException(
               status_code=400, detail="Quantity must be a valid positive integer")


       # Normalize expiration_date (empty string becomes None to match regular add behavior)
       expiration_date_normalized = expiration_date if expiration_date and expiration_date.strip() else None


       # 1. Classify the image
       contents = await file.read()
       image = Image.open(io.BytesIO(contents))


       # Convert to RGB if necessary
       if image.mode != 'RGB':
           image = image.convert('RGB')


       # Preprocess image for ResNet-50
       input_tensor = preprocess_transform(
           image).unsqueeze(0)  # Add batch dimension


       # Run inference with the transformers model
       with torch.no_grad():
           outputs = grocery_model(input_tensor)


       # Apply softmax to get probabilities
       probs = F.softmax(outputs.logits, dim=1)


       # Get top prediction
       predicted_idx = torch.argmax(probs, dim=1).item()
       predicted_name = grocery_model.config.id2label[predicted_idx]
       confidence = float(probs[0][predicted_idx].item())


       print(
           f"✅ Classification result: {predicted_name} (confidence: {confidence:.2%})")


       # 2. Detect category for the predicted item
       category = detect_category(predicted_name)


       # 3. Add to inventory (matching the structure from regular add_inventory endpoint)
       doc_ref = db.collection("inventory").document()
       doc_ref.set({
           "user_id": user_id,
           "name": predicted_name,
           "quantity": quantity_int,
           "expiration_date": expiration_date_normalized,
           "category": category,
           "created_at": datetime.now(),
           "added_by": "image_classification",
           "confidence": confidence
       })


       print(f"✅ Added {predicted_name} to inventory for user {user_id}")


       return {
           "success": True,
           "item_id": doc_ref.id,
           "item_name": predicted_name,
           "confidence": confidence,
           "message": f"Added {predicted_name} to inventory"
       }


   except Exception as e:
       print(f"❌ Error adding item from image: {str(e)}")
       print(f"   Error type: {type(e)}")
       raise HTTPException(
           status_code=500, detail=f"Failed to add item: {str(e)}")




# Auth Endpoints - Used for signing up and logging in users
# The @app.post here responds to POST requests from "/auth/signup" in this case (getting data from the frontend)
@app.post("/auth/signup")
async def signup(req: AuthRequest):
   try:
       # Tries to create a new user for firbase auth
       user = auth.create_user(email=req.email, password=req.password)
       return {"user": {"uid": user.uid, "email": user.email}}
   except Exception as e:
       raise HTTPException(status_code=400, detail=str(e))




@app.post("/auth/login")
async def login(req: AuthRequest):
   try:
       # Checks if user exists in firebase auth for now, but doesn't verify password(Probably ADD LATER )
       user = auth.get_user_by_email(req.email)
       return {"user": {"uid": user.uid, "email": user.email}}
   except Exception as e:
       # If DNE then the exception is raised and the user is informed of invalid credentials
       raise HTTPException(status_code=400, detail="Invalid credentials")


# Inventory Endpoints - Loads all the inventory items tied to the user




@app.get("/inventory/{user_id}")
async def get_inventory(user_id: str):
   # The items the user has in their inventory is stored locally in a list, and populated from the firestore database so we can fetch it for use at later times
   items = []
   # Getting the data from firestore where the user_id matches the user and putting it (streaming) in the items list
   docs = db.collection("inventory").where("user_id", "==", user_id).stream()
   for doc in docs:
       data = doc.to_dict()
       data["id"] = doc.id
       items.append(data)
   return items




@app.post("/inventory")
async def add_inventory(item: InventoryItem):
   # Creates a new inventory item/collection document in firestore for the user
   # Auto-detect category if not provided
   category = item.category if item.category else detect_category(item.name)


   doc_ref = db.collection("inventory").document()
   # Each inventory item is a dictionary with these fields
   doc_ref.set({
       "user_id": item.user_id,
       "name": item.name,
       "quantity": item.quantity,
       "expiration_date": item.expiration_date,
       "category": category,
       "created_at": datetime.now()
   })
   return {"id": doc_ref.id, "message": "Item added", "category": category}



# post request means sending data from the frontend to the backend for processing, in this case for deleting an item
@app.post("/inventory/delete")
async def delete_inventory(req: DeleteItemRequest):
   try:
       # Permanently deletes the inventory item from Firestore using its document ID
       db.collection("inventory").document(req.item_id).delete()
       return {"message": "Item deleted successfully"}
   except Exception as e:
       raise HTTPException(
           status_code=400, detail=f"Error deleting item: {str(e)}")



# put request means updating data in the backend, in this case for updating the category of an item
@app.put("/inventory/{item_id}/category")
async def update_category(item_id: str, req: UpdateCategoryRequest):
   """
   Update the category of an inventory item.
   Validates that the category is one of the allowed values.
   """
   try:
       # Validate category
       if req.category not in VALID_CATEGORIES:
           raise HTTPException(
               status_code=400,
               detail=f"Invalid category. Must be one of: {', '.join(VALID_CATEGORIES)}"
           )


       # Update the document
       doc_ref = db.collection("inventory").document(item_id)
       doc = doc_ref.get()


       if not doc.exists:
           raise HTTPException(status_code=404, detail="Item not found")


       doc_ref.update({"category": req.category})
       return {"message": "Category updated successfully", "category": req.category}
   except HTTPException:
       raise
   except Exception as e:
       raise HTTPException(
           status_code=400, detail=f"Error updating category: {str(e)}")




@app.post("/chat")
async def chat(req: ChatRequest):
   try:
       # Get user's inventory
       # Stores the item with all of its data as a string in the inventory list
       inventory = []
       docs = db.collection("inventory").where(
           "user_id", "==", req.user_id).stream()
       for doc in docs:
           data = doc.to_dict()
           # Using f-strings to get the name, quantity, and expiration date of each item in the inventory in a normalized way
           inventory.append(
               f"{data['name']} (qty: {data['quantity']}, expires: {data.get('expiration_date', 'N/A')})")


       # Build prompt
       # If inventory is empty, then the ternary statement fails and it says no items, otherwise it joins the items with new lines
       # This variable is used in the f-string for the prompt to Vertex AI
       inventory_text = "\n".join(
           inventory) if inventory else "No items in inventory"
       # We will be prompting Vertex AI with this prompt to get recipe suggestions based on the user's inventory
       prompt = f"""You are a helpful cooking assistant.
       
      
Current inventory:
{inventory_text}


User question: {req.message}


Provide helpful recipe suggestions based on their available ingredients. Prioritize items that are expiring soon."""


       # Call Vertex AI with error handling
       # The prompt is sent to Vertex AI and the response is stored in response
       print(f"Sending prompt to Vertex AI: {prompt[:100]}...")  # Debug log
       response = model.generate_content(prompt)


       # Try different ways to access the response
       if hasattr(response, 'text'):
           response_text = response.text
       elif hasattr(response, 'candidates') and response.candidates:
           response_text = response.candidates[0].content.parts[0].text
       else:
           print(f"Response object: {response}")  # Debug log
           response_text = str(response)


       # Returns the response from Vertex AI to the frontend
       return {"response": response_text}
   except Exception as e:
       print(f"Error in chat endpoint: {str(e)}")  # Debug log
       print(f"Error type: {type(e)}")  # Debug log
       raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=BACKEND_PORT)