from fastapi import FastAPI, HTTPException
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
#Allows the frontend(from a different port) to access the backend (which is also in a different port)
#These requests are usually blocked by browsers for security reasons, but this overrides that
#This block is in place to prevent other malicious domains from accessing your backend and taking precious data muhhaha
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
#Reusing the Firebase service account for Vertix AI, just using one service account with both Firebase and Vertex AI enabled
vertex_credentials = service_account.Credentials.from_service_account_file(
    FIREBASE_CREDENTIALS_PATH
)
vertexai.init(
    project=PROJECT_ID, 
    location=LOCATION,
    credentials=vertex_credentials
)
model = GenerativeModel(MODEL_NAME)

# The 3 models use pydantic to validate data requests for each endpoint
#FastAPI is used to automatically checks data requests and matches it to the right model
class AuthRequest(BaseModel):
    email: str
    password: str

class InventoryItem(BaseModel):
    user_id: str
    name: str
    quantity: int
    expiration_date: Optional[str] = None

class ChatRequest(BaseModel):
    user_id: str
    message: str


# Auth Endpoints - Used for signing up and logging in users
#The @app.post here responds to POST requests from "/auth/signup" in this case (getting data from the frontend)
@app.post("/auth/signup")
async def signup(req: AuthRequest):
    try:
        #Tries to create a new user for firbase auth
        user = auth.create_user(email=req.email, password=req.password)
        return {"user": {"uid": user.uid, "email": user.email}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
async def login(req: AuthRequest):
    try:
        #Checks if user exists in firebase auth for now, but doesn't verify password(Probably ADD LATER )
        user = auth.get_user_by_email(req.email)
        return {"user": {"uid": user.uid, "email": user.email}}
    except Exception as e:
        #If DNE then the exception is raised and the user is informed of invalid credentials
        raise HTTPException(status_code=400, detail="Invalid credentials")

# Inventory Endpoints - Loads all the inventory items tied to the user
@app.get("/inventory/{user_id}")
async def get_inventory(user_id: str):
    #The items the user has in their inventory is stored locally in a list, and populated from the firestore database so we can fetch it for use at later times
    items = []
    #Getting the data from firestore where the user_id matches the user and putting it (streaming) in the items list
    docs = db.collection("inventory").where("user_id", "==", user_id).stream()
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        items.append(data)
    return items

@app.post("/inventory")
async def add_inventory(item: InventoryItem):
    #Creates a new inventory item/collection document in firestore for the user
    doc_ref = db.collection("inventory").document()
    #Each inventory item is a dictionary with these fields
    doc_ref.set({
        "user_id": item.user_id,
        "name": item.name,
        "quantity": item.quantity,
        "expiration_date": item.expiration_date,
        "created_at": datetime.now()
    })
    return {"id": doc_ref.id, "message": "Item added"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        # Get user's inventory
        #Stores the item with all of its data as a string in the inventory list
        inventory = []
        docs = db.collection("inventory").where("user_id", "==", req.user_id).stream()
        for doc in docs:
            data = doc.to_dict()
            #Using f-strings to get the name, quantity, and expiration date of each item in the inventory in a normalized way
            inventory.append(f"{data['name']} (qty: {data['quantity']}, expires: {data.get('expiration_date', 'N/A')})")
        
        # Build prompt
        #If inventory is empty, then the ternary statement fails and it says no items, otherwise it joins the items with new lines
        #This variable is used in the f-string for the prompt to Vertex AI
        inventory_text = "\n".join(inventory) if inventory else "No items in inventory"
        #We will be prompting Vertex AI with this prompt to get recipe suggestions based on the user's inventory
        prompt = f"""You are a helpful cooking assistant. 
        
Current inventory:
{inventory_text}

User question: {req.message}

Provide helpful recipe suggestions based on their available ingredients. Prioritize items that are expiring soon."""
        
        # Call Vertex AI with error handling
        #The prompt is sent to Vertex AI and the response is stored in response
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
        
        #Returns the response from Vertex AI to the frontend
        return {"response": response_text}
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")  # Debug log
        print(f"Error type: {type(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=BACKEND_PORT)