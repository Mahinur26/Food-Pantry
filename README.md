# Food Pantry

An AI-driven food pantry management app that lets you organize groceries, keep track of expiration dates, and receive recipe recommendations based on what you have on hand. It uses smart image recognition to automatically detect and catalog food items through your device’s camera.

## Features


Uploading Video of using the image recognition model.mp4…


### 🔐 User Authentication
Secure login and signup system powered by Firebase Authentication.

![Login Screen](./Screenshot%20of%20login.png)

### 📦 Smart Inventory Management
- Add items manually or scan them with your camera
- Automatic category detection for fruits, vegetables, dairy, meat, grains, and more
- Track expiration dates with color-coded warnings:
  - 🔴 Red: 3 days or less until expiration
  - 🟡 Yellow: 4-7 days until expiration
  - ⚪ White: More than 7 days
- Organize items by category
- Edit and delete items with confirmation dialogs

![Main Inventory View](./Screenshot%20of%20main%20window.png)

### 📸 Image Recognition
Use your device camera or upload photos to automatically identify grocery items. Powered by a fine-tuned YOLO model that recognizes 36 different fruits and vegetables.

<div align="center">

https://github.com/user-attachments/assets/8795c2f2-36ff-4c0f-b087-c9898156a41b

</div>

### 🤖 AI Recipe Assistant
Chat with an AI powered by Google's Gemini model to get personalized recipe suggestions based on your current inventory. Get creative meal ideas that help reduce food waste.

![AI Chatbot](./Screenshot%20of%20chatbot.png)

## Tech Stack

### Frontend
- **React** - UI framework
- **Vite** - Build tool and dev server
- **Material-UI** - Component library
- **Tailwind CSS** - Utility-first styling

### Backend
- **FastAPI** - Modern Python web framework
- **Firebase** - Authentication and Firestore database
- **Google Vertex AI** - Gemini model for recipe generation
- **Hugging Face Transformers** - YOLO model for image classification
- **PyTorch** - Deep learning framework

## Development Setup

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Environment Variables
Create a `.env` file in the `backend/` directory:
```
FIREBASE_CREDENTIALS=<your-firebase-credentials-json>
GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
```

## Deployment

- **Frontend**: Deployed on Vercel (see `vercel.json`)
- **Backend**: Deployed on Render (see `backend/render.yaml`)

## How It Works

1. **Sign up or log in** to create your personal pantry
2. **Add items** by manually entering details or scanning with your camera
3. **Monitor expiration dates** with automatic color-coded warnings
4. **Get recipe ideas** by chatting with the AI assistant about what to cook
5. **Reduce food waste** by using items before they expire
