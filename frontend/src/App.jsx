import React, { useState, useEffect, useRef } from "react";
import { Snackbar, Alert, Dialog, DialogTitle, DialogContent, DialogActions, Button, Typography } from "@mui/material";

//Chaged the api url to accept both the local host and deployed url
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";


function App() {
 //The const variables handle the state manegment of key info- ensuring that the ui reflects the variables accurately
 const [user, setUser] = useState(null);
 const [email, setEmail] = useState("");
 const [password, setPassword] = useState("");
 const [isLogin, setIsLogin] = useState(true);
 const [activeTab, setActiveTab] = useState("inventory");


 const [inventory, setInventory] = useState([]);
 const [itemName, setItemName] = useState("");
 const [itemQuantity, setItemQuantity] = useState("");
 const [itemExpiration, setItemExpiration] = useState("");
 const [itemCategory, setItemCategory] = useState(""); // Optional category override


 // Notification state (for expiration warnings)
 const [notification, setNotification] = useState(""); // The message to show
 const [open, setOpen] = useState(false); // Whether the notification is visible

 // Alert state (for general notifications)
 const [alertState, setAlertState] = useState({
   open: false,
   message: '',
   severity: 'info' // 'error' | 'warning' | 'info' | 'success'
 });

 // Delete confirmation dialog state
 const [deleteDialog, setDeleteDialog] = useState({
   open: false,
   itemId: null,
   itemName: ''
 });

 const [chatMessages, setChatMessages] = useState([]);
 const [chatInput, setChatInput] = useState("");
 const [loading, setLoading] = useState(false);


 // Scan groceries state
 const [scanningMode, setScanningMode] = useState(false);
 const [selectedImage, setSelectedImage] = useState(null);
 const [imagePreview, setImagePreview] = useState(null);
 const [predictedItem, setPredictedItem] = useState(null);
 const [scanQuantity, setScanQuantity] = useState("1");
 const [scanExpiration, setScanExpiration] = useState("");
 const [scanning, setScanning] = useState(false);
 const fileInputRef = useRef(null);


 useEffect(() => {
   const stored = localStorage.getItem("bullavor_user");
   if (stored) setUser(JSON.parse(stored));
 }, []);


 useEffect(() => {
   //fetch inventory when user logs in
   if (user) fetchInventory();
 }, [user]);


 // Cleanup image preview URL on unmount or when closing scan mode
 useEffect(() => {
   return () => {
     if (imagePreview) {
       URL.revokeObjectURL(imagePreview);
     }
   };
 }, [imagePreview]);

 // Helper function to show alerts
 const showAlert = (message, severity = 'info') => {
   setAlertState({ open: true, message, severity });
 };

 // Helper function to close alerts
 const handleAlertClose = (event, reason) => {
   if (reason === 'clickaway') return;
   setAlertState({ ...alertState, open: false });
 };

 //Sends a POST request to the backend to either log in or sign up the user based on the isLogin state
 //A POST request means sending data from the frontend to the backend for processing, in this case for logging/signing in
 const handleAuth = async () => {
   setLoading(true);
   try {
     const res = await fetch(
       `${API_URL}/auth/${isLogin ? "login" : "signup"}`,
       {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ email, password }),
       }
     );
     const data = await res.json();
     //If the response is successful, sets the user to both state and local storage
     if (res.ok) {
       setUser(data.user);
       localStorage.setItem("bullavor_user", JSON.stringify(data.user));
       showAlert(isLogin ? 'Welcome back!' : 'Account created successfully!', 'success');
     } else {
       showAlert(data.detail || 'Authentication failed', 'error');
     }
   } catch (err) {
     showAlert('Error: ' + err.message, 'error');
   }
   setLoading(false);
 };


 //function that fetches the inventory of the user that is currently logged in - items in their pantry
 const fetchInventory = async () => {
   try {
     const res = await fetch(`${API_URL}/inventory/${user.uid}`);
     const data = await res.json();
     setInventory(data);
   } catch (err) {
     console.error(err);
   }
 };


 useEffect(() => {
   if (inventory.length === 0) return;


   const today = new Date();
   const expiringSoon = inventory.filter((item) => {
     if (!item.expiration_date) return false;
     const expDate = new Date(item.expiration_date);
     const diffDays = (expDate - today) / (1000 * 60 * 60 * 24);
     return diffDays <= 3 && diffDays >= 0; // expires in 3 days or less
   });


   if (expiringSoon.length > 0) {
     setNotification(
       `⚠️ Heads up! These items are expiring soon: ${expiringSoon
         .map((i) => i.name)
         .join(", ")}`
     );
     setOpen(true);
   }
 }, [inventory]);


 //Sends the new item data to the backend, then refreshes the inventory list to show the new item
 const addItem = async () => {
   if (!itemName || !itemQuantity) {
     showAlert("Name and quantity required", "warning");
     return;
   }
   setLoading(true);
   try {
     await fetch(`${API_URL}/inventory`, {
       method: "POST",
       headers: { "Content-Type": "application/json" },
       body: JSON.stringify({
         user_id: user.uid,
         name: itemName,
         quantity: parseInt(itemQuantity),
         expiration_date: itemExpiration || null,
         category: itemCategory || null, // Send category if manually selected, otherwise let backend auto-detect
       }),
     });
     setItemName("");
     setItemQuantity("");
     setItemExpiration("");
     setItemCategory("");
     await fetchInventory();
     showAlert("Item added successfully!", "success");
   } catch (err) {
     showAlert("Error adding item", "error");
   }
   setLoading(false);
 };


 //Opens the delete confirmation dialog
 const openDeleteDialog = (itemId, itemName) => {
   setDeleteDialog({ open: true, itemId, itemName });
 };

 //Closes the delete confirmation dialog
 const closeDeleteDialog = () => {
   setDeleteDialog({ open: false, itemId: null, itemName: '' });
 };

 //Deletes an item from the inventory by sending a POST request to the backend and updating the UI
 const confirmDelete = async () => {
   const itemId = deleteDialog.itemId;
   closeDeleteDialog();
   setLoading(true);
   try {
     await fetch(`${API_URL}/inventory/delete`, {
       method: "POST",
       headers: { "Content-Type": "application/json" },
       body: JSON.stringify({ item_id: itemId }),
     });
     await fetchInventory(); // Refresh the inventory list
     showAlert("Item deleted successfully", "success");
   } catch (err) {
     showAlert("Error deleting item", "error");
   }
   setLoading(false);
 };


 // Update category for an item
 const updateCategory = async (itemId, newCategory) => {
   setLoading(true);
   try {
     await fetch(`${API_URL}/inventory/${itemId}/category`, {
       method: "PUT",
       headers: { "Content-Type": "application/json" },
       body: JSON.stringify({ category: newCategory }),
     });
     await fetchInventory(); // Refresh the inventory list
     showAlert("Category updated successfully", "success");
   } catch (err) {
     showAlert("Error updating category", "error");
   }
   setLoading(false);
 };

 // Helper function to get expiration color based on days remaining
 const getExpirationColor = (expirationDate) => {
   if (!expirationDate) return "text-gray-600";
   
   const today = new Date();
   const expDate = new Date(expirationDate);
   const diffDays = Math.ceil((expDate - today) / (1000 * 60 * 60 * 24));
   
   if (diffDays <= 3 && diffDays >= 0) {
     return "text-red-600 font-semibold"; // Expires in 3 days or less
   } else if (diffDays >= 4 && diffDays <= 7) {
     return "text-yellow-600 font-semibold"; // Expires in 4-7 days
   }
   return "text-gray-600"; // More than 7 days or already expired
 };

 // Group inventory items by category
 const groupInventoryByCategory = (items) => {
   const grouped = {};
   const categoryOrder = [
     "Fruits",
     "Vegetables",
     "Dairy",
     "Meat",
     "Grains",
     "Other",
   ];


   items.forEach((item) => {
     const category = item.category || "Other";
     if (!grouped[category]) {
       grouped[category] = [];
     }
     grouped[category].push(item);
   });


   // Sort categories according to predefined order
   const sorted = {};
   categoryOrder.forEach((cat) => {
     if (grouped[cat]) {
       sorted[cat] = grouped[cat];
     }
   });


   // Add any categories not in the predefined list
   Object.keys(grouped).forEach((cat) => {
     if (!categoryOrder.includes(cat)) {
       sorted[cat] = grouped[cat];
     }
   });


   return sorted;
 };

 // Parse recipe response into structured sections
 const parseRecipeResponse = (text) => {
   const sections = {
     dishName: '',
     ingredients: [],
     instructions: [],
     rawText: text
   };
   
   // Extract dish name
   const dishMatch = text.match(/DISH NAME:\s*(.+?)(?=\n|INGREDIENTS:|$)/i);
   if (dishMatch) sections.dishName = dishMatch[1].trim();
   
   // Extract ingredients
   const ingredientsMatch = text.match(/INGREDIENTS:\s*([\s\S]+?)(?=INSTRUCTIONS:|$)/i);
   if (ingredientsMatch) {
     sections.ingredients = ingredientsMatch[1]
       .split('\n')
       .filter(line => line.trim().startsWith('-'))
       .map(line => line.trim().substring(1).trim());
   }
   
   // Extract instructions
   const instructionsMatch = text.match(/INSTRUCTIONS:\s*([\s\S]+?)$/i);
   if (instructionsMatch) {
     sections.instructions = instructionsMatch[1]
       .split('\n')
       .filter(line => /^\d+\./.test(line.trim()))
       .map(line => line.trim());
   }
   
   return sections;
 };

 //Used to send the user's message to the ai model in the backend and return the response in the chat UI
 const sendMessage = async () => {
   if (!chatInput.trim()) return;


   setChatMessages([...chatMessages, { role: "user", content: chatInput }]);
   const msg = chatInput;
   setChatInput("");
   setLoading(true);


   try {
     const res = await fetch(`${API_URL}/chat`, {
       method: "POST",
       headers: { "Content-Type": "application/json" },
       body: JSON.stringify({ user_id: user.uid, message: msg }),
     });
     const data = await res.json();
     setChatMessages((prev) => [
       ...prev,
       { role: "assistant", content: data.response },
     ]);
   } catch (err) {
     setChatMessages((prev) => [
       ...prev,
       { role: "assistant", content: "Error occurred" },
     ]);
     showAlert("Chat error occurred", "error");
   }
   setLoading(false);
 };


 //Handle image selection from file input (works for both camera capture and file upload)
 const handleImageSelect = async (e) => {
   const file = e.target.files[0];
   if (!file) return;


   // Create preview URL
   const previewUrl = URL.createObjectURL(file);
   setSelectedImage(file);
   setImagePreview(previewUrl);
   setPredictedItem(null); // Clear previous prediction


   // Auto-trigger classification
   await classifyImage(file);
 };


 //Classify the selected image using the backend YOLO model
 const classifyImage = async (imageFile) => {
   setScanning(true);
   try {
     const formData = new FormData();
     formData.append("file", imageFile);


     const res = await fetch(`${API_URL}/classify/image`, {
       method: "POST",
       body: formData,
     });


     if (!res.ok) {
       const errorData = await res.json();
       throw new Error(errorData.detail || "Classification failed");
     }


     const data = await res.json();
     setPredictedItem({
       name: data.predicted_item,
       confidence: data.confidence,
     });
   } catch (err) {
     showAlert("Error classifying image: " + err.message, "error");
     setPredictedItem(null);
   }
   setScanning(false);
 };


 //Add item to inventory from scanned image
 const addItemFromScan = async () => {
   if (!selectedImage || !predictedItem) {
     showAlert("Please select and classify an image first", "warning");
     return;
   }
   if (!scanQuantity || parseInt(scanQuantity) < 1) {
     showAlert("Please enter a valid quantity (at least 1)", "warning");
     return;
   }


   setLoading(true);
   try {
     const formData = new FormData();
     formData.append("file", selectedImage);
     formData.append("user_id", user.uid);
     formData.append("quantity", scanQuantity.toString()); // Ensure it's a string for FormData
     if (scanExpiration && scanExpiration.trim()) {
       formData.append("expiration_date", scanExpiration);
     }


     const res = await fetch(`${API_URL}/inventory/add-from-image`, {
       method: "POST",
       body: formData,
     });


     if (!res.ok) {
       const errorData = await res.json();
       throw new Error(errorData.detail || "Failed to add item");
     }


     const responseData = await res.json();


     // Reset scan UI
     if (imagePreview) {
       URL.revokeObjectURL(imagePreview);
     }
     if (fileInputRef.current) {
       fileInputRef.current.value = "";
     }
     setSelectedImage(null);
     setImagePreview(null);
     setPredictedItem(null);
     setScanQuantity("1");
     setScanExpiration("");
     setScanningMode(false);


     // Refresh inventory
     await fetchInventory();


     showAlert(
       `Added ${responseData.item_name} to inventory!`,
       "success"
     );
   } catch (err) {
     showAlert("Error adding item: " + err.message, "error");
   }
   setLoading(false);
 };


  //Reset scan UI when closing
  const resetScanUI = () => {
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    setScanningMode(false);
    setSelectedImage(null);
    setImagePreview(null);
    setPredictedItem(null);
    setScanQuantity('1');
    setScanExpiration('');
  };
//The main return statement that renders the UI based on whether the user is logged in or not
return (
  <>
    {/* Unified Alert System - Always rendered */}
    <Snackbar
      open={alertState.open}
      autoHideDuration={6000}
      onClose={handleAlertClose}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
    >
      <Alert
        onClose={handleAlertClose}
        severity={alertState.severity}
        variant="filled"
        sx={{ width: '100%' }}
      >
        {alertState.message}
      </Alert>
    </Snackbar>

    {/* Delete Confirmation Dialog */}
    <Dialog
      open={deleteDialog.open}
      onClose={closeDeleteDialog}
      PaperProps={{
        sx: {
          borderRadius: '20px',
          padding: '12px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.12)'
        }
      }}
    >
      <DialogTitle sx={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#15803d' }}>
        Delete Item
      </DialogTitle>

      <DialogContent>
        <Typography sx={{ fontSize: '1rem', color: '#4b5563' }}>
          Are you sure you want to delete <strong>{deleteDialog.itemName}</strong>? This action cannot be undone.
        </Typography>
      </DialogContent>

      <DialogActions sx={{ padding: '16px 24px' }}>
        <Button 
          onClick={closeDeleteDialog} 
          sx={{ 
            color: '#6b7280',
            fontWeight: 600,
            textTransform: 'none',
            fontSize: '1rem',
            '&:hover': { backgroundColor: '#f3f4f6' }
          }}
        >
          Cancel
        </Button>
        <Button 
          onClick={confirmDelete} 
          variant="contained"
          sx={{ 
            backgroundColor: '#dc2626',
            fontWeight: 600,
            textTransform: 'none',
            fontSize: '1rem',
            borderRadius: '12px',
            paddingX: '24px',
            '&:hover': { backgroundColor: '#b91c1c' }
          }}
        >
          Delete
        </Button>
      </DialogActions>
    </Dialog>

    {!user ? (
      <div className="min-h-screen bg-green-50 flex items-center justify-center p-6">
        <div className="bg-white rounded-2xl shadow-2xl p-10 w-full max-w-md">
          <h1 className="text-4xl font-extrabold text-green-700 mb-8 text-center">Food Pantry</h1>
          <div className="space-y-5">
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleAuth()}
              className="w-full px-5 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500"
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleAuth()}
              className="w-full px-5 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500"
            />
            <button
              onClick={handleAuth}
              disabled={loading}
              className="w-full bg-green-600 text-white py-3 rounded-xl hover:bg-green-700 transition font-semibold disabled:opacity-50"
            >
              {loading ? 'Loading...' : (isLogin ? 'Login' : 'Sign Up')}
            </button>
          </div>
          <p className="text-center mt-6">
            <button
              onClick={() => setIsLogin(!isLogin)}
              className="text-green-600 hover:underline font-medium"
            >
              {isLogin ? 'Need an account? Sign up' : 'Have an account? Login'}
            </button>
          </p>
        </div>
      </div>
    ) : (
      <div className="min-h-screen bg-green-50">
        {/* Header */}
        <header className="bg-white shadow-md">
          <div className="max-w-6xl mx-auto px-6 py-5 flex justify-between items-center">
            <h1 className="text-3xl font-bold text-green-700">Food Pantry</h1>
            <div className="flex gap-6 items-center">
              <span className="text-sm text-gray-600 font-medium">{user.email}</span>
              <button
                onClick={() => {
                  setUser(null);
                  localStorage.removeItem('bullavor_user');
                }}
                className="text-sm text-red-600 hover:underline font-medium"
              >
                Logout
              </button>
            </div>
          </div>
        </header>

        <main className="max-w-6xl mx-auto px-6 py-8 space-y-8">
          {/* Tabs */}
          <div className="flex gap-3">
            <button
              onClick={() => setActiveTab('inventory')}
              className={`px-6 py-2 rounded-full font-semibold ${activeTab === 'inventory' ? 'bg-green-600 text-white shadow-lg' : 'bg-white border hover:bg-green-50'}`}
            >
              Inventory
            </button>
            <button
              onClick={() => setActiveTab('chat')}
              className={`px-6 py-2 rounded-full font-semibold ${activeTab === 'chat' ? 'bg-green-600 text-white shadow-lg' : 'bg-white border hover:bg-green-50'}`}
            >
              Recipe Chat
            </button>
          </div>

          {/* Notification (Expiration Warnings) */}
          <Snackbar
            open={open}
            autoHideDuration={6000}
            onClose={() => setOpen(false)}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
          >
            <Alert
              onClose={() => setOpen(false)}
              severity="warning"
              sx={{ width: '100%', fontSize: 18, color: 'red', backgroundColor: '#FAFAD2', border: '2px solid #FFA500' }}
            >
              {notification}
            </Alert>
          </Snackbar>

          {/* Inventory Tab */}
          {activeTab === 'inventory' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Add / Scan Item */}
          <div className="space-y-6">
            {/* Add Item */}
            <div className="bg-white rounded-2xl shadow-lg p-6 space-y-4">
              <div className="flex justify-between items-center mb-2">
                <h2 className="text-xl font-bold">Add Item</h2>
                <button
                  onClick={() => setScanningMode(true)}
                  className="px-4 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition"
                >
                  Scan Groceries
                </button>
              </div>
              <div className="space-y-3">
                <input
                  type="text"
                  placeholder="Item name"
                  value={itemName}
                  onChange={(e) => setItemName(e.target.value)}
                  className="w-full px-4 py-3 border rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500"
                />
                <input
                  type="number"
                  placeholder="Quantity"
                  value={itemQuantity}
                  onChange={(e) => setItemQuantity(e.target.value)}
                  className="w-full px-4 py-3 border rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500"
                />
                <input
                  type="date"
                  value={itemExpiration}
                  onChange={(e) => setItemExpiration(e.target.value)}
                  className="w-full px-4 py-3 border rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500"
                />
                <select
                  value={itemCategory}
                  onChange={(e) => setItemCategory(e.target.value)}
                  className="w-full px-4 py-3 border rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500 bg-white"
                >
                  <option value="">Auto-detect category</option>
                  <option value="Fruits">Fruits</option>
                  <option value="Vegetables">Vegetables</option>
                  <option value="Dairy">Dairy</option>
                  <option value="Meat">Meat</option>
                  <option value="Grains">Grains</option>
                  <option value="Other">Other</option>
                </select>
                <button
                  onClick={addItem}
                  disabled={loading}
                  className="w-full bg-green-600 text-white py-3 rounded-xl hover:bg-green-700 transition font-semibold disabled:opacity-50"
                >
                  Add
                </button>
              </div>
            </div>

            {/* Scan Groceries */}
            {scanningMode && (
              <div className="bg-white rounded-2xl shadow-lg p-6 space-y-4">
                <div className="flex justify-between items-center">
                  <h2 className="text-xl font-bold">Scan Groceries</h2>
                  <button onClick={resetScanUI} className="text-gray-600 hover:text-gray-800 text-2xl font-bold">✕</button>
                </div>
                <div className="space-y-4">
                  <input
                    type="file"
                    accept="image/*"
                    capture
                    onChange={handleImageSelect}
                    className="hidden"
                    ref={fileInputRef}
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full px-4 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition font-medium"
                  >
                    {selectedImage ? 'Select Different Image' : 'Take Photo / Choose Image'}
                  </button>

                  {imagePreview && (
                    <div className="flex justify-center">
                      <img
                        src={imagePreview}
                        alt="Preview"
                        className="max-w-full h-64 object-contain rounded-xl border"
                      />
                    </div>
                  )}

                  {scanning && <div className="text-center text-gray-600 font-medium">Analyzing image...</div>}

                  {predictedItem && !scanning && (
                    <div className="p-4 bg-green-50 rounded-xl border">
                      <div className="font-semibold text-green-800">{predictedItem.name}</div>
                      <div className="text-sm text-green-600">
                        Confidence: {(predictedItem.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  )}

                  {predictedItem && (
                    <>
                      <input
                        type="number"
                        placeholder="Quantity"
                        value={scanQuantity}
                        onChange={(e) => setScanQuantity(e.target.value)}
                        min="1"
                        className="w-full px-4 py-3 border rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500"
                      />
                      <input
                        type="date"
                        placeholder="Expiration date (optional)"
                        value={scanExpiration}
                        onChange={(e) => setScanExpiration(e.target.value)}
                        className="w-full px-4 py-3 border rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500"
                      />
                      <button
                        onClick={addItemFromScan}
                        disabled={loading || scanning}
                        className="w-full bg-green-600 text-white py-3 rounded-xl hover:bg-green-700 transition font-semibold disabled:opacity-50"
                      >
                        {loading ? 'Adding...' : 'Add to Inventory'}
                      </button>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Inventory List */}
          <div className="space-y-4">
            <div className="bg-white rounded-2xl shadow-lg p-6 space-y-3">
              <h2 className="text-xl font-bold mb-4">Your Items</h2>
              {inventory.length === 0 ? (
                <p className="text-gray-500 text-center">No items yet</p>
              ) : (
                <div className="space-y-6">
                  {Object.entries(groupInventoryByCategory(inventory)).map(
                    ([category, items]) => (
                      <div key={category}>
                        <div className="flex items-center justify-between mb-3">
                          <h3 className="text-lg font-semibold text-gray-800">
                            {category} ({items.length})
                          </h3>
                        </div>
                        <div className="grid grid-cols-1 gap-3">
                          {items.map((item) => (
                            <div
                              key={item.id}
                              className="p-4 bg-green-50 rounded-xl border flex justify-between items-start shadow-sm"
                            >
                              <div className="flex-1">
                                <div className="font-semibold">{item.name}</div>
                                <div className="text-sm text-gray-600">
                                  Qty: {item.quantity}
                                </div>
                                {item.expiration_date && (
                                  <div className={`text-sm ${getExpirationColor(item.expiration_date)}`}>
                                    Expires:{" "}
                                    {new Date(
                                      item.expiration_date
                                    ).toLocaleDateString()}
                                  </div>
                                )}
                              </div>
                              <div className="flex items-center gap-2">
                                <select
                                  value={item.category || "Other"}
                                  onChange={(e) =>
                                    updateCategory(item.id, e.target.value)
                                  }
                                  disabled={loading}
                                  className="px-2 py-1 text-sm border rounded-lg bg-white disabled:opacity-50 hover:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500"
                                >
                                  <option value="Fruits">Fruits</option>
                                  <option value="Vegetables">Vegetables</option>
                                  <option value="Dairy">Dairy</option>
                                  <option value="Meat">Meat</option>
                                  <option value="Grains">Grains</option>
                                  <option value="Other">Other</option>
                                </select>
                                <button
                                  onClick={() => openDeleteDialog(item.id, item.name)}
                                  disabled={loading}
                                  className="ml-2 p-2 hover:bg-red-100 rounded-xl transition-colors disabled:opacity-50"
                                  title="Delete item"
                                >
                                  <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    className="h-5 w-5 text-red-600"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                    strokeWidth={2}
                                  >
                                    <path
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                                    />
                                  </svg>
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Chat Tab */}
      {activeTab === 'chat' && (
        <div className="bg-white rounded-2xl shadow-lg p-6 space-y-4">
          <h2 className="text-xl font-bold mb-4">Recipe Helper</h2>
          <div className="h-96 overflow-y-auto p-4 bg-gray-50 rounded-xl space-y-3">
            {chatMessages.length === 0 ? (
              <p className="text-gray-500 text-center">Ask me for recipe suggestions!</p>
            ) : (
              chatMessages.map((msg, i) => (
                <div key={i} className={`${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
                  <div className={`inline-block p-4 rounded-xl ${
                    msg.role === 'user' ? 'bg-green-600 text-white max-w-xs' : 'bg-gray-200 max-w-2xl'
                  }`}>
                    {msg.role === 'assistant' ? (
                      (() => {
                        const recipe = parseRecipeResponse(msg.content);
                        // If we found structured sections, render them nicely
                        if (recipe.dishName || recipe.ingredients.length > 0 || recipe.instructions.length > 0) {
                          return (
                            <div className="text-gray-800">
                              {recipe.dishName && (
                                <h3 className="font-bold text-lg mb-3 text-green-700">{recipe.dishName}</h3>
                              )}
                              {recipe.ingredients.length > 0 && (
                                <div className="mb-4">
                                  <h4 className="font-semibold mb-2 text-gray-700">Ingredients:</h4>
                                  <ul className="space-y-1">
                                    {recipe.ingredients.map((ing, idx) => (
                                      <li key={idx} className="text-sm flex items-start">
                                        <span className="text-green-600 mr-2">•</span>
                                        <span>{ing}</span>
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                              {recipe.instructions.length > 0 && (
                                <div>
                                  <h4 className="font-semibold mb-2 text-gray-700">Instructions:</h4>
                                  <ol className="space-y-2">
                                    {recipe.instructions.map((step, idx) => (
                                      <li key={idx} className="text-sm flex items-start">
                                        <span className="font-semibold text-green-600 mr-2 min-w-[1.5rem]">
                                          {idx + 1}.
                                        </span>
                                        <span>{step.replace(/^\d+\.\s*/, '')}</span>
                                      </li>
                                    ))}
                                  </ol>
                                </div>
                              )}
                            </div>
                          );
                        } else {
                          // Fallback to plain text if no structure found
                          return <div className="text-sm whitespace-pre-wrap">{msg.content}</div>;
                        }
                      })()
                    ) : (
                      msg.content
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
          <div className="flex gap-3">
            <input
              type="text"
              placeholder="Ask for recipes..."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              className="flex-1 px-4 py-3 border rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500"
              disabled={loading}
            />
            <button
              onClick={sendMessage}
              disabled={loading}
              className="px-6 py-3 bg-green-600 text-white rounded-xl hover:bg-green-700 transition font-semibold disabled:opacity-50"
            >
              {loading ? '...' : 'Send'}
            </button>
          </div>
        </div>
          )}
        </main>
      </div>
    )}
  </>
);
}


export default App;