import React, { useState, useEffect, useRef } from "react";
import { Snackbar, Alert } from "@mui/material";


const API_URL = "http://localhost:8000";


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


 // Notification state
 const [notification, setNotification] = useState(""); // The message to show
 const [open, setOpen] = useState(false); // Whether the notification is visible


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
     } else {
       alert(data.detail);
     }
   } catch (err) {
     alert("Error: " + err.message);
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
   if (!itemName || !itemQuantity) return alert("Name and quantity required");
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
     fetchInventory();
   } catch (err) {
     alert("Error adding item");
   }
   setLoading(false);
 };


 //Deletes an item from the inventory by sending a POST request to the backend and updating the UI
 const deleteItem = async (itemId) => {
   if (!confirm("Are you sure you want to delete this item?")) return;
   setLoading(true);
   try {
     await fetch(`${API_URL}/inventory/delete`, {
       method: "POST",
       headers: { "Content-Type": "application/json" },
       body: JSON.stringify({ item_id: itemId }),
     });
     fetchInventory(); // Refresh the inventory list
   } catch (err) {
     alert("Error deleting item");
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
     fetchInventory(); // Refresh the inventory list
   } catch (err) {
     alert("Error updating category");
   }
   setLoading(false);
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
     alert("Error classifying image: " + err.message);
     setPredictedItem(null);
   }
   setScanning(false);
 };


 //Add item to inventory from scanned image
 const addItemFromScan = async () => {
   if (!selectedImage || !predictedItem) {
     alert("Please select and classify an image first");
     return;
   }
   if (!scanQuantity || parseInt(scanQuantity) < 1) {
     alert("Please enter a valid quantity (at least 1)");
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


     alert(
       `Item added successfully! Added ${responseData.item_name} to inventory.`
     );
   } catch (err) {
     alert("Error adding item: " + err.message);
   }
   setLoading(false);
 };


 //Reset scan UI when closing
 const resetScanUI = () => {
   if (imagePreview) {
     URL.revokeObjectURL(imagePreview);
   }
   if (fileInputRef.current) {
     fileInputRef.current.value = "";
   }
   setScanningMode(false);
   setSelectedImage(null);
   setImagePreview(null);
   setPredictedItem(null);
   setScanQuantity("1");
   setScanExpiration("");
 };
 //The main return statement that renders the UI based on whether the user is logged in or not
 if (!user) {
   return (
     //Login/Sign in page, uses google AuthO
     <div className="min-h-screen bg-green-50 flex items-center justify-center p-4">
       <div className="bg-white rounded-lg shadow p-8 w-full max-w-md">
         <h1 className="text-3xl font-bold text-green-700 mb-6 text-center">
           Food Pantry
         </h1>
         <div className="space-y-4">
           <input
             type="email"
             placeholder="Email"
             value={email}
             onChange={(e) => setEmail(e.target.value)}
             onKeyPress={(e) => e.key === "Enter" && handleAuth()}
             className="w-full px-4 py-2 border rounded"
           />
           <input
             type="password"
             placeholder="Password"
             value={password}
             onChange={(e) => setPassword(e.target.value)}
             onKeyPress={(e) => e.key === "Enter" && handleAuth()}
             className="w-full px-4 py-2 border rounded"
           />
           <button
             onClick={handleAuth}
             disabled={loading}
             className="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700 disabled:opacity-50"
           >
             {loading ? "Loading..." : isLogin ? "Login" : "Sign Up"}
           </button>
         </div>
         <p className="text-center mt-4">
           <button
             onClick={() => setIsLogin(!isLogin)}
             className="text-green-600 hover:underline"
           >
             {isLogin ? "Need an account? Sign up" : "Have an account? Login"}
           </button>
         </p>
       </div>
     </div>
   );
 }


 return (
   <div className="min-h-screen bg-green-50">
     <div className="bg-white shadow mb-4">
       <div className="max-w-4xl mx-auto px-4 py-4 flex justify-between items-center">
         <h1 className="text-2xl font-bold text-green-700">Food Pantry</h1>
         <div className="flex gap-4 items-center">
           <span className="text-sm text-gray-600">{user.email}</span>
           <button
             onClick={() => {
               setUser(null);
               localStorage.removeItem("bullavor_user");
             }}
             className="text-sm text-red-600 hover:underline"
           >
             Logout
           </button>
         </div>
       </div>
     </div>


     <div className="max-w-4xl mx-auto px-4">
       <div className="flex gap-2 mb-4">
         <button
           onClick={() => setActiveTab("inventory")}
           className={`px-6 py-2 rounded ${
             activeTab === "inventory" ? "bg-green-600 text-white" : "bg-white"
           }`}
         >
           Inventory
         </button>
         <button
           onClick={() => setActiveTab("chat")}
           className={`px-6 py-2 rounded ${
             activeTab === "chat" ? "bg-green-600 text-white" : "bg-white"
           }`}
         >
           Recipe Chat
         </button>
       </div>
       {/* Step 4: Notification goes here */}
       <Snackbar
         open={open}
         autoHideDuration={6000}
         onClose={() => setOpen(false)}
         anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
       >
         <Alert
           onClose={() => setOpen(false)}
           severity="warning"
           sx={{
             width: "100%",
             fontSize: 18,
             color: "red",
             backgroundColor: "#FAFAD2",
             border: "2px solid #FFA500",
           }}
         >
           {notification}
         </Alert>
       </Snackbar>
       {/* Used to add items to the inventory*/}
       {activeTab === "inventory" && (
         <div className="space-y-4">
           <div className="bg-white rounded-lg shadow p-6">
             <div className="flex justify-between items-center mb-4">
               <h2 className="text-xl font-bold">Add Item</h2>
               <button
                 onClick={() => setScanningMode(true)}
                 className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
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
                 className="w-full px-3 py-2 border rounded"
               />
               <input
                 type="number"
                 placeholder="Quantity"
                 value={itemQuantity}
                 onChange={(e) => setItemQuantity(e.target.value)}
                 className="w-full px-3 py-2 border rounded"
               />
               <input
                 type="date"
                 value={itemExpiration}
                 onChange={(e) => setItemExpiration(e.target.value)}
                 className="w-full px-3 py-2 border rounded"
               />
               <select
                 value={itemCategory}
                 onChange={(e) => setItemCategory(e.target.value)}
                 className="w-full px-3 py-2 border rounded"
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
                 className="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700 disabled:opacity-50"
               >
                 Add
               </button>
             </div>
           </div>
           {/* Scan Groceries UI */}
           {scanningMode && (
             <div className="bg-white rounded-lg shadow p-6">
               <div className="flex justify-between items-center mb-4">
                 <h2 className="text-xl font-bold">Scan Groceries</h2>
                 <button
                   onClick={resetScanUI}
                   className="text-gray-600 hover:text-gray-800"
                 >
                   ✕
                 </button>
               </div>


               <div className="space-y-4">
                 {/* File input with capture attribute */}
                 <div>
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
                     className="w-full px-4 py-3 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                   >
                     {selectedImage
                       ? "Select Different Image"
                       : "Take Photo / Choose Image"}
                   </button>
                 </div>


                 {/* Image preview */}
                 {imagePreview && (
                   <div className="flex justify-center">
                     <img
                       src={imagePreview}
                       alt="Preview"
                       className="max-w-full h-64 object-contain rounded border"
                     />
                   </div>
                 )}


                 {/* Classification result */}
                 {scanning && (
                   <div className="text-center text-gray-600">
                     Analyzing image...
                   </div>
                 )}


                 {predictedItem && !scanning && (
                   <div className="p-4 bg-green-50 rounded border">
                     <div className="font-semibold text-green-800">
                       {predictedItem.name}
                     </div>
                     <div className="text-sm text-green-600">
                       Confidence:{" "}
                       {(predictedItem.confidence * 100).toFixed(1)}%
                     </div>
                   </div>
                 )}


                 {/* Quantity and expiration inputs */}
                 {predictedItem && (
                   <>
                     <input
                       type="number"
                       placeholder="Quantity"
                       value={scanQuantity}
                       onChange={(e) => setScanQuantity(e.target.value)}
                       min="1"
                       className="w-full px-3 py-2 border rounded"
                     />
                     <input
                       type="date"
                       placeholder="Expiration date (optional)"
                       value={scanExpiration}
                       onChange={(e) => setScanExpiration(e.target.value)}
                       className="w-full px-3 py-2 border rounded"
                     />
                     <button
                       onClick={addItemFromScan}
                       disabled={loading || scanning}
                       className="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700 disabled:opacity-50"
                     >
                       {loading ? "Adding..." : "Add to Inventory"}
                     </button>
                   </>
                 )}
               </div>
             </div>
           )}
           {/* Shows the items in your inventory grouped by category*/}
           <div className="bg-white rounded-lg shadow p-6">
             <h2 className="text-xl font-bold mb-4">Your Items</h2>
             {inventory.length === 0 ? (
               <p className="text-gray-500">No items yet</p>
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
                       <div className="space-y-2">
                         {items.map((item) => (
                           <div
                             key={item.id}
                             className="p-3 bg-green-50 rounded border flex justify-between items-start"
                           >
                             <div className="flex-1">
                               <div className="font-semibold">{item.name}</div>
                               <div className="text-sm text-gray-600">
                                 Qty: {item.quantity}
                               </div>
                               {item.expiration_date && (
                                 <div className="text-sm text-gray-600">
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
                                 className="px-2 py-1 text-sm border rounded bg-white disabled:opacity-50"
                               >
                                 <option value="Fruits">Fruits</option>
                                 <option value="Vegetables">Vegetables</option>
                                 <option value="Dairy">Dairy</option>
                                 <option value="Meat">Meat</option>
                                 <option value="Grains">Grains</option>
                                 <option value="Other">Other</option>
                               </select>
                               <button
                                 onClick={() => deleteItem(item.id)}
                                 disabled={loading}
                                 className="ml-2 p-2 hover:bg-red-100 rounded transition-colors disabled:opacity-50"
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
       )}
       {/* Chatting with the model*/}
       {activeTab === "chat" && (
         <div className="bg-white rounded-lg shadow p-6">
           <h2 className="text-xl font-bold mb-4">Recipe Helper</h2>
           <div className="h-96 overflow-y-auto mb-4 p-4 bg-gray-50 rounded">
             {chatMessages.length === 0 && (
               <p className="text-gray-500 text-center">
                 Ask me for recipe suggestions!
               </p>
             )}
             {chatMessages.map((msg, i) => (
               <div
                 key={i}
                 className={`mb-3 ${
                   msg.role === "user" ? "text-right" : "text-left"
                 }`}
               >
                 <div
                   className={`inline-block p-3 rounded-lg max-w-xs ${
                     msg.role === "user"
                       ? "bg-green-600 text-white"
                       : "bg-gray-200"
                   }`}
                 >
                   {msg.content}
                 </div>
               </div>
             ))}
           </div>
           <div className="flex gap-2">
             <input
               type="text"
               placeholder="Ask for recipes..."
               value={chatInput}
               onChange={(e) => setChatInput(e.target.value)}
               onKeyPress={(e) => e.key === "Enter" && sendMessage()}
               className="flex-1 px-3 py-2 border rounded"
               disabled={loading}
             />
             <button
               onClick={sendMessage}
               disabled={loading}
               className="px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
             >
               {loading ? "..." : "Send"}
             </button>
           </div>
         </div>
       )}
     </div>
   </div>
 );
}


export default App;