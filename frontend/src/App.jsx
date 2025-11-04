import React, { useState, useEffect } from 'react';

const API_URL = 'http://localhost:8000';

function App() {
  //The const variables handle the state manegment of key info- ensuring that the ui reflects the variables accurately
  const [user, setUser] = useState(null);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLogin, setIsLogin] = useState(true);
  const [activeTab, setActiveTab] = useState('inventory');
  
  const [inventory, setInventory] = useState([]);
  const [itemName, setItemName] = useState('');
  const [itemQuantity, setItemQuantity] = useState('');
  const [itemExpiration, setItemExpiration] = useState('');
  
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem('bullavor_user');
    if (stored) setUser(JSON.parse(stored));
  }, []);

  useEffect(() => {
    //fetch inventory when user logs in
    if (user) fetchInventory();
  }, [user]);

  //Sends a POST request to the backend to either log in or sign up the user based on the isLogin state
  //A POST request means sending data from the frontend to the backend for processing, in this case for logging/signing in
  const handleAuth = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/auth/${isLogin ? 'login' : 'signup'}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const data = await res.json();
      //If the response is successful, sets the user to both state and local storage
      if (res.ok) {
        setUser(data.user);
        localStorage.setItem('bullavor_user', JSON.stringify(data.user));
      } else {
        alert(data.detail);
      }
    } catch (err) {
      alert('Error: ' + err.message);
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

  //Sends the new item data to the backend, then refreshes the inventory list to show the new item
  const addItem = async () => {
    if (!itemName || !itemQuantity) return alert('Name and quantity required');
    setLoading(true);
    try {
      await fetch(`${API_URL}/inventory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: user.uid,
          name: itemName,
          quantity: parseInt(itemQuantity),
          expiration_date: itemExpiration || null
        })
      });
      setItemName('');
      setItemQuantity('');
      setItemExpiration('');
      fetchInventory();
    } catch (err) {
      alert('Error adding item');
    }
    setLoading(false);
  };

  //Used to send the user's message to the ai model in the backend and return the response in the chat UI
  const sendMessage = async () => {
    if (!chatInput.trim()) return;
    
    setChatMessages([...chatMessages, { role: 'user', content: chatInput }]);
    const msg = chatInput;
    setChatInput('');
    setLoading(true);
    
    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: user.uid, message: msg })
      });
      const data = await res.json();
      setChatMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch (err) {
      setChatMessages(prev => [...prev, { role: 'assistant', content: 'Error occurred' }]);
    }
    setLoading(false);
  };
//The main return statement that renders the UI based on whether the user is logged in or not
  if (!user) {
    return (
      //Login/Sign in page, uses google AuthO
      <div className="min-h-screen bg-green-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg shadow p-8 w-full max-w-md">
          <h1 className="text-3xl font-bold text-green-700 mb-6 text-center">Food Pantry</h1>
          <div className="space-y-4">
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleAuth()}
              className="w-full px-4 py-2 border rounded"
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleAuth()}
              className="w-full px-4 py-2 border rounded"
            />
            <button
              onClick={handleAuth}
              disabled={loading}
              className="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700 disabled:opacity-50"
            >
              {loading ? 'Loading...' : (isLogin ? 'Login' : 'Sign Up')}
            </button>
          </div>
          <p className="text-center mt-4">
            <button
              onClick={() => setIsLogin(!isLogin)}
              className="text-green-600 hover:underline"
            >
              {isLogin ? 'Need an account? Sign up' : 'Have an account? Login'}
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
                localStorage.removeItem('bullavor_user');
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
            onClick={() => setActiveTab('inventory')}
            className={`px-6 py-2 rounded ${activeTab === 'inventory' ? 'bg-green-600 text-white' : 'bg-white'}`}
          >
            Inventory
          </button>
          <button
            onClick={() => setActiveTab('chat')}
            className={`px-6 py-2 rounded ${activeTab === 'chat' ? 'bg-green-600 text-white' : 'bg-white'}`}
          >
            Recipe Chat
          </button>
        </div>
{/* Used to add items to the inventory*/}
        {activeTab === 'inventory' && (
          <div className="space-y-4">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold mb-4">Add Item</h2>
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
                <button
                  onClick={addItem}
                  disabled={loading}
                  className="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700 disabled:opacity-50"
                >
                  Add
                </button>
              </div>
            </div>
{/* Shows the items in your inventory*/}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold mb-4">Your Items</h2>
              {inventory.length === 0 ? (
                <p className="text-gray-500">No items yet</p>
              ) : (
                <div className="space-y-2">
                  {inventory.map((item) => (
                    <div key={item.id} className="p-3 bg-green-50 rounded border">
                      <div className="font-semibold">{item.name}</div>
                      <div className="text-sm text-gray-600">Qty: {item.quantity}</div>
                      {item.expiration_date && (
                        <div className="text-sm text-gray-600">
                          Expires: {new Date(item.expiration_date).toLocaleDateString()}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
{/* Chatting with the model*/}
        {activeTab === 'chat' && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold mb-4">Recipe Helper</h2>
            <div className="h-96 overflow-y-auto mb-4 p-4 bg-gray-50 rounded">
              {chatMessages.length === 0 && (
                <p className="text-gray-500 text-center">Ask me for recipe suggestions!</p>
              )}
              {chatMessages.map((msg, i) => (
                <div key={i} className={`mb-3 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
                  <div className={`inline-block p-3 rounded-lg max-w-xs ${
                    msg.role === 'user' ? 'bg-green-600 text-white' : 'bg-gray-200'
                  }`}>
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
                onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                className="flex-1 px-3 py-2 border rounded"
                disabled={loading}
              />
              <button
                onClick={sendMessage}
                disabled={loading}
                className="px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
              >
                {loading ? '...' : 'Send'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;