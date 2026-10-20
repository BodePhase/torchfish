import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// StrictMode rimosso per evitare conflitti con react-dnd
createRoot(document.getElementById('root')).render(
    <App />
)