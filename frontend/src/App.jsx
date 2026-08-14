import { useEffect, useState } from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import axios from 'axios'
import { MessageSquare, BarChart2, BookOpen } from 'lucide-react'
import ChatPage from './pages/ChatPage'
import EvalPage from './pages/EvalPage'
import './index.css'
import './App.css'

function Navbar({ apiOnline }) {
  return (
    <nav className="navbar" id="navbar">
      <div className="navbar-brand">
        <div className="logo-dot" />
        <span>Ask <span style={{ color: 'var(--brand-1)' }}>My</span> Docs</span>
      </div>

      <div className="navbar-links">
        <NavLink
          to="/"
          end
          id="nav-chat"
          className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
        >
          <MessageSquare size={15} />
          Chat
        </NavLink>
        <NavLink
          to="/eval"
          id="nav-eval"
          className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
        >
          <BarChart2 size={15} />
          Evaluation
        </NavLink>
      </div>

      <div className="navbar-status">
        <div className={`status-dot${apiOnline ? '' : ' offline'}`} />
        <span>{apiOnline ? 'API online' : 'API offline'}</span>
      </div>
    </nav>
  )
}

export default function App() {
  const [apiOnline, setApiOnline] = useState(false)

  useEffect(() => {
    const check = async () => {
      try {
        await axios.get('/health')
        setApiOnline(true)
      } catch {
        setApiOnline(false)
      }
    }
    check()
    const id = setInterval(check, 15000)
    return () => clearInterval(id)
  }, [])

  return (
    <BrowserRouter>
      <Navbar apiOnline={apiOnline} />
      <Routes>
        <Route path="/"     element={<ChatPage />} />
        <Route path="/eval" element={<EvalPage />} />
      </Routes>
    </BrowserRouter>
  )
}
