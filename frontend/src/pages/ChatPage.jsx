import DocumentUpload from '../components/DocumentUpload'
import ChatInterface from '../components/ChatInterface'
import { useState } from 'react'

export default function ChatPage() {
  const [corpus, setCorpus] = useState(null)
  return (
    <div className="page">
      <aside className="sidebar" id="sidebar">
        <DocumentUpload onCorpusChange={setCorpus} />
      </aside>
      <main className="main-content" id="chat-area">
        <ChatInterface />
      </main>
    </div>
  )
}
