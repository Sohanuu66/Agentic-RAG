import { useState, useRef } from 'react'
import axios from 'axios'
import { Upload, FileText, Trash2, RefreshCw, ChevronDown } from 'lucide-react'

const CHUNKING_STRATEGIES = [
  { value: 'fixed',           label: 'Fixed Size (512 tokens)' },
  { value: 'semantic',        label: 'Semantic Chunking' },
  { value: 'sentence_window', label: 'Sentence Window' },
]

export default function DocumentUpload({ onCorpusChange }) {
  const [dragOver, setDragOver]       = useState(false)
  const [uploading, setUploading]     = useState(false)
  const [progress, setProgress]       = useState(0)
  const [strategy, setStrategy]       = useState('fixed')
  const [stats, setStats]             = useState(null)
  const [lastDoc, setLastDoc]         = useState(null)
  const fileRef = useRef()

  const fetchStats = async () => {
    try {
      const { data } = await axios.get('/ingest/stats')
      setStats(data)
      onCorpusChange?.(data)
    } catch { /* silent */ }
  }

  const handleFile = async (file) => {
    if (!file) return
    setUploading(true)
    setProgress(10)
    const form = new FormData()
    form.append('file', file)
    try {
      setProgress(40)
      const { data } = await axios.post(`/ingest/upload?chunking_strategy=${strategy}`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: e => setProgress(Math.round(40 + (e.loaded / e.total) * 40)),
      })
      setProgress(100)
      setLastDoc({ name: file.name, chunks: data.chunks_created, strategy: data.chunking_strategy })
      await fetchStats()
    } catch (err) {
      console.error(err)
    } finally {
      setTimeout(() => { setUploading(false); setProgress(0) }, 600)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault(); setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  const clearIndex = async () => {
    if (!window.confirm('Clear all indexed documents?')) return
    await axios.delete('/ingest/')
    setStats(null); setLastDoc(null)
    onCorpusChange?.(null)
  }

  // Fetch stats on mount
  useState(() => { fetchStats() }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Documents</span>
        <div style={{ display: 'flex', gap: 6 }}>
          <button className="btn btn-ghost" style={{ padding: '5px 8px' }} onClick={fetchStats} title="Refresh stats">
            <RefreshCw size={14} />
          </button>
          <button className="btn btn-danger" style={{ padding: '5px 8px' }} onClick={clearIndex} title="Clear index">
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      {/* Drop zone */}
      <div
        id="upload-dropzone"
        className={`upload-zone${dragOver ? ' drag-over' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
      >
        <input ref={fileRef} type="file" accept=".pdf,.txt,.md" onChange={e => handleFile(e.target.files[0])} />
        <div className="upload-icon"><Upload size={28} /></div>
        <h3>Drop a document here</h3>
        <p>PDF · TXT · Markdown</p>
      </div>

      {/* Progress */}
      {uploading && (
        <div className="upload-progress">
          <div className="upload-progress-bar" style={{ width: `${progress}%` }} />
        </div>
      )}

      {/* Chunking strategy */}
      <div className="strategy-select">
        <label htmlFor="chunking-strategy">Chunking strategy</label>
        <div style={{ position: 'relative' }}>
          <select
            id="chunking-strategy"
            className="input"
            value={strategy}
            onChange={e => setStrategy(e.target.value)}
            style={{ paddingRight: 32, appearance: 'none' }}
          >
            {CHUNKING_STRATEGIES.map(s => (
              <option key={s.value} value={s.value}>{s.label}</option>
            ))}
          </select>
          <ChevronDown size={14} style={{ position:'absolute', right:10, top:'50%', transform:'translateY(-50%)', color:'var(--text-2)', pointerEvents:'none' }} />
        </div>
      </div>

      {/* Last uploaded */}
      {lastDoc && (
        <div style={{ background:'var(--bg-3)', border:'1px solid var(--border)', borderRadius:'var(--r-md)', padding:'10px 12px', fontSize:'0.8rem' }}>
          <div style={{ display:'flex', alignItems:'center', gap:6, marginBottom:4 }}>
            <FileText size={13} style={{ color:'var(--brand-1)' }} />
            <span style={{ fontWeight:600, color:'var(--text-0)' }}>{lastDoc.name}</span>
          </div>
          <span style={{ color:'var(--text-2)' }}>{lastDoc.chunks} chunks · {lastDoc.strategy}</span>
        </div>
      )}

      {/* Corpus stats */}
      {stats && (
        <div className="corpus-stats">
          <div className="stat-pill">
            <div className="stat-value">{stats.total_chunks ?? stats.chunk_count ?? '—'}</div>
            <div className="stat-label">Chunks</div>
          </div>
          <div className="stat-pill">
            <div className="stat-value">{stats.total_documents ?? stats.doc_count ?? '—'}</div>
            <div className="stat-label">Docs</div>
          </div>
        </div>
      )}
    </div>
  )
}
