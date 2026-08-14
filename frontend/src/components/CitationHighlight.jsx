import { useState, useRef } from 'react'
import { FileText, ChevronDown, ChevronUp } from 'lucide-react'

/**
 * CitationHighlight
 *
 * Renders the answer text with [1], [2] … markers replaced by interactive
 * superscript badges. Hovering a badge shows a popover with the source doc,
 * page number, and snippet from the citation.
 *
 * Props:
 *   text        — raw answer string (may contain [1], [2] … references)
 *   citations   — array of Citation objects from QueryResponse
 */
export function CitationHighlight({ text = '', citations = [] }) {
  const [openIdx, setOpenIdx] = useState(null)
  const containerRef = useRef()

  // Build a lookup: citation number (1-based) → citation object
  // Citations returned by the API are 0-indexed in the array.
  const citationMap = {}
  citations.forEach((c, i) => { citationMap[i + 1] = c })

  // Split text on [N] patterns
  const parts = []
  const regex = /\[(\d+)\]/g
  let last = 0, match
  while ((match = regex.exec(text)) !== null) {
    if (match.index > last) parts.push({ type: 'text', value: text.slice(last, match.index) })
    parts.push({ type: 'citation', num: parseInt(match[1], 10) })
    last = match.index + match[0].length
  }
  if (last < text.length) parts.push({ type: 'text', value: text.slice(last) })

  const toggle = (n) => setOpenIdx(prev => prev === n ? null : n)

  return (
    <span ref={containerRef} style={{ position: 'relative' }}>
      {parts.map((p, i) => {
        if (p.type === 'text') return <span key={i}>{p.value}</span>
        const cit = citationMap[p.num]
        return (
          <span key={i} style={{ position: 'relative', display: 'inline-block' }}>
            <span
              id={`citation-marker-${p.num}`}
              className="citation-marker"
              onClick={() => toggle(p.num)}
              role="button"
              aria-label={`Citation ${p.num}`}
            >
              {p.num}
            </span>
            {openIdx === p.num && cit && (
              <div className="citation-popover" role="tooltip">
                <div className="popover-source">
                  <FileText size={12} />
                  <span>{cit.source || 'Unknown source'}</span>
                  {cit.page_num != null && <span style={{ color: 'var(--text-2)' }}>· p. {cit.page_num}</span>}
                  {cit.section && <span style={{ color: 'var(--text-2)' }}>· {cit.section}</span>}
                </div>
                <div className="popover-snippet">
                  {cit.snippet ? `"${cit.snippet.slice(0, 200)}…"` : 'No snippet available.'}
                </div>
              </div>
            )}
          </span>
        )
      })}
    </span>
  )
}

/**
 * CitationList
 *
 * Renders the full citations list below the answer as expandable cards.
 */
export function CitationList({ citations = [] }) {
  const [expanded, setExpanded] = useState(false)
  if (!citations.length) return null

  return (
    <div className="citations-list">
      <button
        className="btn btn-ghost"
        style={{ alignSelf: 'flex-start', fontSize: '0.78rem', padding: '4px 10px', gap: 5 }}
        onClick={() => setExpanded(e => !e)}
        id="toggle-citations"
      >
        {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        {citations.length} source{citations.length !== 1 ? 's' : ''}
      </button>
      {expanded && citations.map((c, i) => (
        <div key={c.chunk_id || i} className="citation-card">
          <div className="citation-card-header">
            <span className="citation-marker" style={{ cursor: 'default' }}>{i + 1}</span>
            <FileText size={13} style={{ color: 'var(--brand-1)' }} />
            <span style={{ color: 'var(--text-0)', fontWeight: 600 }}>{c.source || 'Unknown'}</span>
            {c.page_num != null && <span style={{ color: 'var(--text-2)', fontSize: '0.75rem' }}>p. {c.page_num}</span>}
            {c.section && <span style={{ color: 'var(--text-2)', fontSize: '0.75rem' }}>· {c.section}</span>}
          </div>
          {c.snippet && (
            <div className="citation-card-snippet">{c.snippet.slice(0, 280)}{c.snippet.length > 280 ? '…' : ''}</div>
          )}
        </div>
      ))}
    </div>
  )
}
