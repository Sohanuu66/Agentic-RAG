import { useState } from 'react'
import { AlertTriangle, CheckCircle, ChevronDown, ChevronUp, XCircle } from 'lucide-react'

/**
 * HallucinationBadge
 *
 * Props:
 *   flags    — array of HallucinationFlag objects
 *   confidence — overall confidence score (0-1) or null
 */
export default function HallucinationBadge({ flags = [], confidence }) {
  const [open, setOpen] = useState(false)

  if (!flags || flags.length === 0) return null

  const flagged       = flags.filter(f => f.flagged)
  const contradictions= flags.filter(f => f.label === 'contradiction')
  const entailed      = flags.filter(f => f.label === 'entailment')

  const hasProblem = contradictions.length > 0

  const summaryText = hasProblem
    ? `${contradictions.length} potential hallucination${contradictions.length !== 1 ? 's' : ''} detected`
    : flagged.length > 0
      ? `${flagged.length} claim${flagged.length !== 1 ? 's' : ''} need review`
      : `All ${entailed.length} claims verified`

  const Icon = hasProblem ? AlertTriangle : flagged.length > 0 ? AlertTriangle : CheckCircle
  const bannerStyle = hasProblem || flagged.length > 0
    ? {}
    : { background: 'rgba(16,185,129,0.08)', borderColor: 'rgba(16,185,129,0.25)' }

  return (
    <div className="hallucination-banner" style={bannerStyle} id="hallucination-banner">
      <div className="hallucination-header" onClick={() => setOpen(o => !o)}>
        <Icon size={15} color={hasProblem ? 'var(--warning)' : flagged.length > 0 ? 'var(--warning)' : 'var(--success)'} />
        <span className="banner-title" style={{ color: hasProblem || flagged.length > 0 ? 'var(--warning)' : 'var(--success)' }}>
          {summaryText}
        </span>
        {confidence != null && (
          <span style={{ fontSize: '0.75rem', color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>
            {Math.round(confidence * 100)}% confidence
          </span>
        )}
        {open ? <ChevronUp size={14} color="var(--text-2)" /> : <ChevronDown size={14} color="var(--text-2)" />}
      </div>

      {open && (
        <div className="hallucination-claims">
          {flags.map((f, i) => (
            <div key={i} className={`claim-row ${f.label}`} id={`claim-${i}`}>
              <span className={`claim-label ${f.label}`}>
                {f.label === 'entailment'    ? '✓' :
                 f.label === 'contradiction' ? '✗' : '~'}
                {' '}{f.label}
              </span>
              <span style={{ flex: 1, color: 'var(--text-0)' }}>{f.claim}</span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-2)', fontFamily: 'var(--font-mono)', flexShrink: 0 }}>
                {Math.round(f.confidence * 100)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
