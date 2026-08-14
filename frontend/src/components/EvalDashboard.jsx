import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts'
import { Play, BarChart2, RefreshCw, AlertCircle } from 'lucide-react'

const METRICS     = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
const METRIC_COLORS = ['#6366f1', '#22d3ee', '#10b981', '#f59e0b']
const METRIC_LABELS = {
  faithfulness:      'Faithfulness',
  answer_relevancy:  'Answer Relevancy',
  context_precision: 'Context Precision',
  context_recall:    'Context Recall',
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: 'var(--bg-2)', border: '1px solid var(--border-med)', borderRadius: 'var(--r-md)', padding: '10px 14px', fontSize: '0.82rem' }}>
      <div style={{ color: 'var(--text-2)', marginBottom: 6 }}>{label}</div>
      {payload.map(p => (
        <div key={p.name} style={{ color: p.color, display: 'flex', gap: 8, justifyContent: 'space-between' }}>
          <span>{METRIC_LABELS[p.name] || p.name}</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{(p.value * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

export default function EvalDashboard() {
  const [results, setResults]       = useState([])
  const [ablation, setAblation]     = useState(null)
  const [ablationType, setAblType]  = useState('chunking')
  const [running, setRunning]       = useState(false)
  const [limit, setLimit]           = useState(10)
  const [error, setError]           = useState(null)
  const [baseline, setBaseline]     = useState(null)

  const fetchResults = useCallback(async () => {
    try {
      const { data } = await axios.get('/eval/results?limit=20')
      setResults(data.results || [])
    } catch { /* silent */ }
  }, [])

  const fetchBaseline = useCallback(async () => {
    try {
      const { data } = await axios.get('/eval/baseline')
      setBaseline(data.aggregate_scores)
    } catch { /* silent */ }
  }, [])

  const fetchAblation = useCallback(async (type) => {
    try {
      const { data } = await axios.get(`/eval/ablation/${type}`)
      setAblation(data)
    } catch { setAblation(null) }
  }, [])

  useEffect(() => {
    fetchResults()
    fetchBaseline()
    fetchAblation(ablationType)
  }, [])

  const runEval = async () => {
    setRunning(true); setError(null)
    try {
      await axios.post('/eval/run', { limit: limit || null })
      await fetchResults()
    } catch (err) {
      setError(err.response?.data?.detail || 'Evaluation failed.')
    } finally {
      setRunning(false)
    }
  }

  // Build chart data from results history
  const chartData = [...results].reverse().map(r => {
    const scores = r.aggregate_scores || {}
    return {
      name: new Date(r.timestamp).toLocaleDateString('en', { month:'short', day:'numeric' }),
      ...METRICS.reduce((acc, m) => ({ ...acc, [m]: scores[m] && scores[m] > 0 ? scores[m] : null }), {}),
    }
  })

  const latestScores = results[0]?.aggregate_scores || {}
  const latestValid  = (m) => latestScores[m] != null && latestScores[m] >= 0

  return (
    <div className="eval-page" id="eval-dashboard">
      {/* Header */}
      <div>
        <h1>Evaluation <span className="gradient-text">Dashboard</span></h1>
        <p style={{ color: 'var(--text-1)', fontSize: '0.9rem', marginTop: 6 }}>
          RAGAS metrics, ablation experiments, and regression tracking.
        </p>
      </div>

      {/* Run controls */}
      <div className="card eval-run-controls">
        <Play size={16} style={{ color: 'var(--brand-1)' }} />
        <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Run Evaluation</span>
        <div style={{ display:'flex', alignItems:'center', gap:6 }}>
          <label style={{ fontSize:'0.8rem', color:'var(--text-1)' }}>Questions:</label>
          <input
            id="eval-limit"
            type="number"
            className="input run-limit-input"
            value={limit}
            onChange={e => setLimit(parseInt(e.target.value) || 10)}
            min={1} max={100}
          />
        </div>
        <button
          id="run-eval-btn"
          className="btn btn-primary"
          onClick={runEval}
          disabled={running}
        >
          {running ? <><div className="spinner" />Running…</> : <><Play size={14} />Run Eval</>}
        </button>
        <button className="btn btn-ghost" onClick={() => { fetchResults(); fetchBaseline(); }} style={{ padding: '9px 12px' }}>
          <RefreshCw size={14} />
        </button>
        {error && (
          <span style={{ color: 'var(--danger)', fontSize: '0.82rem', display:'flex', alignItems:'center', gap:5 }}>
            <AlertCircle size={13} />{error}
          </span>
        )}
      </div>

      {/* Latest score cards */}
      <div>
        <h2>Latest Scores</h2>
        <div className="metrics-grid">
          {METRICS.map((m, i) => {
            const val  = latestValid(m) ? latestScores[m] : null
            const base = baseline?.[m]
            const delta = val != null && base != null && base >= 0 ? val - base : null
            return (
              <div className="metric-card" key={m} id={`metric-card-${m}`}>
                <div className="metric-label">{METRIC_LABELS[m]}</div>
                <div className="metric-value">{val != null ? `${(val * 100).toFixed(1)}%` : '—'}</div>
                <div className="metric-sub">
                  {delta != null
                    ? <span style={{ color: delta >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                        {delta >= 0 ? '+' : ''}{(delta * 100).toFixed(1)}% vs baseline
                      </span>
                    : 'No runs yet'}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Trend chart */}
      {chartData.length > 1 && (
        <div className="chart-container">
          <h2>Metric Trends Over Time</h2>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={{ fill: 'var(--text-2)', fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 1]} tickFormatter={v => `${(v*100).toFixed(0)}%`} tick={{ fill: 'var(--text-2)', fontSize: 12 }} axisLine={false} tickLine={false} width={40} />
              <Tooltip content={<CustomTooltip />} />
              <Legend formatter={n => METRIC_LABELS[n] || n} wrapperStyle={{ fontSize: '0.8rem' }} />
              {METRICS.map((m, i) => (
                <Line key={m} type="monotone" dataKey={m} stroke={METRIC_COLORS[i]} strokeWidth={2} dot={{ r: 4, fill: METRIC_COLORS[i] }} activeDot={{ r: 6 }} connectNulls />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Ablation table */}
      <div className="chart-container">
        <div style={{ display:'flex', alignItems:'center', gap:14, marginBottom:14 }}>
          <BarChart2 size={16} style={{ color:'var(--brand-1)' }} />
          <h2 style={{ margin: 0 }}>Ablation Study</h2>
          <div style={{ display:'flex', gap:4 }}>
            {['chunking','retrieval'].map(t => (
              <button
                key={t}
                className={`btn ${ablationType===t ? 'btn-primary' : 'btn-ghost'}`}
                style={{ fontSize:'0.78rem', padding:'5px 12px' }}
                onClick={() => { setAblType(t); fetchAblation(t) }}
                id={`ablation-tab-${t}`}
              >
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>
        </div>
        {ablation ? (
          <table className="ablation-table" id="ablation-table">
            <thead>
              <tr>
                <th>Config</th>
                {METRICS.map(m => <th key={m}>{METRIC_LABELS[m]}</th>)}
              </tr>
            </thead>
            <tbody>
              {(ablation.results || []).map((row, i) => {
                const s = row.scores || {}
                return (
                  <tr key={i}>
                    <td style={{ fontWeight: 500 }}>{row.config}</td>
                    {METRICS.map(m => {
                      const v = s[m]
                      return <td key={m} className={v > 0.75 ? 'best' : ''}>{v != null && v >= 0 ? `${(v*100).toFixed(1)}%` : 'N/A'}</td>
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        ) : (
          <p style={{ color:'var(--text-2)', fontSize:'0.85rem' }}>
            No ablation results yet. Run: <code>python -m evaluation.ablation --experiment {ablationType}</code>
          </p>
        )}
      </div>

      {/* History table */}
      {results.length > 0 && (
        <div className="chart-container">
          <h2>Run History</h2>
          <table className="ablation-table" id="history-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Questions</th>
                {METRICS.map(m => <th key={m}>{METRIC_LABELS[m]}</th>)}
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => {
                const s = r.aggregate_scores || {}
                return (
                  <tr key={i}>
                    <td style={{ fontFamily:'var(--font-mono)', fontSize:'0.78rem', color:'var(--text-2)' }}>
                      {new Date(r.timestamp).toLocaleString()}
                    </td>
                    <td>{r.num_questions}</td>
                    {METRICS.map(m => {
                      const v = s[m]
                      return <td key={m}>{v != null && v >= 0 ? `${(v*100).toFixed(1)}%` : '—'}</td>
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
