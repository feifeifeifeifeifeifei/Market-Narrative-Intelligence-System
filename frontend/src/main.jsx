import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  Database,
  Loader2,
  Search,
  ShieldCheck,
} from 'lucide-react';
import './styles.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

const EXAMPLES = [
  'China tariff threats and market reaction',
  'Oil energy policy posts',
  'War defense spending narratives',
  'Fed rates and dollar comments',
];

function App() {
  const [question, setQuestion] = useState(EXAMPLES[0]);
  const [topK, setTopK] = useState(10);
  const [topics, setTopics] = useState({});
  const [health, setHealth] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API_BASE}/api/health`).then((res) => res.json()).then(setHealth).catch(() => setHealth({ status: 'offline' }));
    fetch(`${API_BASE}/api/topics`).then((res) => res.json()).then(setTopics).catch(() => setTopics({}));
  }, []);

  async function analyze() {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: topK }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(errorMessage(body.detail));
      setResult(body);
    } catch (err) {
      setError(err.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  }

  function submitAnalyze(event) {
    event.preventDefault();
    if (!loading) analyze();
  }

  function submitOnEnter(event) {
    if (event.key === 'Enter') {
      event.preventDefault();
      if (!loading) analyze();
    }
  }

  function updateTopK(event) {
    const parsed = Number.parseInt(event.target.value, 10);
    if (!Number.isFinite(parsed)) return;
    setTopK(Math.min(50, Math.max(1, parsed)));
  }

  const topicRows = Object.entries(topics).slice(0, 12);
  const normalizedQuestion = question.trim().replace(/\s+/g, ' ');
  const questionWasRedacted = result?.redacted_question && result.redacted_question !== normalizedQuestion;

  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <h1>Market Narrative Intelligence</h1>
          <p>Truth Social narratives, similar-event retrieval, and daily open-to-close market reactions.</p>
        </div>
        <StatusBadge health={health} />
      </section>

      <form className="query-band" onSubmit={submitAnalyze}>
        <div className="query-input">
          <Search size={18} aria-hidden="true" />
          <input
            aria-label="Market narrative question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={submitOnEnter}
          />
        </div>
        <label className="topk">
          <span>Top K</span>
          <input
            aria-label="Number of similar posts"
            type="number"
            min="1"
            max="50"
            value={topK}
            onChange={updateTopK}
          />
        </label>
        <button type="submit" className="primary" disabled={loading}>
          {loading ? <Loader2 className="spin" size={18} /> : <Brain size={18} />}
          Analyze
        </button>
      </form>

      <section className="examples" aria-label="Example queries">
        {EXAMPLES.map((example) => (
          <button key={example} onClick={() => setQuestion(example)}>{example}</button>
        ))}
      </section>

      {error && <Notice tone="danger" icon={<AlertTriangle size={18} />} text={error} />}
      {result?.query_type === 'refusal' && <Notice tone="warn" icon={<ShieldCheck size={18} />} text={result.summary} />}
      {questionWasRedacted && (
        <Notice
          tone="warn"
          icon={<ShieldCheck size={18} />}
          text={`Question was redacted before analysis: "${result.redacted_question}"`}
        />
      )}

      <section className="workspace">
        <div className="main-column">
          <SummaryPanel result={result} loading={loading} />
          <SimilarPosts posts={result?.similar_posts || []} hasResult={Boolean(result && result.query_type !== 'refusal')} />
        </div>
        <aside className="side-column">
          <TickerPanel tickers={result?.selected_tickers || []} selectedTopic={result?.selected_topic} />
          <MarketReaction rows={result?.market_reaction || []} />
          <TopicMap rows={topicRows} />
        </aside>
      </section>
    </main>
  );
}

function StatusBadge({ health }) {
  const ok = health?.status === 'ok';
  const degraded = health?.status === 'degraded';
  const label = ok
    ? `API ready · ${health.collection}`
    : degraded
      ? `API degraded · ${health.collection}`
      : 'API offline';
  return (
    <div className={`status ${ok ? 'ok' : 'offline'}`}>
      {ok ? <CheckCircle2 size={17} /> : <Database size={17} />}
      <span>{label}</span>
    </div>
  );
}

function Notice({ tone, icon, text }) {
  const role = tone === 'danger' ? 'alert' : 'status';
  return <div className={`notice ${tone}`} role={role}>{icon}<span>{text}</span></div>;
}

function SummaryPanel({ result, loading }) {
  return (
    <section className="panel summary-panel">
      <header><Activity size={18} /><h2>Analysis</h2></header>
      {loading && <p className="muted">Running retrieval and market reaction analysis...</p>}
      {!loading && !result && <p className="muted">Submit a market narrative question to populate this workspace.</p>}
      {result && result.query_type !== 'refusal' && (
        <div>
          <p className="summary">{result.summary}</p>
          <dl className="metrics">
            <div><dt>Retrieved</dt><dd>{result.retrieved_count ?? 0}</dd></div>
            <div><dt>Topic</dt><dd>{result.selected_topic || 'none'}</dd></div>
            <div><dt>Guardrail</dt><dd>{result.guardrail_decision}</dd></div>
          </dl>
        </div>
      )}
    </section>
  );
}

function TickerPanel({ tickers, selectedTopic }) {
  return (
    <section className="panel">
      <header><Database size={18} /><h2>Selected Tickers</h2></header>
      <p className="mini-label">{selectedTopic || 'No topic selected'}</p>
      <div className="ticker-list">
        {tickers.length ? tickers.map((ticker) => <span key={ticker}>{ticker}</span>) : <span className="empty-pill">none</span>}
      </div>
    </section>
  );
}

function MarketReaction({ rows }) {
  return (
    <section className="panel">
      <header><BarChart3 size={18} /><h2>Market Reaction</h2></header>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Ticker</th><th>Avg</th><th>Median</th><th>N</th></tr></thead>
          <tbody>
            {rows.length ? rows.map((row) => (
              <tr key={row.ticker}>
                <td>{row.ticker}</td>
                <td>{formatPct(row.avg_daily_return)}</td>
                <td>{formatPct(row.median_daily_return)}</td>
                <td>{row.sample_size}</td>
              </tr>
            )) : <tr><td colSpan="4" className="muted">No rows</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function SimilarPosts({ posts, hasResult }) {
  return (
    <section className="panel">
      <header><Search size={18} /><h2>Similar Posts</h2></header>
      <div className="post-list">
        {posts.length ? posts.map((post) => (
          <article className="post-row" key={post.post_id}>
            <div className="post-meta">
              <span>{formatDate(post.date)}</span>
              <span>{post.primary_topic}</span>
              <span>{score(post.similarity_score)}</span>
            </div>
            <p>{post.cleaned_text}</p>
          </article>
        )) : <p className="muted">{hasResult ? 'No posts matched this query.' : 'No retrieved posts yet.'}</p>}
      </div>
    </section>
  );
}

function TopicMap({ rows }) {
  return (
    <section className="panel topic-map">
      <header><ShieldCheck size={18} /><h2>Topic Map</h2></header>
      {rows.map(([topic, tickers]) => (
        <div className="topic-row" key={topic}>
          <span>{topic}</span>
          <small>{tickers.join(', ')}</small>
        </div>
      ))}
    </section>
  );
}

function formatPct(value) {
  if (value === null || value === undefined) return 'n/a';
  return `${(value * 100).toFixed(2)}%`;
}

function score(value) {
  if (value === null || value === undefined) return 'score n/a';
  return `score ${value.toFixed(3)}`;
}

function formatDate(value) {
  if (!value) return 'no date';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'no date' : date.toISOString().slice(0, 10);
}

function errorMessage(detail) {
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || String(item)).join(', ');
  }
  return detail || 'Request failed';
}

createRoot(document.getElementById('root')).render(<App />);
