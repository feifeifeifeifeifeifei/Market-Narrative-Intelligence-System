import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  Database,
  ListFilter,
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

const EMPTY_FILTERS = {
  tone: '',
  market_relevance: '',
  policy_direction: '',
};

const FILTER_LABELS = {
  tone: 'Tone',
  market_relevance: 'Relevance',
  policy_direction: 'Policy',
};

function App() {
  const [question, setQuestion] = useState(EXAMPLES[0]);
  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [filterOptions, setFilterOptions] = useState({});
  const [topics, setTopics] = useState({});
  const [health, setHealth] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API_BASE}/api/health`).then((res) => res.json()).then(setHealth).catch(() => setHealth({ status: 'offline' }));
    fetch(`${API_BASE}/api/topics`).then((res) => res.json()).then(setTopics).catch(() => setTopics({}));
    fetch(`${API_BASE}/api/filter-options`).then((res) => res.json()).then(setFilterOptions).catch(() => setFilterOptions({}));
  }, []);

  async function analyze() {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, ...activeFilters(filters) }),
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

  function updateFilter(field, value) {
    setFilters((current) => ({ ...current, [field]: value }));
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
        <button type="submit" className="primary" disabled={loading}>
          {loading ? <Loader2 className="spin" size={18} /> : <Brain size={18} />}
          Analyze
        </button>
      </form>

      <section className="filter-band" aria-label="Retrieval filters">
        <div className="filter-title"><ListFilter size={17} /><span>Filters</span></div>
        {Object.entries(FILTER_LABELS).map(([field, label]) => (
          <label className="filter-control" key={field}>
            <span>{label}</span>
            <select
              aria-label={`${label} filter`}
              value={filters[field]}
              onChange={(event) => updateFilter(field, event.target.value)}
            >
              <option value="">All</option>
              {(filterOptions[field] || []).map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </label>
        ))}
      </section>

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
          <SimilarPosts
            posts={result?.similar_posts || []}
            hasResult={Boolean(result && result.query_type !== 'refusal')}
            clusteringApplied={Boolean(result?.clustering_applied)}
          />
        </div>
        <aside className="side-column">
          <NarrativesPanel narratives={result?.narratives || []} noiseCount={result?.noise_count || 0} />
          <TickerPanel tickers={result?.selected_tickers || []} selectedTopics={result?.selected_topics || []} />
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
            <div><dt>Top Topics</dt><dd>{formatTopicMix(result.selected_topics)}</dd></div>
            <div><dt>Guardrail</dt><dd>{result.guardrail_decision}</dd></div>
            <div><dt>Filters</dt><dd>{formatFilters(result.filters)}</dd></div>
          </dl>
        </div>
      )}
    </section>
  );
}

function TickerPanel({ tickers, selectedTopics }) {
  return (
    <section className="panel">
      <header><Database size={18} /><h2>Selected Tickers</h2></header>
      <p className="mini-label">{selectedTopics.length ? `From ${formatTopicMix(selectedTopics, 2)}` : 'No retrieved topic mix'}</p>
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

function SimilarPosts({ posts, hasResult, clusteringApplied }) {
  if (!posts.length) {
    return (
      <section className="panel">
        <header><Search size={18} /><h2>Similar Posts</h2></header>
        <p className="muted">{hasResult ? 'No posts matched this query.' : 'No retrieved posts yet.'}</p>
      </section>
    );
  }

  if (!clusteringApplied) {
    return (
      <section className="panel">
        <header><Search size={18} /><h2>Similar Posts</h2></header>
        <div className="post-list">
          {posts.map((post) => <PostRow key={post.post_id} post={post} />)}
        </div>
      </section>
    );
  }

  const { clusters, noise } = groupByCluster(posts);
  return (
    <section className="panel">
      <header><Search size={18} /><h2>Similar Posts by Narrative</h2></header>
      {clusters.map(([label, group]) => (
        <div className="cluster-block" key={label}>
          <div className="cluster-head">
            <span className="cluster-tag">Narrative {Number(label) + 1}</span>
            <small>{group[0].primary_topic} · {group.length} posts</small>
          </div>
          <div className="post-list">
            {group.map((post) => <PostRow key={post.post_id} post={post} />)}
          </div>
        </div>
      ))}
      {noise.length > 0 && (
        <div className="cluster-block noise-block">
          <div className="cluster-head">
            <span className="cluster-tag noise">Outliers</span>
            <small>excluded from aggregates · {noise.length} posts</small>
          </div>
          <div className="post-list">
            {noise.map((post) => <PostRow key={post.post_id} post={post} />)}
          </div>
        </div>
      )}
    </section>
  );
}

function PostRow({ post }) {
  return (
    <article className={`post-row${post.is_noise ? ' noise' : ''}`}>
      <div className="post-meta">
        <span>{formatDate(post.date)}</span>
        <span>{post.primary_topic}</span>
        {post.tone && <span>{post.tone}</span>}
        {post.policy_direction && <span>{post.policy_direction}</span>}
        <span>{score(post.similarity_score)}</span>
      </div>
      <p>{post.cleaned_text}</p>
    </article>
  );
}

function NarrativesPanel({ narratives, noiseCount }) {
  if (!narratives.length) return null;
  return (
    <section className="panel">
      <header><Brain size={18} /><h2>Narratives</h2></header>
      <div className="narrative-list">
        {narratives.map((narrative) => (
          <div className="narrative-row" key={narrative.cluster_id}>
            <div className="narrative-top">
              <span>{narrative.dominant_topic || 'mixed'}</span>
              <small>{narrative.size} posts · avg {score(narrative.avg_similarity)}</small>
            </div>
            <p className="muted">{narrative.representative_text}</p>
          </div>
        ))}
      </div>
      {noiseCount > 0 && <p className="mini-label">Outliers excluded: {noiseCount}</p>}
    </section>
  );
}

function groupByCluster(posts) {
  const clusterMap = new Map();
  const noise = [];
  for (const post of posts) {
    if (post.is_noise || post.cluster_label === null || post.cluster_label === undefined) {
      if (post.is_noise) noise.push(post);
      else pushToCluster(clusterMap, post);
      continue;
    }
    pushToCluster(clusterMap, post);
  }
  const clusters = [...clusterMap.entries()].sort((a, b) => Number(a[0]) - Number(b[0]));
  return { clusters, noise };
}

function pushToCluster(clusterMap, post) {
  const key = post.cluster_label ?? 0;
  if (!clusterMap.has(key)) clusterMap.set(key, []);
  clusterMap.get(key).push(post);
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

function activeFilters(filters) {
  return Object.fromEntries(Object.entries(filters).filter(([, value]) => value));
}

function formatFilters(filters) {
  const entries = Object.entries(filters || {});
  if (!entries.length) return 'all';
  return entries.map(([field, value]) => `${FILTER_LABELS[field] || field}: ${value}`).join(', ');
}

function formatTopicMix(topics, limit = 3) {
  const rows = (topics || []).slice(0, limit);
  if (!rows.length) return 'none';
  return rows.map((item) => `${item.primary_topic} (${item.count})`).join(', ');
}

function errorMessage(detail) {
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || String(item)).join(', ');
  }
  return detail || 'Request failed';
}

createRoot(document.getElementById('root')).render(<App />);
