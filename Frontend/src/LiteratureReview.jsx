import React, { useEffect, useMemo, useState } from 'react';
import { api } from './api';

const STORAGE_KEY = 'archstudio_literature_review_v1';

export default function LiteratureReview({ projectId: initialProjectId = '' }) {
  const [projects, setProjects] = useState([]);
  const [projectId, setProjectId] = useState(initialProjectId);
  const [papers, setPapers] = useState(() => {
    try { return (JSON.parse(sessionStorage.getItem(STORAGE_KEY) || '{}').papers || []).filter(paper => paper.type === 'url'); } catch { return []; }
  });
  const [urls, setUrls] = useState('');
  const [files, setFiles] = useState([]);
  const [result, setResult] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem(STORAGE_KEY) || '{}').result || null; } catch { return null; }
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { api.projects().then(items => { setProjects(items); setProjectId(initialProjectId || items.find(p => p.status === 'embedded')?.id || items[0]?.id || ''); }).catch(e => setError(e.message)); }, [initialProjectId]);
  useEffect(() => { sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ papers, result })); }, [papers, result]);
  const addUrl = () => { const value = urls.trim(); if (!value) return; setPapers(items => [...items, { type: 'url', value, label: value }]); setUrls(''); };
  const addFiles = event => { const selected = [...(event.target.files || [])].filter(file => file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')); setFiles(items => [...items, ...selected]); setPapers(items => [...items, ...selected.map(file => ({ type: 'pdf', value: file.name, label: file.name }))]); event.target.value = ''; };
  const removePaper = index => { const paper = papers[index]; setPapers(items => items.filter((_, i) => i !== index)); if (paper?.type === 'pdf') setFiles(items => { const copy = [...items]; const fileIndex = copy.findIndex(file => file.name === paper.value); if (fileIndex >= 0) copy.splice(fileIndex, 1); return copy; }); };
  const generate = async () => { const hasUrl = papers.some(paper => paper.type === 'url'); const hasFile = files.length > 0; if (!projectId || (!hasUrl && !hasFile)) { setError('Select an indexed project and add at least one paper or PDF file.'); return; } setBusy(true); setError(''); setResult(null); try { const response = await api.literatureReview(projectId, papers.filter(p => p.type === 'url').map(p => p.value), files); setResult(response); } catch (e) { setError(e.message); } finally { setBusy(false); } };
  const copy = text => navigator.clipboard?.writeText(text || '');
  const sections = useMemo(() => [['Literature Review', result?.literature_review], ['Research Gap & Motivation', result?.research_gap_motivation], ['Comparison with Existing System', result?.comparison_existing_system]], [result]);
  return <div className="literature-page"><div className="page-header"><div className="eyebrow">LITERATURE REVIEW</div><h1>Compare research with your codebase.</h1><p className="muted">Add multiple public papers or PDFs, then generate copy-ready sections grounded in ArchStudio's indexed project evidence.</p></div>
    <div className="literature-controls"><label className="eyebrow">PROJECT</label><select value={projectId} onChange={e => setProjectId(e.target.value)}>{projects.map(project => <option key={project.id} value={project.id}>{project.name} — {project.status}</option>)}</select><div className="literature-add"><input value={urls} onChange={e => setUrls(e.target.value)} onKeyDown={e => e.key === 'Enter' && addUrl()} placeholder="Paste a public paper URL" /><button className="secondary-button" onClick={addUrl}>Add Paper</button><label className="secondary-button">＋ Add PDF<input type="file" accept="application/pdf,.pdf" multiple hidden onChange={addFiles} /></label></div></div>
    {papers.length > 0 && <div className="literature-sources"><h3>Reference papers</h3>{papers.map((paper, index) => <div className="literature-source" key={`${paper.type}-${paper.value}-${index}`}><span>{paper.type === 'pdf' ? 'PDF' : 'URL'}</span><strong>{paper.label}</strong><button className="text-button" onClick={() => removePaper(index)}>Remove</button></div>)}</div>}
    {error && <div className="error-box">{error}</div>}<button className="primary-button" disabled={busy || !papers.length} onClick={generate}>{busy ? 'Analyzing papers…' : 'Generate Literature Review'}</button>
    <div className="literature-results">{sections.map(([heading, content]) => <section className="literature-result" key={heading}><div className="viewer-head"><h2>{heading}</h2><button className="download" disabled={!content || busy} onClick={() => copy(content)}>Copy</button></div><div className={`result-copy ${busy ? 'result-loading' : ''}`}>{busy ? 'Analyzing the supplied papers and comparing them with the ArchStudio codebase…' : content || 'This section will appear here after you generate the literature analysis.'}</div></section>)}</div>
  </div>;
}
