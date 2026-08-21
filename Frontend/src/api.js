const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export function getToken() { return localStorage.getItem('archstudio_token'); }
export function setToken(token) { token ? localStorage.setItem('archstudio_token', token) : localStorage.removeItem('archstudio_token'); }

async function request(path, options = {}) {
  const headers = { ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...(options.headers || {}) };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  let response;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    response = await fetch(`${API_URL}${path}`, { ...options, headers });
    if (![502, 503, 504].includes(response.status) || attempt === 2) break;
    await new Promise(resolve => setTimeout(resolve, 700 * (attempt + 1)));
  }
  let body = null;
  try { body = await response.json(); } catch { /* binary/empty */ }
  if (response.status === 401) { setToken(null); window.dispatchEvent(new Event('auth-expired')); }
  if (!response.ok) throw new Error(body?.detail || body?.message || `Request failed (${response.status})`);
  return body;
}

export const api = {
  login: (data) => request('/auth/login', { method: 'POST', body: JSON.stringify(data) }),
  register: (data) => request('/auth/register', { method: 'POST', body: JSON.stringify(data) }),
  forgotPassword: (data) => request('/auth/forgot-password', { method: 'POST', body: JSON.stringify(data) }),
  updatePassword: (data) => request('/auth/update-password', { method: 'POST', body: JSON.stringify(data) }),
  oauth: (provider) => request(`/auth/oauth/${provider}`),
  me: () => request('/me'),
  projects: () => request('/projects'),
  project: (id) => request(`/projects/${id}`),
  deleteProject: (id) => request(`/projects/${id}`, { method: 'DELETE' }),
  upload: (file, name) => { const form = new FormData(); form.append('file', file); if (name) form.append('name', name); return request('/projects/upload', { method: 'POST', body: form }); },
  github: (url, name) => request('/projects/github', { method: 'POST', body: JSON.stringify({ url, name: name || null }) }),
  retrieve: (id) => request(`/projects/${id}/retrieve`, { method: 'POST' }),
  process: (id) => request(`/projects/${id}/process`, { method: 'POST' }),
  embed: (id) => request(`/projects/${id}/embed`, { method: 'POST' }),
  search: (id, data) => request(`/projects/${id}/search`, { method: 'POST', body: JSON.stringify(data) }),
  queueArtifact: (id, kind) => {
    const endpoints = {
      report: 'reports',
      ieee_report: 'ieee-report',
      research_paper: 'research-paper',
      database_diagram: 'database-diagram',
      architecture_diagram: 'architecture-diagram',
    };
    const endpoint = endpoints[kind];
    if (!endpoint) throw new Error('Unsupported artifact type.');
    return request(`/projects/${id}/${endpoint}`, { method: 'POST' });
  },
  artifact: (id, artifactId) => request(`/projects/${id}/artifacts/${artifactId}`),
  artifacts: (id) => request(`/projects/${id}/artifacts`),
  updateArtifact: (id, artifactId, data) => request(`/projects/${id}/artifacts/${artifactId}`, { method: 'PATCH', body: JSON.stringify(data) }),
  downloadUrl: (id, artifactId, format) => `${API_URL}/projects/${id}/artifacts/${artifactId}/download?format=${format}`,
  chat: (id, query, top_k = 5) => request(`/projects/${id}/chat`, { method: 'POST', body: JSON.stringify({ query, top_k }) }),
  literatureReview: (id, urls, files) => { const form = new FormData(); form.append('urls', JSON.stringify(urls)); (files || []).forEach(file => form.append('files', file)); return request(`/projects/${id}/literature-review`, { method: 'POST', body: form }); },
};
