const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export function getAuthToken() {
	return localStorage.getItem('df_token') || '';
}

export function setAuthToken(token) {
	if (token) localStorage.setItem('df_token', token);
}

export function clearAuthToken() {
	localStorage.removeItem('df_token');
}

async function request(path, options = {}) {
	const headers = options.headers ? { ...options.headers } : {};
	if (!headers['Content-Type'] && options.body) headers['Content-Type'] = 'application/json';
	const token = getAuthToken();
	if (token) headers['Authorization'] = `Bearer ${token}`;
	const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
	const isJson = res.headers.get('content-type')?.includes('application/json');
	const data = isJson ? await res.json() : await res.text();
	if (!res.ok) throw new Error((data && data.error) || res.statusText);
	return data;
}

export const api = {
	signup: (payload) => request('/signup', { method: 'POST', body: JSON.stringify(payload) }),
	login: (payload) => request('/login', { method: 'POST', body: JSON.stringify(payload) }),
	me: () => request('/users/me'),
	updateProfile: (payload) => request('/users/me', { method: 'PUT', body: JSON.stringify(payload) }),
	getMySubscription: () => request('/subscription'),
	subscribe: (plan) => request('/subscribe', { method: 'POST', body: JSON.stringify({ plan }) }),
	cancelSubscription: () => request('/cancel-subscription', { method: 'POST', body: JSON.stringify({}) }),
	createChat: () => request('/chats', { method: 'POST', body: JSON.stringify({}) }),
	listChats: (userId) => request(`/users/${userId}/chats`),
	getChats: () => request('/chats'),
	postChat: (payload) => request('/chats', { method: 'POST', body: JSON.stringify(payload) }),
	clearChats: () => request('/chats', { method: 'DELETE' }),
};

export default api;


