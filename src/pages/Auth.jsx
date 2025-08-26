import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, setAuthToken } from '../api/client.js';
import BackButton from '../components/BackButton.jsx';

export default function Auth() {
	const navigate = useNavigate();
	const [mode, setMode] = useState('signup');
	const [form, setForm] = useState({ username: '', email: '', password: '' });
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState('');

	const onChange = (e) => setForm({ ...form, [e.target.name]: e.target.value });

	async function handleSubmit(e) {
		e.preventDefault();
		setLoading(true);
		setError('');
		try {
			let res;
			if (mode === 'signup') {
				res = await api.signup({ username: form.username, email: form.email, password: form.password });
			} else {
				res = await api.login({ email: form.email, password: form.password });
			}
			if (res.token) setAuthToken(res.token);
			navigate('/builder');
		} catch (err) {
			setError(err.message || 'Request failed');
		} finally {
			setLoading(false);
		}
	}

	return (
		<div className="builder-hub" style={{ maxWidth: 480, margin: '80px auto' }}>
			<BackButton />
			<h1 className="hub-title" style={{ textAlign: 'center' }}>{mode === 'signup' ? 'Create account' : 'Log in'}</h1>
			<div className="hub-cards" style={{ padding: 24 }}>
				<form onSubmit={handleSubmit}>
					{mode === 'signup' && (
						<label style={{ display: 'block', marginBottom: 12 }}>
							<span>Username</span>
							<input name="username" value={form.username} onChange={onChange} required className="nav-link" style={{ width: '100%', padding: 10, borderRadius: 8, background: '#0b1220', color: '#e2e8f0', border: '1px solid #1f2937' }} />
						</label>
					)}
					<label style={{ display: 'block', marginBottom: 12 }}>
						<span>Email</span>
						<input name="email" type="email" value={form.email} onChange={onChange} required className="nav-link" style={{ width: '100%', padding: 10, borderRadius: 8, background: '#0b1220', color: '#e2e8f0', border: '1px solid #1f2937' }} />
					</label>
					<label style={{ display: 'block', marginBottom: 12 }}>
						<span>Password</span>
						<input name="password" type="password" value={form.password} onChange={onChange} required className="nav-link" style={{ width: '100%', padding: 10, borderRadius: 8, background: '#0b1220', color: '#e2e8f0', border: '1px solid #1f2937' }} />
					</label>
					{error && <div style={{ color: '#f87171', marginBottom: 12 }}>{error}</div>}
					<button type="submit" className="signup-btn" disabled={loading} style={{ width: '100%' }}>
						{loading ? 'Please wait…' : (mode === 'signup' ? 'Sign Up' : 'Log In')}
					</button>
				</form>
				<div style={{ marginTop: 16, textAlign: 'center' }}>
					<button className="nav-link" onClick={() => setMode(mode === 'signup' ? 'login' : 'signup')}>
						{mode === 'signup' ? 'Have an account? Log in' : "New here? Create an account"}
					</button>
				</div>
			</div>
		</div>
	);
}


