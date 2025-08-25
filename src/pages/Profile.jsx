import React, { useEffect, useState } from 'react';
import { api } from '../api/client.js';

export default function Profile() {
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState('');
	const [user, setUser] = useState(null);
	const [form, setForm] = useState({ username: '', password: '' });
	const [saving, setSaving] = useState(false);
	const [success, setSuccess] = useState('');

	useEffect(() => {
		(async () => {
			try {
				const res = await api.me();
				setUser(res.user || res);
				setForm({ username: (res.user || res)?.username || '', password: '' });
			} catch (e) {
				setError(e.message || 'Failed to load profile');
			} finally {
				setLoading(false);
			}
		})();
	}, []);

	function onChange(e) {
		setForm({ ...form, [e.target.name]: e.target.value });
	}

	async function onSave(e) {
		e.preventDefault();
		setSaving(true);
		setError('');
		setSuccess('');
		try {
			const payload = { username: form.username };
			if (form.password) payload.password = form.password;
			const res = await api.updateProfile(payload);
			setUser(res.user || res);
			setForm({ username: (res.user || res)?.username || '', password: '' });
			setSuccess('Profile updated');
		} catch (e) {
			setError(e.message || 'Update failed');
		} finally {
			setSaving(false);
		}
	}

	if (loading) return <div className="builder-hub" style={{ textAlign: 'center', margin: '80px auto' }}>Loading…</div>;
	if (error) return <div className="builder-hub" style={{ textAlign: 'center', margin: '80px auto', color: '#f87171' }}>{error}</div>;

	return (
		<div className="builder-hub" style={{ maxWidth: 640, margin: '80px auto' }}>
			<h1 className="hub-title">Profile</h1>
			<div className="hub-cards" style={{ padding: 24 }}>
				<div style={{ marginBottom: 16, color: '#94a3b8' }}>
					<div><b>Email:</b> {user?.email}</div>
					<div><b>Username:</b> {user?.username}</div>
					{user?.created_at && <div><b>Joined:</b> {new Date(user.created_at).toLocaleString()}</div>}
					{/* Subscription badge placeholder - optional enhancement: fetch summary */}
				</div>
				<form onSubmit={onSave}>
					<label style={{ display: 'block', marginBottom: 12 }}>
						<span>Username</span>
						<input name="username" value={form.username} onChange={onChange} className="nav-link" style={{ width: '100%', padding: 10, borderRadius: 8, background: '#0b1220', color: '#e2e8f0', border: '1px solid #1f2937' }} />
					</label>
					<label style={{ display: 'block', marginBottom: 12 }}>
						<span>New Password</span>
						<input name="password" type="password" value={form.password} onChange={onChange} className="nav-link" placeholder="Leave blank to keep current" style={{ width: '100%', padding: 10, borderRadius: 8, background: '#0b1220', color: '#e2e8f0', border: '1px solid #1f2937' }} />
					</label>
					{error && <div style={{ color: '#f87171', marginBottom: 8 }}>{error}</div>}
					{success && <div style={{ color: '#34d399', marginBottom: 8 }}>{success}</div>}
					<button type="submit" className="signup-btn" disabled={saving}>{saving ? 'Saving…' : 'Save Changes'}</button>
				</form>
			</div>
		</div>
	);
}


