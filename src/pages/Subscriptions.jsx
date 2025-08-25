import React, { useEffect, useState } from 'react';
import { api } from '../api/client.js';

const plans = [
	{ id: 'free', name: 'Free', price: 0, features: ['Basic usage', 'Community support'] },
	{ id: 'pro', name: 'Pro', price: 19.99, features: ['Everything in Free', 'Priority builds', 'Private projects'] },
	{ id: 'enterprise', name: 'Enterprise', price: 99.0, features: ['All Pro features', 'SLA & SSO', 'Dedicated support'] },
];

export default function Subscriptions() {
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState('');
	const [current, setCurrent] = useState(null);
	const [working, setWorking] = useState('');
	const [toast, setToast] = useState('');

	useEffect(() => {
		(async () => {
			try {
				const res = await api.getMySubscription();
				setCurrent(res.subscription);
			} catch (e) {
				setError(e.message || 'Failed to fetch subscription');
			} finally {
				setLoading(false);
			}
		})();
	}, []);

	async function choose(planId) {
		setWorking(planId);
		setError('');
		try {
			await api.subscribe(planId);
			const res = await api.getMySubscription();
			setCurrent(res.subscription);
			setToast('Subscription updated');
			setTimeout(() => setToast(''), 2000);
		} catch (e) {
			setError(e.message || 'Subscription failed');
		} finally {
			setWorking('');
		}
	}

	async function cancel() {
		setWorking('cancel');
		try {
			await api.cancelSubscription();
			setCurrent(null);
			setToast('Subscription cancelled');
			setTimeout(() => setToast(''), 2000);
		} catch (e) {
			setError(e.message || 'Cancel failed');
		} finally {
			setWorking('');
		}
	}

	return (
		<div className="builder-hub" style={{ maxWidth: 1100, margin: '80px auto' }}>
			<h1 className="hub-title" style={{ color: '#111827' }}>Choose your plan</h1>
			{toast && <div style={{ background: '#ecfdf5', color: '#065f46', padding: 12, borderRadius: 8, marginTop: 12 }}>{toast}</div>}
			{error && <div style={{ background: '#fee2e2', color: '#991b1b', padding: 12, borderRadius: 8, marginTop: 12 }}>{error}</div>}
			{loading ? (
				<div style={{ textAlign: 'center', padding: 40 }}>Loading…</div>
			) : (
				<div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 16, marginTop: 24 }}>
					{plans.map(p => {
						const isCurrent = current && p.name.toLowerCase() === (current.name || '').toLowerCase();
						return (
							<div key={p.id} style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 20 }}>
								<div style={{ fontWeight: 700, fontSize: 18, color: '#111827' }}>{p.name}</div>
								<div style={{ fontSize: 28, marginTop: 8, color: '#111827' }}>{p.price === 0 ? 'Free' : `$${p.price}/mo`}</div>
								<ul style={{ marginTop: 12, color: '#374151' }}>
									{p.features.map((f, i) => (<li key={i} style={{ marginBottom: 6 }}>• {f}</li>))}
								</ul>
								<div style={{ marginTop: 16 }}>
									{isCurrent ? (
										<button className="signup-btn" onClick={cancel} disabled={working==='cancel'} style={{ background: '#ef4444' }}>{working==='cancel' ? 'Cancelling…' : 'Cancel plan'}</button>
									) : (
										<button className="signup-btn" onClick={() => choose(p.id)} disabled={working===p.id} style={{ background: '#22c55e' }}>{working===p.id ? 'Updating…' : 'Choose'}</button>
									)}
								</div>
							</div>
						);
					})}
				</div>
			)}
		</div>
	);
}


