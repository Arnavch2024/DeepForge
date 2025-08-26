import React, { useEffect, useRef, useState } from 'react';
import { api } from '../api/client.js';
import BackButton from '../components/BackButton.jsx';

export default function Chats() {
	const [messages, setMessages] = useState([]);
	const [input, setInput] = useState('');
	const [loading, setLoading] = useState(true);
	const [working, setWorking] = useState(false);
	const [error, setError] = useState('');
	const [mode, setMode] = useState(localStorage.getItem('df_chat_mode') || 'cnn');
	const endRef = useRef(null);

	useEffect(() => {
		(async () => {
			try {
				const res = await api.getChats();
				setMessages(res.chats || []);
			} catch (e) {
				setError(e.message || 'Failed to load chats');
			} finally {
				setLoading(false);
			}
		})();
	}, []);

	useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);
	useEffect(() => { localStorage.setItem('df_chat_mode', mode); }, [mode]);

	async function send() {
		if (!input.trim()) return;
		setWorking(true);
		setError('');
		const userMsg = { role: 'user', message: input };
		try {
			const saved = await api.postChat(userMsg);
			setMessages(prev => [...prev, saved.message]);
			setInput('');
			// Mock assistant echo; in real app call LLM then persist assistant reply
			// Call backend per mode (mocked here by storing assistant echo)
			const endpoint = mode === 'rag' ? '/chat-rag' : '/chat-cnn';
			try { await fetch((import.meta.env.VITE_API_BASE||'http://localhost:8000')+endpoint, { method:'POST', headers:{'Content-Type':'application/json', ...(localStorage.getItem('df_token')?{Authorization:`Bearer ${localStorage.getItem('df_token')}`}:{})}, body: JSON.stringify({ prompt: userMsg.message }) }); } catch (e) { /* Mock call, can ignore */ }
			const assistant = await api.postChat({ role: 'assistant', message: `[${mode.toUpperCase()}] You said: ${userMsg.message}` });
			setMessages(prev => [...prev, assistant.message]);
		} catch (e) {
			setError(e.message || 'Send failed');
		} finally {
			setWorking(false);
		}
	}

	async function clearAll() {
		setWorking(true);
		try {
			await api.clearChats();
			setMessages([]);
		} catch (e) {
			setError(e.message || 'Clear failed');
		} finally {
			setWorking(false);
		}
	}

	return (
		<div className="builder-hub" style={{ maxWidth: 800, margin: '80px auto' }}>
			<BackButton />
			<h1 className="hub-title" style={{ color: '#111827' }}>Chats</h1>
			<div style={{ marginTop: 8, display: 'flex', gap: 8, alignItems: 'center' }}>
				<span style={{ color: '#6b7280' }}>Mode:</span>
				<div style={{ display: 'inline-flex', background: '#f3f4f6', border: '1px solid #e5e7eb', borderRadius: 999 }}>
					<button onClick={() => setMode('cnn')} className="nav-link" style={{ padding: '6px 12px', borderRadius: 999, background: mode==='cnn'?'#22c55e':'transparent', color: mode==='cnn'?'#fff':'#111827' }}>CNN</button>
					<button onClick={() => setMode('rag')} className="nav-link" style={{ padding: '6px 12px', borderRadius: 999, background: mode==='rag'?'#22c55e':'transparent', color: mode==='rag'?'#fff':'#111827' }}>RAG</button>
				</div>
			</div>
			{error && <div style={{ background: '#fee2e2', color: '#991b1b', padding: 12, borderRadius: 8, marginTop: 12 }}>{error}</div>}
			<div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, minHeight: 400, padding: 16, marginTop: 16 }}>
				{loading ? (
					<div>Loading…</div>
				) : (
					<div>
						{messages.map(m => (
							<div key={`${m.id}-${m.created_at}`} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', marginBottom: 8 }}>
								<div style={{
									background: m.role === 'user' ? '#dcfce7' : '#f3f4f6',
									border: '1px solid #e5e7eb',
									color: '#111827',
									borderRadius: 12,
									padding: '8px 12px',
									maxWidth: '75%'
								}}>
									{m.message}
								</div>
							</div>
						))}
						<div ref={endRef} />
					</div>
				)}
			</div>
			<div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
				<input value={input} onChange={e => setInput(e.target.value)} placeholder="Type a message" style={{ flex: 1, padding: 12, border: '1px solid #e5e7eb', borderRadius: 10 }} />
				<button className="signup-btn" onClick={send} disabled={working} style={{ background: '#22c55e' }}>{working ? 'Sending…' : 'Send'}</button>
				<button className="signup-btn" onClick={clearAll} disabled={working} style={{ background: '#ef4444' }}>{working ? 'Clearing…' : 'Clear'}</button>
			</div>
		</div>
	);
}


