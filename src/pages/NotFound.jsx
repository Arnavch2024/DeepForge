import React from 'react';
import { Link } from 'react-router-dom';

export default function NotFound() {
	return (
		<div className="builder-hub" style={{ textAlign: 'center', margin: '80px auto' }}>
			<h1 className="hub-title">404 – Page not found</h1>
			<p style={{ color: '#94a3b8', marginTop: 12 }}>The page you are looking for doesn't exist.</p>
			<div style={{ marginTop: 24 }}>
				<Link className="signup-btn" to="/">Go Home</Link>
			</div>
		</div>
	);
}


