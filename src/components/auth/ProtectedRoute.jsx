import React from 'react';
import { Navigate } from 'react-router-dom';
import { getAuthToken } from '../../api/client.js';

export default function ProtectedRoute({ children }) {
	const token = getAuthToken();
	if (!token) return <Navigate to="/auth" replace />;
	return children;
}


