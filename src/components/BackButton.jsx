import React from "react";
import { useNavigate } from "react-router-dom";

export default function BackButton({ style }) {
  const navigate = useNavigate();
  return (
    <button
      onClick={() => navigate(-1)}
      style={{
        marginBottom: 16,
        padding: "8px 16px",
        borderRadius: 8,
        background: "#1f2937",
        color: "#e2e8f0",
        border: "none",
        cursor: "pointer",
        ...style,
      }}
    >
      ← Back
    </button>
  );
}
