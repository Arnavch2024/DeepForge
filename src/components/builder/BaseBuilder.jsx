import React, { useCallback, useMemo, useRef, useState, useEffect } from 'react';
import ReactFlow, {
  Background,
  Controls,
  addEdge,
  useEdgesState,
  useNodesState
} from 'reactflow';
import 'reactflow/dist/style.css';
import { v4 as uuidv4 } from 'uuid';
import '../../styles/builder.css';

const defaultNodeStyle = {
  border: '1px solid #1f2937',
  borderRadius: 10,
  padding: 10,
  background: '#0b1220',
  color: '#e2e8f0'
};

function getDefaultParamsForType(schemas, type) {
  const schema = schemas?.[type];
  if (!schema?.fields) return {};
  const defaults = {};
  schema.fields.forEach((f) => {
    defaults[f.key] = f.default ?? (f.type === 'number' ? 0 : '');
  });
  return defaults;
}

export default function BaseBuilder({ title, palette, storageKey, schemas, builderType }) {
  const loadFromStorage = useCallback(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      if (!raw) return { nodes: [], edges: [] };
      const parsed = JSON.parse(raw);
      return { nodes: parsed.nodes ?? [], edges: parsed.edges ?? [] };
    } catch {
      return { nodes: [], edges: [] };
    }
  }, [storageKey]);

  const saveToStorage = useCallback((nodes, edges) => {
    localStorage.setItem(storageKey, JSON.stringify({ nodes, edges }));
  }, [storageKey]);

  const initial = useMemo(loadFromStorage, [loadFromStorage]);
  const [nodes, setNodes, onNodesChange] = useNodesState(initial.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initial.edges);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const reactFlowWrapperRef = useRef(null);

  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [activeTab, setActiveTab] = useState('inspector');

  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');

  const [hoverCard, setHoverCard] = useState({ visible: false, x: 0, y: 0, type: null });

  // Generated code state
  const [generatedCode, setGeneratedCode] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [genError, setGenError] = useState(null);
  const genAbortRef = useRef(null);

  const onConnect = useCallback((connection) => {
    setEdges((eds) => addEdge({ ...connection, animated: true }, eds));
  }, [setEdges]);

  const onDragStart = useCallback((event, item) => {
    setHoverCard({ visible: false, x: 0, y: 0, type: null });
    event.dataTransfer.setData('application/reactflow', JSON.stringify(item));
    event.dataTransfer.effectAllowed = 'move';
  }, []);

  const onDrop = useCallback((event) => {
    event.preventDefault();
    const raw = event.dataTransfer.getData('application/reactflow');
    if (!raw) return;
    const item = JSON.parse(raw);
    const bounds = reactFlowWrapperRef.current.getBoundingClientRect();
    const position = reactFlowInstance.project({
      x: event.clientX - bounds.left,
      y: event.clientY - bounds.top,
    });
    const params = getDefaultParamsForType(schemas, item.type);
    const newNode = {
      id: uuidv4(),
      type: 'default',
      position,
      data: { label: item.label, type: item.type, params },
      style: defaultNodeStyle,
    };
    setNodes((nds) => nds.concat(newNode));
  }, [reactFlowInstance, setNodes, schemas]);

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const handleSave = useCallback(() => {
    saveToStorage(nodes, edges);
  }, [nodes, edges, saveToStorage]);

  const handleClear = useCallback(() => {
    setNodes([]);
    setEdges([]);
    saveToStorage([], []);
    setSelectedNodeId(null);
  }, [setNodes, setEdges, saveToStorage]);

  const handleSelectionChange = useCallback(({ nodes: selectedNodes }) => {
    setSelectedNodeId(selectedNodes?.[0]?.id ?? null);
  }, []);

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedNodeId) || null,
    [nodes, selectedNodeId]
  );

  function updateNodeParam(nodeId, key, value, cast) {
    setNodes((nds) => nds.map((n) => {
      if (n.id !== nodeId) return n;
      const nextParams = { ...(n.data?.params || {}) };
      nextParams[key] = cast === 'number' ? Number(value) : value;
      return { ...n, data: { ...(n.data || {}), params: nextParams } };
    }));
  }

  const onSendChat = useCallback(() => {
    const trimmed = chatInput.trim();
    if (!trimmed) return;
    const userMsg = { role: 'user', content: trimmed, id: uuidv4() };
    setChatMessages((msgs) => [...msgs, userMsg]);
    setChatInput('');
    setTimeout(() => {
      const response = {
        role: 'assistant',
        id: uuidv4(),
        content: 'This is a stubbed chatbot response. Wire this to your RAG backend API.'
      };
      setChatMessages((msgs) => [...msgs, response]);
    }, 300);
  }, [chatInput]);

  const showHover = useCallback((e, item) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setHoverCard({
      visible: true,
      x: rect.right + 12,
      y: rect.top + window.scrollY,
      type: item.type
    });
  }, []);

  const moveHover = useCallback((e) => {
    setHoverCard((hc) => hc.visible ? { ...hc, y: e.clientY + window.scrollY - 20 } : hc);
  }, []);

  const hideHover = useCallback(() => {
    setHoverCard({ visible: false, x: 0, y: 0, type: null });
  }, []);

  const hoverSchema = hoverCard.visible && hoverCard.type ? schemas?.[hoverCard.type] : null;

  // Auto-generate backend code when graph changes (debounced)
  useEffect(() => {
    const nodesPayload = nodes.map((n) => ({
      id: n.id,
      type: n.data?.type || 'default',
      label: n.data?.label || '',
      params: n.data?.params || {}
    }));
    const edgesPayload = edges.map((e) => ({ id: e.id, source: e.source, target: e.target }));

    // Skip if no builderType
    if (!builderType) return;

    // Abort previous in-flight request
    if (genAbortRef.current) {
      genAbortRef.current.abort();
    }

    const controller = new AbortController();
    genAbortRef.current = controller;

    const timer = setTimeout(async () => {
      try {
        setIsGenerating(true);
        setGenError(null);
        const res = await fetch('http://localhost:8000/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ type: builderType, graph: { nodes: nodesPayload, edges: edgesPayload } }),
          signal: controller.signal,
        });
        if (!res.ok) {
          const msg = await res.text();
          throw new Error(msg || 'Failed to generate code');
        }
        const data = await res.json();
        setGeneratedCode(data.code || '');
      } catch (err) {
        if (err.name !== 'AbortError') {
          setGenError(err.message || String(err));
        }
      } finally {
        setIsGenerating(false);
      }
    }, 400);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [nodes, edges, builderType]);

  const copyCode = useCallback(() => {
    if (!generatedCode) return;
    navigator.clipboard.writeText(generatedCode);
  }, [generatedCode]);

  return (
    <div className="builder-root">
      <div className="builder-header">
        <div className="builder-title">{title}</div>
        <div className="builder-actions">
          <button className="builder-btn" onClick={handleSave}>Save</button>
          <button className="builder-btn" onClick={handleClear}>Clear</button>
        </div>
      </div>
      <div className="builder-container three-cols">
        <aside className="builder-sidebar">
          <div className="palette-title">Palette</div>
          {palette.map((item) => (
            <div
              key={item.type}
              className="node-palette-item card"
              draggable
              onDragStart={(e) => onDragStart(e, item)}
              onMouseEnter={(e) => showHover(e, item)}
              onMouseMove={moveHover}
              onMouseLeave={hideHover}
            >
              <div className="node-title">{item.label}</div>
              <div className="node-sub">{item.type}</div>
            </div>
          ))}
        </aside>
        <div className="builder-canvas" ref={reactFlowWrapperRef}>
          <ReactFlow
            fitView
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={setReactFlowInstance}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onSelectionChange={handleSelectionChange}
          >
            <Background />
            <Controls />
          </ReactFlow>
        </div>
        <aside className="builder-right">
          <div className="panel-tabs">
            <button
              className={"tab" + (activeTab === 'inspector' ? ' active' : '')}
              onClick={() => setActiveTab('inspector')}
            >Inspector</button>
            <button
              className={"tab" + (activeTab === 'chat' ? ' active' : '')}
              onClick={() => setActiveTab('chat')}
            >Chat</button>
            <button
              className={"tab" + (activeTab === 'code' ? ' active' : '')}
              onClick={() => setActiveTab('code')}
            >Code</button>
          </div>
          {activeTab === 'inspector' && (
            <div className="inspector">
              {!selectedNode && (
                <div className="empty">Select a node to edit its parameters</div>
              )}
              {selectedNode && (
                <div>
                  <div className="inspector-title">{selectedNode.data?.label}</div>
                  <div className="inspector-sub">{selectedNode.data?.type}</div>
                  <div className="inspector-form">
                    {(schemas?.[selectedNode.data?.type]?.fields || []).map((field) => (
                      <div className="form-group" key={field.key}>
                        <label className="form-label" htmlFor={field.key}>{field.label}</label>
                        {field.type === 'select' ? (
                          <select
                            id={field.key}
                            className="form-input"
                            value={selectedNode.data?.params?.[field.key] ?? ''}
                            onChange={(e) => updateNodeParam(selectedNode.id, field.key, e.target.value, field.type)}
                          >
                            {(field.options || []).map((opt) => (
                              <option key={opt} value={opt}>{opt}</option>
                            ))}
                          </select>
                        ) : (
                          <input
                            id={field.key}
                            className="form-input"
                            type={field.type === 'number' ? 'number' : 'text'}
                            step={field.type === 'number' ? 'any' : undefined}
                            value={selectedNode.data?.params?.[field.key] ?? ''}
                            onChange={(e) => updateNodeParam(selectedNode.id, field.key, e.target.value, field.type)}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          {activeTab === 'chat' && (
            <div className="chat-panel">
              <div className="chat-messages">
                {chatMessages.map((m) => (
                  <div key={m.id} className={"bubble " + m.role}>
                    {m.content}
                  </div>
                ))}
              </div>
              <div className="chat-input-row">
                <input
                  className="chat-input"
                  placeholder="Ask the builder assistant..."
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') onSendChat(); }}
                />
                <button className="builder-btn" onClick={onSendChat}>Send</button>
              </div>
            </div>
          )}
          {activeTab === 'code' && (
            <div className="code-panel">
              <div className="code-toolbar">
                <div className="code-status">{isGenerating ? 'Generating…' : (genError ? 'Error' : 'Up to date')}</div>
                <button className="builder-btn" onClick={copyCode}>Copy</button>
              </div>
              {genError && (
                <div className="code-error">{genError}</div>
              )}
              <pre className="code-block"><code>{generatedCode || '# Code will appear here as you build'}</code></pre>
            </div>
          )}
        </aside>
      </div>

      {hoverSchema && (
        <div className="hover-card" style={{ top: hoverCard.y, left: hoverCard.x }}>
          <div className="hover-title">{hoverSchema.title ?? hoverCard.type}</div>
          {hoverSchema.description && (
            <div className="hover-desc">{hoverSchema.description}</div>
          )}
          {(hoverSchema.fields?.length > 0) && (
            <div className="hover-params">
              <div className="hover-section-title">Parameters</div>
              {hoverSchema.fields.map((f) => (
                <div key={f.key} className="hover-param-row">
                  <div className="hover-param-name">{f.label}</div>
                  <div className="hover-param-meta">{f.type}{f.default !== undefined ? ` · default: ${f.default}` : ''}</div>
                  {f.help && <div className="hover-param-help">{f.help}</div>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
} 