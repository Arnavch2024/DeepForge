import React, { useCallback, useMemo, useRef, useState, useEffect } from 'react';
import ReactFlow, {
  Background,
  Controls,
  addEdge,
  useEdgesState,
  useNodesState,
  getIncomers,
  getOutgoers
} from 'reactflow';
import 'reactflow/dist/style.css';
import { v4 as uuidv4 } from 'uuid';
import CodeDisplay from './CodeDisplay';
import { FiAlertTriangle, FiCheckCircle, FiInfo } from 'react-icons/fi';
import { TbCopy, TbCopyCheck } from 'react-icons/tb';
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

export default function BaseBuilder({ title, palette, storageKey, schemas, builderType, presets }) {
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

  // Code generation and validation states
  const [generatedCode, setGeneratedCode] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [genError, setGenError] = useState(null);
  const [validation, setValidation] = useState({ errors: [], warnings: [] });
  const [isValidating, setIsValidating] = useState(false);
  const [codeLanguage, setCodeLanguage] = useState('python');
  const [showRawCode, setShowRawCode] = useState(false);

  // Theme and presets
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('builder-theme');
    return saved || 'dark';
  });

  // Debounced API calls for live code generation
  useEffect(() => {
    if (!builderType || (nodes.length === 0 && edges.length === 0)) {
      setGeneratedCode('');
      setValidation({ errors: [], warnings: [] });
      return;
    }

    const timeoutId = setTimeout(async () => {
      try {
        setIsGenerating(true);
        setIsValidating(true);
        setGenError(null);

        // First validate the graph
        const validationResponse = await fetch('http://localhost:8000/validate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            builder_type: builderType,
            nodes,
            edges
          })
        });

        const validationData = await validationResponse.json();
        setValidation(validationData);

        if (validationData.errors && validationData.errors.length > 0) {
          throw new Error('Validation failed');
        }

        // If validation passes, generate code
        const response = await fetch('http://localhost:8000/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            graph: { nodes, edges },
            builder_type: builderType
          })
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        setGeneratedCode(data.code || '');
        setCodeLanguage(data.language || 'python');
      } catch (error) {
        console.error('Code generation error:', error);
        setGenError(error.message);
        setGeneratedCode('');
        setValidation({ errors: [], warnings: [] });
      } finally {
        setIsGenerating(false);
        setIsValidating(false);
      }
    }, 500); // 500ms debounce

    return () => clearTimeout(timeoutId);
  }, [nodes, edges, builderType]);

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
    const position = reactFlowInstance.screenToFlowPosition({
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
    // keep x fixed, follow Y for better feel
    setHoverCard((hc) => hc.visible ? { ...hc, y: e.clientY + window.scrollY - 20 } : hc);
  }, []);

  const hideHover = useCallback(() => {
    setHoverCard({ visible: false, x: 0, y: 0, type: null });
  }, []);

  // New functions for enhanced features
  const copyCode = useCallback(() => {
    if (generatedCode) {
      navigator.clipboard.writeText(generatedCode);
      // Could add toast notification here
    }
  }, [generatedCode]);

  const toggleTheme = useCallback(() => {
    const newTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    localStorage.setItem('builder-theme', newTheme);
  }, [theme]);

  const applyPresetById = useCallback((presetId) => {
    const preset = presets?.find(p => p.id === presetId);
    if (!preset?.build) return;
    
    try {
      const { nodes: presetNodes, edges: presetEdges } = preset.build();
      setNodes(presetNodes);
      setEdges(presetEdges);
      saveToStorage(presetNodes, presetEdges);
    } catch (error) {
      console.error('Failed to apply preset:', error);
    }
  }, [presets, setNodes, setEdges, saveToStorage]);

  const onSelectionChange = useCallback(({ nodes: selectedNodes }) => {
    setSelectedNodeId(selectedNodes?.[0]?.id ?? null);
  }, []);

  const hoverSchema = hoverCard.visible && hoverCard.type ? schemas?.[hoverCard.type] : null;

  const renderCodePanel = () => (
    <div className="flex h-full overflow-hidden">
      {/* Code Panel */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex justify-between items-center p-4 border-b border-gray-700">
          <h3 className="text-lg font-medium">Generated Code</h3>
          <div className="flex space-x-2">
            <button
              onClick={() => setShowRawCode(!showRawCode)}
              className="text-xs px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-md transition-colors"
            >
              {showRawCode ? 'Show Formatted' : 'Show Raw'}
            </button>
          </div>
        </div>
        
        <div className="flex-1 overflow-auto">
          {showRawCode ? (
            <div className="p-4">
              <pre className="bg-gray-900 p-4 rounded-md overflow-x-auto">
                <code className="text-sm">
                  {generatedCode || '// Your generated code will appear here'}
                </code>
              </pre>
            </div>
          ) : (
            <CodeDisplay 
              code={generatedCode} 
              language={codeLanguage}
              className="h-full"
            />
          )}
        </div>
        
        {genError && (
          <div className="p-4 bg-red-900/30 border-t border-red-800 text-red-300 text-sm">
            <div className="font-medium mb-1">Code Generation Error</div>
            <div>{genError}</div>
          </div>
        )}
      </div>
      
      {/* Validation Panel */}
      {(validation.errors.length > 0 || validation.warnings.length > 0) && (
        <div className="w-80 border-l border-gray-700 flex flex-col">
          <div className="p-4 border-b border-gray-700">
            <h3 className="font-medium">Validation</h3>
          </div>
          <div className="p-4 overflow-auto">
            {validation.errors.length > 0 && (
              <div className="mb-4">
                <h4 className="text-red-400 font-medium mb-2 flex items-center">
                  <FiAlertTriangle className="mr-2" />
                  Validation Errors
                </h4>
                <ul className="text-sm text-gray-300 space-y-1">
                  {validation.errors.map((err, i) => (
                    <li key={i} className="flex items-start">
                      <span className="text-red-400 mr-2">•</span>
                      <span>{err}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {validation.warnings.length > 0 && (
              <div>
                <h4 className="text-yellow-400 font-medium mb-2 flex items-center">
                  <FiInfo className="mr-2" />
                  Warnings
                </h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  {validation.warnings.map((warn, i) => (
                    <li key={i} className="flex items-start">
                      <span className="text-yellow-400 mr-2">•</span>
                      <span>{warn}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className={`builder-root ${theme === 'light' ? 'theme-light' : ''}`}>
      <div className="builder-header">
        <div className="builder-title">{title}</div>
        <div className="builder-actions">
          {presets && presets.length > 0 && (
            <select 
              className="builder-select" 
              onChange={(e) => e.target.value && applyPresetById(e.target.value)}
              value=""
            >
              <option value="">Load Preset...</option>
              {presets.map(preset => (
                <option key={preset.id} value={preset.id}>{preset.name}</option>
              ))}
            </select>
          )}
          <button className="builder-btn" onClick={toggleTheme}>
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
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
            onSelectionChange={onSelectionChange}
          >
            <Background />
            <Controls />
          </ReactFlow>
        </div>
        <aside className="builder-right">
          <div className="panel-tabs">
            <button 
              className={`tab ${activeTab === 'inspector' ? 'active' : ''}`}
              onClick={() => setActiveTab('inspector')}
            >
              Inspector
            </button>
            <button 
              className={`tab ${activeTab === 'code' ? 'active' : ''}`}
              onClick={() => setActiveTab('code')}
            >
              Code
            </button>
            <button 
              className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              Chat
            </button>
          </div>
          <div className="flex-1 overflow-hidden flex flex-col">
            {activeTab === 'inspector' && (
              <div className="flex-1 overflow-auto">
                <div className="inspector">
                  {!selectedNode ? (
                    <div className="empty">Select a node to edit its parameters</div>
                  ) : (
                    <div>
                      <div className="inspector-title">{selectedNode.data?.label}</div>
                      <div className="inspector-sub">{selectedNode.data?.type}</div>
                      <div className="inspector-form">
                        {(schemas?.[selectedNode.data?.type]?.fields || []).map((field) => (
                          <div className="form-group" key={field.key}>
                            <label className="form-label" htmlFor={field.key}>
                              {field.label}
                            </label>
                            {field.type === 'select' ? (
                              <select
                                id={field.key}
                                className="form-input"
                                value={selectedNode.data?.params?.[field.key] ?? ''}
                                onChange={(e) =>
                                  updateNodeParam(selectedNode.id, field.key, e.target.value, field.type)
                                }
                              >
                                {(field.options || []).map((opt) => (
                                  <option key={opt} value={opt}>
                                    {opt}
                                  </option>
                                ))}
                              </select>
                            ) : (
                              <input
                                id={field.key}
                                className="form-input"
                                type={field.type === 'number' ? 'number' : 'text'}
                                step={field.type === 'number' ? 'any' : undefined}
                                value={selectedNode.data?.params?.[field.key] ?? ''}
                                onChange={(e) =>
                                  updateNodeParam(
                                    selectedNode.id,
                                    field.key,
                                    e.target.value,
                                    field.type
                                  )
                                }
                              />
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'chat' && (
              <div className="flex-1 flex flex-col">
                <div className="chat-messages flex-1 overflow-auto">
                  {chatMessages.map((msg) => (
                    <div key={msg.id} className={`message ${msg.role}`}>
                      {msg.content}
                    </div>
                  ))}
                </div>
                <div className="chat-input-container">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && onSendChat()}
                    placeholder="Type a message..."
                  />
                  <button onClick={onSendChat}>Send</button>
                </div>
              </div>
            )}

            {activeTab === 'code' && (
              <div className="code-panel">
                <div className="code-toolbar">
                  <button onClick={() => setShowRawCode(!showRawCode)}>
                    {showRawCode ? 'Show Formatted' : 'Show Raw'}
                  </button>
                </div>
                <div className="code-content">
                  {showRawCode ? (
                    <pre>{generatedCode}</pre>
                  ) : (
                    <CodeDisplay
                      code={generatedCode}
                      language={codeLanguage}
                      onLanguageChange={setCodeLanguage}
                    />
                  )}
                </div>
              </div>
            )}
          </div>
        </aside>
      </div>
      
      {hoverSchema && (
        <div className="hover-card" style={{ top: hoverCard.y, left: hoverCard.x }}>
          <div className="hover-title">{hoverSchema.title ?? hoverCard.type}</div>
          {hoverSchema.description && (
            <div className="hover-desc">{hoverSchema.description}</div>
          )}
          {hoverSchema.fields?.length > 0 && (
            <div className="hover-params">
              <div className="hover-section-title">Parameters</div>
              {hoverSchema.fields.map((f) => (
                <div key={f.key} className="hover-param-row">
                  <div className="hover-param-name">{f.label}</div>
                  <div className="hover-param-meta">
                    {f.type}
                    {f.default !== undefined ? ` · default: ${f.default}` : ''}
                  </div>
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