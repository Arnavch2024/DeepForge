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
  // State management
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [activeTab, setActiveTab] = useState('inspector');
  const [generatedCode, setGeneratedCode] = useState('');
  const [codeLanguage, setCodeLanguage] = useState('python');
  const [showRawCode, setShowRawCode] = useState(false);
  const [validation, setValidation] = useState({ errors: [], warnings: [] });
  const [isValidating, setIsValidating] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [hoverCard, setHoverCard] = useState({ visible: false, x: 0, y: 0, type: null });
  const [genError, setGenError] = useState(null);
  
  const reactFlowWrapper = useRef(null);
  const reactFlowInstance = useRef(null);

  // Load and save state
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

  // Node and edge handlers
  const onConnect = useCallback((params) => {
    setEdges((eds) => addEdge(params, eds));
  }, [setEdges]);

  const onNodeClick = useCallback((event, node) => {
    setSelectedNode(node);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  // Update node parameters
  const updateNodeParam = useCallback((nodeId, key, value, cast) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === nodeId) {
          const params = { ...node.data.params, [key]: cast === 'number' ? parseFloat(value) : value };
          return { ...node, data: { ...node.data, params } };
        }
        return node;
      })
    );
  }, [setNodes]);

  // Chat functionality
  const onSendChat = useCallback(() => {
    const trimmed = chatInput.trim();
    if (!trimmed) return;
    const userMsg = { role: 'user', content: trimmed, id: uuidv4() };
    setChatMessages(prev => [...prev, userMsg]);
    setChatInput('');
    // Here you would typically send the message to a chat API
  }, [chatInput]);

  // Generate code
  const generateCode = useCallback(() => {
    try {
      // This would typically call an API or generate code based on the current graph
      const code = "# Generated code will appear here";
      setGeneratedCode(code);
      setGenError(null);
      setActiveTab('code');
    } catch (error) {
      console.error('Error generating code:', error);
      setGenError(error.message);
    }
  }, [nodes, edges]);

  // Validate graph
  const validateGraph = useCallback(() => {
    setIsValidating(true);
    // Simulate validation
    setTimeout(() => {
      const errors = [];
      const warnings = [];
      
      // Example validation
      if (nodes.length === 0) {
        warnings.push('The graph is empty');
      }
      
      setValidation({ errors, warnings });
      setIsValidating(false);
    }, 500);
  }, [nodes, edges]);

  // Effect to save changes to localStorage
  useEffect(() => {
    saveToStorage(nodes, edges);
  }, [nodes, edges, saveToStorage]);

  // Initial load
  useEffect(() => {
    const { nodes: savedNodes, edges: savedEdges } = loadFromStorage();
    setNodes(savedNodes);
    setEdges(savedEdges);
  }, [loadFromStorage, setNodes, setEdges]);

  // Render the code panel with validation sidebar
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
      
      {/* Validation Panel - Only shown when there are errors or warnings */}
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

  // Main render
  return (
    <div className="builder-root">
      <header className="builder-header">
        <h1 className="builder-title">{title}</h1>
        <div className="builder-actions">
          <button 
            className="builder-btn"
            onClick={validateGraph}
            disabled={isValidating}
          >
            {isValidating ? 'Validating...' : 'Validate'}
          </button>
          <button 
            className="builder-btn"
            onClick={generateCode}
          >
            Generate Code
          </button>
        </div>
      </header>

      <div className="builder-container three-cols">
        {/* Left Sidebar - Node Palette */}
        <aside className="builder-sidebar">
          <h2 className="palette-title">Nodes</h2>
          {Object.entries(palette).map(([category, items]) => (
            <div key={category} className="mb-6">
              <h3 className="text-sm font-semibold text-gray-400 mb-2 uppercase tracking-wider">
                {category}
              </h3>
              <div className="space-y-2">
                {items.map((item) => (
                  <div 
                    key={item.type}
                    className="node-palette-item card"
                    style={{ '--primary-color': item.color, '--primary-hover': item.hoverColor }}
                    onDragStart={(event) => {
                      event.dataTransfer.setData('application/reactflow', JSON.stringify(item));
                      event.dataTransfer.effectAllowed = 'move';
                    }}
                    draggable
                  >
                    <div className="node-title">{item.label}</div>
                    <div className="node-sub">{item.type}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </aside>

        {/* Main Canvas */}
        <div className="builder-canvas" ref={reactFlowWrapper}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            nodeTypes={schemas}
            fitView
            proOptions={{ hideAttribution: true }}
            ref={reactFlowInstance}
          >
            <Background />
            <Controls />
          </ReactFlow>
        </div>

        {/* Right Sidebar - Inspector/Code/Chat */}
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
                                  updateNodeParam(selectedNode.id, field.key, e.target.value, field.type)
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
                <div className="chat-messages flex-1 overflow-auto p-4">
                  {chatMessages.map((m) => (
                    <div 
                      key={m.id} 
                      className={`bubble ${m.role} ${m.role === 'user' ? 'ml-auto' : 'mr-auto'}`}
                    >
                      {m.content}
                    </div>
                  ))}
                </div>
                <div className="chat-input-row p-4 border-t border-gray-700 flex">
                  <input
                    className="chat-input flex-1 bg-gray-800 text-white rounded-l-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Ask the builder assistant..."
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') onSendChat();
                    }}
                  />
                  <button
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-r-lg transition-colors"
                    onClick={onSendChat}
                  >
                    Send
                  </button>
                </div>
              </div>
            )}
            
            {activeTab === 'code' && renderCodePanel()}
          </div>
        </aside>
      </div>

      {/* Hover Card */}
      {hoverCard.visible && hoverCard.type && (
        <div 
          className="hover-card" 
          style={{ 
            position: 'fixed',
            top: hoverCard.y,
            left: hoverCard.x,
            zIndex: 1000,
            background: '#111827',
            border: '1px solid #1f2937',
            borderRadius: '8px',
            padding: '12px',
            maxWidth: '300px',
            boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1)',
            pointerEvents: 'none'
          }}
        >
          <div className="hover-title font-medium text-white mb-1">
            {hoverCard.type}
          </div>
          {schemas?.[hoverCard.type]?.description && (
            <div className="hover-desc text-sm text-gray-300 mb-2">
              {schemas[hoverCard.type].description}
            </div>
          )}
          {schemas?.[hoverCard.type]?.fields && (
            <div className="hover-params">
              <div className="hover-section-title text-xs font-semibold text-gray-400 mb-1">
                Parameters
              </div>
              <div className="space-y-1">
                {schemas[hoverCard.type].fields.map((f) => (
                  <div key={f.key} className="hover-param-row text-xs text-gray-300">
                    <div className="hover-param-name font-medium">{f.label}</div>
                    <div className="hover-param-meta text-gray-400">
                      {f.type}
                      {f.default !== undefined ? ` · default: ${f.default}` : ''}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
