import React, { useState } from 'react';
import { Highlight, themes } from 'prism-react-renderer';
import { FaCopy, FaCheck } from 'react-icons/fa';

const CodeDisplay = ({ code, language = 'python', className = '' }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  if (!code) {
    return (
      <div className={`bg-gray-900 text-gray-400 p-4 rounded-md ${className}`}>
        <div className="text-center py-8">
          <p>Your generated code will appear here</p>
          <p className="text-sm text-gray-500 mt-2">
            Add nodes and connect them to see the generated code
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative bg-gray-900 rounded-md overflow-hidden code-display-container ${className}`}>
      <div className="flex justify-between items-center bg-gray-800 px-4 py-2 border-b border-gray-700">
        <div className="text-sm text-gray-300">
          {language.toUpperCase()}
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center space-x-1 text-sm text-gray-300 hover:text-white transition-colors"
          title="Copy to clipboard"
        >
          {copied ? (
            <>
              <FaCheck className="text-green-400" />
              <span>Copied!</span>
            </>
          ) : (
            <>
              <FaCopy />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <div className="overflow-auto max-h-[500px] bg-gray-900 code-display-container">
        <Highlight
          theme={{
            plain: {
              color: '#e2e8f0',
              backgroundColor: '#111827'
            },
            styles: [
              {
                types: ['comment', 'prolog', 'doctype', 'cdata'],
                style: {
                  color: '#6b7280'
                }
              },
              {
                types: ['punctuation'],
                style: {
                  color: '#9ca3af'
                }
              },
              {
                types: ['property', 'tag', 'boolean', 'number', 'constant', 'symbol', 'deleted'],
                style: {
                  color: '#f59e0b'
                }
              },
              {
                types: ['selector', 'attr-name', 'string', 'char', 'builtin', 'inserted'],
                style: {
                  color: '#10b981'
                }
              },
              {
                types: ['operator', 'entity', 'url', 'variable'],
                style: {
                  color: '#3b82f6'
                }
              },
              {
                types: ['keyword'],
                style: {
                  color: '#8b5cf6'
                }
              },
              {
                types: ['function', 'class-name'],
                style: {
                  color: '#f59e0b'
                }
              }
            ]
          }}
          code={code}
          language={language}
        >
          {({ className, style, tokens, getLineProps, getTokenProps }) => (
            <pre 
              className={`${className} bg-gray-900 prism-code`}
              style={{
                margin: 0,
                padding: '1rem',
                backgroundColor: '#111827 !important',
                fontSize: '0.875rem',
                lineHeight: '1.5',
                fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace',
                color: '#e2e8f0'
              }}
            >
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line })} className="flex bg-gray-900">
                  <span className="text-gray-500 select-none mr-4 w-8 text-right text-xs">
                    {i + 1}
                  </span>
                  <span className="flex-1">
                    {line.map((token, key) => (
                      <span key={key} {...getTokenProps({ token })} />
                    ))}
                  </span>
                </div>
              ))}
            </pre>
          )}
        </Highlight>
      </div>
    </div>
  );
};

export default CodeDisplay;
