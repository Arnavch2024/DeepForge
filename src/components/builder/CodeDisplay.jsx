import React, { useState, useEffect } from 'react';
import { Prism as SyntaxHighlighter } from 'prism-react-renderer';
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
    <div className={`relative bg-gray-900 rounded-md overflow-hidden ${className}`}>
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
      <div className="overflow-auto max-h-[500px] p-4">
        <pre className="text-sm leading-relaxed">
          <code>
            {code.split('\n').map((line, i) => (
              <div key={i} className="flex">
                <span className="text-gray-500 select-none mr-4 w-8 text-right">
                  {i + 1}
                </span>
                <span className="flex-1">
                  {line || ' '}
                </span>
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
};

export default CodeDisplay;
