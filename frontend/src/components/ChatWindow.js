import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const ChatWindow = ({ messages }) => {
    return (
        <div className="chat-window">
            {messages.map((msg, index) => (
                <div key={index} className={`message ${msg.sender}`}>
                    {msg.sender === 'bot' ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                    ) : (
                        <p>{msg.text}</p>
                    )}
                </div>
            ))}
        </div>
    );
};

export default ChatWindow;