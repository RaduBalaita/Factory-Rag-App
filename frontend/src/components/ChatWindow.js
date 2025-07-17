import React, { useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const ChatWindow = ({ messages }) => {
    // Create a ref for the very last message element.
    const lastMessageRef = useRef(null);

    useEffect(() => {
        // If the last message ref is attached to an element, scroll it into view smoothly.
        if (lastMessageRef.current) {
            lastMessageRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages]);

    return (
        <div className="chat-window">
            {messages.map((msg, index) => {
                // Check if this is the last message in the array.
                const isLastMessage = index === messages.length - 1;
                return (
                    // Attach the ref *only* to the div of the last message.
                    <div 
                        key={index} 
                        className={`message ${msg.sender}`} 
                        ref={isLastMessage ? lastMessageRef : null}
                    >
                        {msg.sender === 'bot' ? (
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                        ) : (
                            <p>{msg.text}</p>
                        )}
                    </div>
                );
            })}
        </div>
    );
};

export  default ChatWindow;