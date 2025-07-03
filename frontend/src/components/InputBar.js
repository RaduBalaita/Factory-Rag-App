import React, { useState } from 'react';

const InputBar = ({ onSendMessage }) => {
    const [query, setQuery] = useState('');

    const handleSubmit = () => {
        if (query.trim()) {
            onSendMessage(query);
            setQuery('');
        }
    };

    return (
        <div className="input-bar">
            <input 
                type="text" 
                placeholder="Describe your problem..." 
                value={query} 
                onChange={(e) => setQuery(e.target.value)} 
                onKeyPress={(e) => e.key === 'Enter' && handleSubmit()} 
            />
            <button onClick={handleSubmit}>Send</button>
        </div>
    );
};

export default InputBar;