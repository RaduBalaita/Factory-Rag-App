import React, { useState } from 'react';

const translations = {
    en: {
        describeProblem: 'Describe your problem...',
        clear: 'Clear',
        send: 'Send',
    },
    ro: {
        describeProblem: 'Descrie problema ta...',
        clear: 'Curăță',
        send: 'Trimite',
    },
};

const InputBar = ({ onSendMessage, onClearChat, language }) => {
    const [query, setQuery] = useState('');

    const t = translations[language];

    const handleSubmit = () => {
        if (query.trim()) {
            onSendMessage(query);
            setQuery('');
        }
    };

    return (
        <div className="input-bar">
            <button className="clear-btn" onClick={onClearChat}>{t.clear}</button>
            <input 
                type="text" 
                placeholder={t.describeProblem} 
                value={query} 
                onChange={(e) => setQuery(e.target.value)} 
                onKeyPress={(e) => e.key === 'Enter' && handleSubmit()} 
            />
            <button className="send-btn" onClick={handleSubmit}>{t.send}</button>
        </div>
    );
};

export default InputBar;