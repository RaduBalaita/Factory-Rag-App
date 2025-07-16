import React from 'react';

const SystemPromptModal = ({ isOpen, onClose, systemPrompt, setSystemPrompt, onSave }) => {
    if (!isOpen) return null;

    return (
        <div className="modal-overlay">
            <div className="modal">
                <h3>Edit System Prompt</h3>
                <textarea 
                    value={systemPrompt} 
                    onChange={(e) => setSystemPrompt(e.target.value)} 
                    rows="10"
                />
                <button onClick={onSave}>Save and Close</button>
            </div>
        </div>
    );
};

export default SystemPromptModal;