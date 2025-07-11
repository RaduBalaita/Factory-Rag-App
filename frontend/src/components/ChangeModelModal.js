import React, { useState } from 'react';

const translations = {
    en: {
        changeModel: 'Change Model',
        cloudAPIs: 'Cloud APIs',
        localModels: 'Local Models',
        googleAPIKey: 'Google API Key',
        openaiAPIKey: 'OpenAI API Key',
        claudeAPIKey: 'Claude API Key',
        save: 'Save',
        browse: 'Browse',
        currentLocalModel: 'Current Local Model',
        noModelSelected: 'No model selected',
    },
    ro: {
        changeModel: 'Schimbă Modelul',
        cloudAPIs: 'API-uri Cloud',
        localModels: 'Modele Locale',
        googleAPIKey: 'Cheie API Google',
        openaiAPIKey: 'Cheie API OpenAI',
        claudeAPIKey: 'Cheie API Claude',
        save: 'Salvează',
        browse: 'Răsfoiește',
        currentLocalModel: 'Model Local Actual',
        noModelSelected: 'Niciun model selectat',
    }
};

const ChangeModelModal = ({ isOpen, onClose, language, modelConfig, setModelConfig }) => {
    const [activeTab, setActiveTab] = useState('cloud');
    const [apiKeys, setApiKeys] = useState({
        google: '',
        openai: '',
        claude: '',
    });
    const [localModelPath, setLocalModelPath] = useState('');

    const t = translations[language];

    const handleFileChange = (event) => {
        if (event.target.files && event.target.files[0]) {
            setLocalModelPath(event.target.files[0].path);
        }
    };
    
    const handleSave = () => {
        // Logic to save the model configuration will be added here
        onClose();
    };

    if (!isOpen) return null;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal" onClick={(e) => e.stopPropagation()}>
                <h3>{t.changeModel}</h3>
                <div className="tab-buttons">
                    <button onClick={() => setActiveTab('cloud')} className={activeTab === 'cloud' ? 'active' : ''}>{t.cloudAPIs}</button>
                    <button onClick={() => setActiveTab('local')} className={activeTab === 'local' ? 'active' : ''}>{t.localModels}</button>
                </div>

                {activeTab === 'cloud' && (
                    <div className="tab-content">
                        <div className="api-key-input">
                            <label>{t.googleAPIKey}</label>
                            <input type="password" value={apiKeys.google} onChange={(e) => setApiKeys({...apiKeys, google: e.target.value })} />
                        </div>
                        <div className="api-key-input">
                            <label>{t.openaiAPIKey}</label>
                            <input type="password" value={apiKeys.openai} onChange={(e) => setApiKeys({...apiKeys, openai: e.target.value })} />
                        </div>
                        <div className="api-key-input">
                            <label>{t.claudeAPIKey}</label>
                            <input type="password" value={apiKeys.claude} onChange={(e) => setApiKeys({...apiKeys, claude: e.target.value })} />
                        </div>
                    </div>
                )}

                {activeTab === 'local' && (
                    <div className="tab-content">
                        <p>{t.currentLocalModel}: {localModelPath || t.noModelSelected}</p>
                        <input type="file" id="local-model-input" style={{ display: 'none' }} onChange={handleFileChange} accept=".gguf" />
                        <button onClick={() => document.getElementById('local-model-input').click()}>{t.browse}</button>
                    </div>
                )}

                <button onClick={handleSave}>{t.save}</button>
            </div>
        </div>
    );
};

export default ChangeModelModal;
