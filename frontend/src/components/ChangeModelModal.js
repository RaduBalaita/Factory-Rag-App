import React, { useState, useEffect } from 'react';

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
        availableModels: 'Available Models',
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
        availableModels: 'Modele Disponibile',
    }
};

const ChangeModelModal = ({ isOpen, onClose, language, modelConfig, setModelConfig }) => {
    const [activeTab, setActiveTab] = useState(modelConfig.type === 'local' ? 'local' : 'cloud');
    const [cloudProvider, setCloudProvider] = useState(modelConfig.provider || 'google');
    const [apiKeys, setApiKeys] = useState({
        google: modelConfig.provider === 'google' ? modelConfig.api_key || '' : '',
        openai: modelConfig.provider === 'openai' ? modelConfig.api_key || '' : '',
        claude: modelConfig.provider === 'claude' ? modelConfig.api_key || '' : ''
    });
    const [localModelPath, setLocalModelPath] = useState(modelConfig.type === 'local' ? modelConfig.path : '');
    const [availableLocalModels, setAvailableLocalModels] = useState([]);

    const t = translations[language];

    useEffect(() => {
        // Update internal state if the modal is reopened with new props
        if (isOpen) {
            setActiveTab(modelConfig.type === 'local' ? 'local' : 'cloud');
            setCloudProvider(modelConfig.provider || 'google');
            // Load the correct API key for the current provider
            const newApiKeys = { ...apiKeys };
            if (modelConfig.provider && modelConfig.api_key) {
                newApiKeys[modelConfig.provider] = modelConfig.api_key;
            }
            setApiKeys(newApiKeys);
            setLocalModelPath(modelConfig.type === 'local' ? modelConfig.path : '');
        }
    }, [isOpen, modelConfig]);

    useEffect(() => {
        const fetchLocalModels = async () => {
            try {
                const res = await fetch('http://127.0.0.1:8000/api/models');
                const data = await res.json();
                if (data.models) {
                    setAvailableLocalModels(data.models);
                }
            } catch (error) {
                console.error("Failed to fetch local models:", error);
            }
        };

        if (isOpen && activeTab === 'local') {
            fetchLocalModels();
        }
    }, [isOpen, activeTab]);

    const handleFileChange = (event) => {
        if (event.target.files && event.target.files[0]) {
            // Note: Accessing the full path of a file selected by a user is a security restriction in modern browsers.
            // We can only get the file name. The backend will need to know the base path to the models.
            setLocalModelPath(event.target.files[0].name);
        }
    };
    
    const handleSave = () => {
        if (activeTab === 'cloud') {
            setModelConfig({ 
                type: 'api', 
                provider: cloudProvider, 
                api_key: apiKeys[cloudProvider] 
            });
        } else {
            // Prepend the base path for the backend
            const fullPath = `C:/Users/Tempest/Desktop/RAG/.models/${localModelPath}`;
            setModelConfig({ type: 'local', path: fullPath });
        }
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
                        <div className="provider-radios">
                            <label>
                                <input type="radio" value="google" checked={cloudProvider === 'google'} onChange={() => setCloudProvider('google')} />
                                Google
                            </label>
                            <label>
                                <input type="radio" value="openai" checked={cloudProvider === 'openai'} onChange={() => setCloudProvider('openai')} />
                                OpenAI
                            </label>
                            <label>
                                <input type="radio" value="claude" checked={cloudProvider === 'claude'} onChange={() => setCloudProvider('claude')} />
                                Claude
                            </label>
                        </div>
                        <div className="api-key-input">
                            <label>{t[`${cloudProvider}APIKey`]}</label>
                            <input 
                                type="password" 
                                value={apiKeys[cloudProvider]} 
                                onChange={(e) => setApiKeys({...apiKeys, [cloudProvider]: e.target.value})} 
                            />
                        </div>
                    </div>
                )}

                {activeTab === 'local' && (
                    <div className="tab-content">
                        <p>{t.currentLocalModel}: {localModelPath.split('/').pop() || t.noModelSelected}</p>
                        <input type="file" id="local-model-input" style={{ display: 'none' }} onChange={handleFileChange} accept=".gguf" />
                        <button onClick={() => document.getElementById('local-model-input').click()}>{t.browse}</button>
                        
                        <h4>{t.availableModels}</h4>
                        <div className="local-model-list">
                            {availableLocalModels.map(model => (
                                <div 
                                    key={model} 
                                    className={`local-model-item ${localModelPath.endsWith(model) ? 'selected' : ''}`}
                                    onClick={() => setLocalModelPath(model)}
                                >
                                    {model}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                <button onClick={handleSave}>{t.save}</button>
            </div>
        </div>
    );
};

export default ChangeModelModal;
