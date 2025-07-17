import React, { useState, useEffect } from 'react';
import './App.css';
import ChatWindow from './components/ChatWindow';
import InputBar from './components/InputBar';
import MachineSidebar from './components/MachineSidebar';
import SettingsSidebar from './components/SettingsSidebar';
import SystemPromptModal from './components/SystemPromptModal';
import ChangeModelModal from './components/ChangeModelModal';
import ManageDocumentsModal from './components/ManageDocumentsModal';

const translations = {
    en: {
        thinking: 'Thinking...',
    },
    ro: {
        thinking: 'Gândire...',
    },
};

function App() {
    const [isMachineSidebarOpen, setMachineSidebarOpen] = useState(false);
    const [isSettingsSidebarOpen, setSettingsSidebarOpen] = useState(false);
    const [isSystemPromptModalOpen, setSystemPromptModalOpen] = useState(false);
    const [isChangeModelModalOpen, setChangeModelModalOpen] = useState(false);
    const [isManageDocumentsModalOpen, setManageDocumentsModalOpen] = useState(false);

    const [machine, setMachine] = useState(() => localStorage.getItem('machine') || 'No machine selected');
    const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');
    const [language, setLanguage] = useState(() => localStorage.getItem('language') || 'en');
    const [fontSize, setFontSize] = useState(() => parseInt(localStorage.getItem('fontSize'), 10) || 16);
    const [promptTemplate, setPromptTemplate] = useState(`You will be given context from a machine's technical manual and an error code.
Structure your response in three distinct sections:
1. **Problem Description:** Briefly explain what the error code means based on the context.
2. **Probable Causes:** List the most likely reasons for this error from the context.
3. **Solution Steps:** Provide a clear, step-by-step guide to fix the issue using only information from the context.

**IMPORTANT:** Only use information from the provided context. If the context is empty or does not contain specific information for the queried error code, state that "No specific information was found for this error code in the manual."`);
    const [modelConfig, setModelConfig] = useState(() => {
        const savedConfig = localStorage.getItem('modelConfig');
        return savedConfig ? JSON.parse(savedConfig) : { type: 'api', provider: 'google' };
    });
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);

    const t = translations[language];


    useEffect(() => {
        const fetchPromptTemplate = async () => {
            try {
                const res = await fetch('http://127.0.0.1:8000/prompt_template');
                const data = await res.json();
                setPromptTemplate(data.template);
            } catch (error) {
                console.error("Failed to fetch prompt template:", error);
                // Set a fallback template if the fetch fails
                setPromptTemplate('You are a helpful assistant.');
            }
        };
        fetchPromptTemplate();
    }, []);

    useEffect(() => {
        localStorage.setItem('machine', machine);
        localStorage.setItem('theme', theme);
        localStorage.setItem('language', language);
        localStorage.setItem('fontSize', fontSize);
        localStorage.setItem('promptTemplate', promptTemplate);
        localStorage.setItem('modelConfig', JSON.stringify(modelConfig));
    }, [machine, theme, language, fontSize, promptTemplate, modelConfig]);

    const handleSendMessage = async (query) => {
    if (machine === 'No machine selected') {
        setMessages(prev => [...prev, 
            { text: query, sender: 'user' },
            { text: 'Please select a machine from the sidebar first.', sender: 'bot' }
        ]);
        return;
    }

    const userMessage = { text: query, sender: 'user' };
    
    // Step 1: Add the user message and the "Thinking..." placeholder.
    const thinkingPlaceholder = { text: t.thinking, sender: 'bot' };
    // Create an index to track which message we need to update.
    const botMessageIndex = messages.length + 1; 
    setMessages(prev => [...prev, userMessage, thinkingPlaceholder]);
    
    setLoading(true);

    try {
        const res = await fetch('http://127.0.0.1:8000/query', { // Corrected IP
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                machine, query, language, 
                system_prompt: promptTemplate,
                model_settings: modelConfig 
            }),
        });
        
        if (!res.ok) {
            const errorData = await res.json();
            throw new Error(errorData.detail || 'The server returned an error.');
        }

        // Step 2: Read the ENTIRE response from the stream at once.
        // Since the backend sends it all in one go now, this is efficient.
        const fullResponseText = await res.text();

        // Step 3: Replace the "Thinking..." message with the final, complete response.
        setMessages(prev => {
            const newMessages = [...prev];
            // Find the message at our specific index and update its text.
            newMessages[botMessageIndex] = { ...newMessages[botMessageIndex], text: fullResponseText };
            return newMessages;
        });

    } catch (error) {
        // If an error occurs, replace the "Thinking..." message with the error.
        setMessages(prev => {
            const newMessages = [...prev];
            newMessages[botMessageIndex] = { ...newMessages[botMessageIndex], text: `An error occurred: ${error.message}` };
            return newMessages;
        });
    } finally {
        setLoading(false);
    }
};

    const handleClearChat = () => {
        setMessages([]);
    };

    const handleContentClick = () => {
        if (isMachineSidebarOpen) setMachineSidebarOpen(false);
        if (isSettingsSidebarOpen) setSettingsSidebarOpen(false);
        if (isSystemPromptModalOpen) setSystemPromptModalOpen(false);
        if (isChangeModelModalOpen) setChangeModelModalOpen(false);
        if (isManageDocumentsModalOpen) setManageDocumentsModalOpen(false);
    };

    const handlePromptTemplateSave = async () => {
        try {
            await fetch('http://127.0.0.1:8000/prompt_template', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ template: promptTemplate }),
            });
        } catch (error) {
            console.error('Failed to save prompt template:', error);
        }
        setSystemPromptModalOpen(false);
    };

    return (
        <div className={`App ${theme} ${isMachineSidebarOpen ? 'machine-sidebar-open' : ''} ${isSettingsSidebarOpen ? 'settings-sidebar-open' : ''}`} style={{ fontSize: `${fontSize}px` }}>
            <MachineSidebar isOpen={isMachineSidebarOpen} onClose={() => setMachineSidebarOpen(false)} setMachine={setMachine} />
            <SettingsSidebar 
                isOpen={isSettingsSidebarOpen} 
                onClose={() => setSettingsSidebarOpen(false)} 
                theme={theme} 
                setTheme={setTheme} 
                language={language} 
                setLanguage={setLanguage} 
                fontSize={fontSize}
                setFontSize={setFontSize}
                openSystemPrompt={() => setSystemPromptModalOpen(true)} 
                openChangeModel={() => setChangeModelModalOpen(true)}
                openManageDocuments={() => setManageDocumentsModalOpen(true)}
                modelConfig={modelConfig}
            />
            <div className="main-content" onClick={handleContentClick}>
                <div className="top-bar">
                    <button onClick={(e) => { e.stopPropagation(); setMachineSidebarOpen(true); }}>&#9776;</button>
                    <span className="machine-name">{machine}</span>
                    <button onClick={(e) => { e.stopPropagation(); setSettingsSidebarOpen(true); }}>&#9881;</button>
                </div>
                <ChatWindow messages={messages} />
                <InputBar onSendMessage={handleSendMessage} onClearChat={handleClearChat} language={language} />
            </div>
            <SystemPromptModal 
                isOpen={isSystemPromptModalOpen} 
                onClose={() => setSystemPromptModalOpen(false)} 
                systemPrompt={promptTemplate} 
                setSystemPrompt={setPromptTemplate} 
                onSave={handlePromptTemplateSave}
            />
            <ChangeModelModal
                isOpen={isChangeModelModalOpen}
                onClose={() => setChangeModelModalOpen(false)}
                language={language}
                modelConfig={modelConfig}
                setModelConfig={setModelConfig}
            />
            <ManageDocumentsModal
                isOpen={isManageDocumentsModalOpen}
                onClose={() => setManageDocumentsModalOpen(false)}
                language={language}
                onDocumentChange={() => {
                    // Trigger a refresh of the machine list in MachineSidebar
                    // This could be improved with a more sophisticated state management
                }}
            />
        </div>
    );
}

export default App;
