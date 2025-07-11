import React, { useState, useEffect } from 'react';
import './App.css';
import ChatWindow from './components/ChatWindow';
import InputBar from './components/InputBar';
import MachineSidebar from './components/MachineSidebar';
import SettingsSidebar from './components/SettingsSidebar';
import SystemPromptModal from './components/SystemPromptModal';
import ChangeModelModal from './components/ChangeModelModal';

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

    const [machine, setMachine] = useState(() => localStorage.getItem('machine') || 'Yaskawa Alarm 380500');
    const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');
    const [language, setLanguage] = useState(() => localStorage.getItem('language') || 'en');
    const [fontSize, setFontSize] = useState(() => parseInt(localStorage.getItem('fontSize'), 10) || 16);
    const [promptTemplate, setPromptTemplate] = useState('');
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
                setPromptTemplate(data.prompt_template);
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
        const newMessages = [...messages, { text: query, sender: 'user' }];
        setMessages(newMessages);
        setLoading(true);

        // Add a thinking message
        const thinkingMessage = { text: t.thinking, sender: 'bot' };
        setMessages([...newMessages, thinkingMessage]);

        try {
            const res = await fetch('http://127.0.0.1:8000/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    machine, 
                    query, 
                    language, 
                    system_prompt: "You are a helpful assistant.",
                    model_config: modelConfig 
                }),
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let responseText = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                responseText += decoder.decode(value, { stream: true });
                setMessages([...newMessages, { text: responseText, sender: 'bot' }]);
            }

        } catch (error) {
            setMessages([...newMessages, { text: 'An error occurred while fetching the response.', sender: 'bot' }]);
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
    };

    const handlePromptTemplateSave = async () => {
        try {
            await fetch('http://127.0.0.1:8000/prompt_template', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt_template: promptTemplate }),
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
        </div>
    );
}

export default App;
