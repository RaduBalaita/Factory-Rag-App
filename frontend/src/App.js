import React, { useState } from 'react';
import './App.css';
import ChatWindow from './components/ChatWindow';
import InputBar from './components/InputBar';
import MachineSidebar from './components/MachineSidebar';
import SettingsSidebar from './components/SettingsSidebar';
import SystemPromptModal from './components/SystemPromptModal';

function App() {
    const [isMachineSidebarOpen, setMachineSidebarOpen] = useState(false);
    const [isSettingsSidebarOpen, setSettingsSidebarOpen] = useState(false);
    const [isSystemPromptModalOpen, setSystemPromptModalOpen] = useState(false);
    const [machine, setMachine] = useState('Yaskawa Alarm 380500');
    const [theme, setTheme] = useState('light');
    const [language, setLanguage] = useState('en');
    const [fontSize, setFontSize] = useState(16);
    const [systemPrompt, setSystemPrompt] = useState('You are a helpful assistant.');
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleSendMessage = async (query) => {
        const newMessages = [...messages, { text: query, sender: 'user' }];
        setMessages(newMessages);
        setLoading(true);

        try {
            const res = await fetch('http://127.0.0.1:8000/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ machine, query, language, system_prompt: systemPrompt }),
            });
            const data = await res.json();
            const responseText = data.response || data.error || 'An error occurred.';
            setMessages([...newMessages, { text: responseText, sender: 'bot' }]);
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
            />
            <div className="main-content" onClick={handleContentClick}>
                <div className="top-bar">
                    <button onClick={(e) => { e.stopPropagation(); setMachineSidebarOpen(true); }}>&#9776;</button>
                    <span className="machine-name">{machine}</span>
                    <button onClick={(e) => { e.stopPropagation(); setSettingsSidebarOpen(true); }}>&#9881;</button>
                </div>
                <ChatWindow messages={messages} />
                <InputBar onSendMessage={handleSendMessage} onClearChat={handleClearChat} />
            </div>
            <SystemPromptModal 
                isOpen={isSystemPromptModalOpen} 
                onClose={() => setSystemPromptModalOpen(false)} 
                systemPrompt={systemPrompt} 
                setSystemPrompt={setSystemPrompt} 
            />
        </div>
    );
}

export default App;