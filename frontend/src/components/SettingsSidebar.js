import React from 'react';

const SettingsSidebar = ({ isOpen, onClose, theme, setTheme, language, setLanguage, openSystemPrompt }) => {

    return (
        <div className={`sidebar settings-sidebar ${isOpen ? 'open' : ''}`}>
            <button onClick={onClose}>Close</button>
            <h3>Settings</h3>
            <div>
                <label>Theme:</label>
                <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>
                    {theme === 'light' ? 'Dark' : 'Light'} Mode
                </button>
            </div>
            <div>
                <label>Language:</label>
                <button onClick={() => setLanguage(language === 'en' ? 'ro' : 'en')}>
                    {language === 'en' ? 'Romanian' : 'English'}
                </button>
            </div>
            <button onClick={openSystemPrompt}>Edit System Prompt</button>
        </div>
    );
};

export default SettingsSidebar;