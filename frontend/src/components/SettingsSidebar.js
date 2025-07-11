import React, { useState, useEffect } from 'react';
import { DndContext, closestCenter, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import { SortableContext, useSortable, arrayMove, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

const translations = {
    en: {
        settings: 'Settings',
        theme: 'Theme',
        lightMode: 'Light Mode',
        darkMode: 'Dark Mode',
        language: 'Language',
        english: 'English',
        romanian: 'Romanian',
        fontSize: 'Font Size',
        systemPrompt: 'System Prompt',
        edit: 'Edit',
        changeModel: 'Change Model',
    },
    ro: {
        settings: 'Setări',
        theme: 'Temă',
        lightMode: 'Mod Luminos',
        darkMode: 'Mod Întunecat',
        language: 'Limbă',
        english: 'Engleză',
        romanian: 'Română',
        fontSize: 'Dimensiune Font',
        systemPrompt: 'Prompt Sistem',
        edit: 'Editează',
        changeModel: 'Schimbă Modelul',
    },
};

const initialSettingIds = ['theme', 'language', 'font-size', 'system-prompt', 'change-model'];

const SortableItem = ({ id, children }) => {
    const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ id });

    const style = {
        transform: CSS.Transform.toString(transform),
        transition,
    };

    return (
        <div ref={setNodeRef} style={style} {...attributes} {...listeners}>
            {children}
        </div>
    );
};

const SettingsSidebar = ({ isOpen, onClose, theme, setTheme, language, setLanguage, openSystemPrompt, fontSize, setFontSize, openChangeModel, modelConfig }) => {
    const [settingIds, setSettingIds] = useState(() => {
        // Temporarily force reset to initial settings
        return initialSettingIds;
        
        // const savedSettingIds = localStorage.getItem('settingIds');
        // if (savedSettingIds) {
        //     return JSON.parse(savedSettingIds);
        // }
        // return initialSettingIds;
    });

    useEffect(() => {
        localStorage.setItem('settingIds', JSON.stringify(settingIds));
    }, [settingIds]);

    const t = translations[language];

    const getSettingContent = (id) => {
        switch (id) {
            case 'theme':
                return <div className="setting-item"><span>{t.theme}</span><button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>{theme === 'light' ? t.darkMode : t.lightMode}</button></div>;
            case 'language':
                return <div className="setting-item"><span>{t.language}</span><button onClick={() => setLanguage(language === 'en' ? 'ro' : 'en')}>{language === 'en' ? t.romanian : t.english}</button></div>;
            case 'font-size':
                return <div className="setting-item"><span>{t.fontSize}</span><div><button onClick={() => setFontSize(14)}>14</button><button onClick={() => setFontSize(16)}>16</button><button onClick={() => setFontSize(18)}>18</button><button onClick={() => setFontSize(20)}>20</button></div></div>;
            case 'system-prompt':
                return <div className="setting-item"><span style={{ color: 'red' }}>{t.systemPrompt}</span><button onClick={openSystemPrompt}>{t.edit}</button></div>;
            case 'change-model':
                return <div className="setting-item"><span>{t.changeModel} ({modelConfig.type === 'local' ? 'Local' : `Cloud - ${modelConfig.provider}`})</span><button onClick={openChangeModel}>{t.edit}</button></div>;
            default:
                return null;
        }
    };

    const sensors = useSensors(
        useSensor(PointerSensor, {
            activationConstraint: {
                delay: 100, // 0.1 second delay
                tolerance: 5,
            },
        })
    );

    const handleDragEnd = (event) => {
        const { active, over } = event;
        if (active.id !== over.id) {
            setSettingIds((items) => {
                const oldIndex = items.findIndex((item) => item === active.id);
                const newIndex = items.findIndex((item) => item === over.id);
                return arrayMove(items, oldIndex, newIndex);
            });
        }
    };

    return (
        <div className={`sidebar settings-sidebar ${isOpen ? 'open' : ''}`}>
            <div className="sidebar-header">
                <button onClick={onClose} className="close-btn">&times;</button>
                <h3>{t.settings}</h3>
            </div>
            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
                <SortableContext items={settingIds} strategy={verticalListSortingStrategy}>
                    <div className="sidebar-content">
                        {settingIds.map(id => (
                            <SortableItem key={id} id={id}>
                                {getSettingContent(id)}
                            </SortableItem>
                        ))}
                    </div>
                </SortableContext>
            </DndContext>
        </div>
    );
};

export default SettingsSidebar;