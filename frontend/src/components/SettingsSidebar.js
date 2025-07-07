import React, { useState } from 'react';
import { DndContext, closestCenter, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import { SortableContext, useSortable, arrayMove, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

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

const SettingsSidebar = ({ isOpen, onClose, theme, setTheme, language, setLanguage, openSystemPrompt, fontSize, setFontSize }) => {
    const [settings, setSettings] = useState([
        { id: 'theme', content: <div className="setting-item"><span>Theme</span><button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>{theme === 'light' ? 'Dark' : 'Light'} Mode</button></div> },
        { id: 'language', content: <div className="setting-item"><span>Language</span><button onClick={() => setLanguage(language === 'en' ? 'ro' : 'en')}>{language === 'en' ? 'Romanian' : 'English'}</button></div> },
        { id: 'font-size', content: <div className="setting-item"><span>Font Size</span><div><button onClick={() => setFontSize(fontSize - 1)}>-</button><span>{fontSize}px</span><button onClick={() => setFontSize(fontSize + 1)}>+</button></div></div> },
        { id: 'system-prompt', content: <div className="setting-item"><button onClick={openSystemPrompt}>Edit System Prompt</button></div> },
    ]);

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
            setSettings((items) => {
                const oldIndex = items.findIndex((item) => item.id === active.id);
                const newIndex = items.findIndex((item) => item.id === over.id);
                return arrayMove(items, oldIndex, newIndex);
            });
        }
    };

    return (
        <div className={`sidebar settings-sidebar ${isOpen ? 'open' : ''}`}>
            <div className="sidebar-header">
                <button onClick={onClose} className="close-btn">&times;</button>
                <h3>Settings</h3>
            </div>
            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
                <SortableContext items={settings} strategy={verticalListSortingStrategy}>
                    <div className="sidebar-content">
                        {settings.map(setting => (
                            <SortableItem key={setting.id} id={setting.id}>
                                {setting.content}
                            </SortableItem>
                        ))}
                    </div>
                </SortableContext>
            </DndContext>
        </div>
    );
};

export default SettingsSidebar;