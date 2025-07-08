import React, { useState, useEffect } from 'react';
import { DndContext, closestCenter, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import { SortableContext, useSortable, arrayMove, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

const initialMachines = [
    { id: 'yaskawa', name: 'Yaskawa Alarm 380500' },
    { id: 'general', name: 'General Error Codes' },
    { id: 'fagor', name: 'Fagor CNC 8055' },
    { id: 'fc-gtr', name: 'FC-GTR V2.1' },
    { id: 'num-cnc', name: 'NUM CNC' },
];

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

const MachineSidebar = ({ isOpen, onClose, setMachine }) => {
    const [machines, setMachines] = useState(() => {
        const savedMachines = localStorage.getItem('machines');
        if (savedMachines) {
            return JSON.parse(savedMachines);
        }
        return initialMachines;
    });

    useEffect(() => {
        localStorage.setItem('machines', JSON.stringify(machines));
    }, [machines]);

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
            setMachines((items) => {
                const oldIndex = items.findIndex((item) => item.id === active.id);
                const newIndex = items.findIndex((item) => item.id === over.id);
                return arrayMove(items, oldIndex, newIndex);
            });
        }
    };

    return (
        <div className={`sidebar machine-sidebar ${isOpen ? 'open' : ''}`}>
            <div className="sidebar-header">
                <h3>Machines</h3>
                <button onClick={onClose} className="close-btn">&times;</button>
            </div>
            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
                <SortableContext items={machines} strategy={verticalListSortingStrategy}>
                    <div className="sidebar-content">
                        {machines.map(machine => (
                            <SortableItem key={machine.id} id={machine.id}>
                                <div className="machine-item" onClick={() => setMachine(machine.name)}>
                                    {machine.name}
                                </div>
                            </SortableItem>
                        ))}
                    </div>
                </SortableContext>
            </DndContext>
        </div>
    );
};

export default MachineSidebar;