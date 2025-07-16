import React, { useState, useEffect } from 'react';
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

const MachineSidebar = ({ isOpen, onClose, setMachine }) => {
    const [machines, setMachines] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchMachines = async () => {
            setLoading(true);
            try {
                const response = await fetch('http://127.0.0.1:8000/api/machines');
                const data = await response.json();
                if (data.machines) {
                    const machineList = data.machines.map((name, index) => ({
                        id: `machine-${index}`,
                        name: name
                    }));
                    setMachines(machineList);
                } else {
                    setMachines([]);
                }
            } catch (error) {
                console.error('Failed to fetch machines:', error);
                setMachines([]);
            } finally {
                setLoading(false);
            }
        };

        if (isOpen) {
            fetchMachines();
        }
    }, [isOpen]);

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
                        {loading ? <p>Loading...</p> : machines.map(machine => (
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