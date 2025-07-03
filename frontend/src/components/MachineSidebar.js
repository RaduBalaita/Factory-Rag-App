import React from 'react';

const MachineSidebar = ({ isOpen, onClose, setMachine }) => {
    const machines = [
        'Yaskawa Alarm 380500',
        'General Error Codes',
        'Fagor CNC 8055',
        'FC-GTR V2.1',
        'NUM CNC'
    ];

    return (
        <div className={`sidebar machine-sidebar ${isOpen ? 'open' : ''}`}>
            <button onClick={onClose}>Close</button>
            <h3>Select Machine</h3>
            <ul>
                {machines.map(machine => (
                    <li key={machine} onClick={() => setMachine(machine)}>{machine}</li>
                ))}
            </ul>
        </div>
    );
};

export default MachineSidebar;