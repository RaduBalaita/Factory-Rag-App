import React, { useState } from 'react';
import './App.css';

function App() {
    const [machine, setMachine] = useState('Yaskawa Alarm 380500');
    const [query, setQuery] = useState('');
    const [response, setResponse] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setResponse('');

        try {
            const res = await fetch('http://127.0.0.1:8000/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ machine, query }),
            });
            const data = await res.json();
            setResponse(data.response || data.error);
        } catch (error) {
            setResponse('An error occurred while fetching the response.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1>Factory Maintenance Assistant</h1>
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label htmlFor="machine-select">Select Machine:</label>
                        <select id="machine-select" value={machine} onChange={(e) => setMachine(e.target.value)}>
                            <option value="Yaskawa Alarm 380500">Yaskawa Alarm 380500</option>
                            <option value="General Error Codes">General Error Codes</option>
                            <option value="Fagor CNC 8055">Fagor CNC 8055</option>
                            <option value="FC-GTR V2.1">FC-GTR V2.1</option>
                            <option value="NUM CNC">NUM CNC</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label htmlFor="query-input">Enter Problem Code:</label>
                        <textarea
                            id="query-input"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="e.g., 178, 2654, 235"
                            rows="3"
                        />
                    </div>
                    <button type="submit" disabled={loading}>
                        {loading ? 'Analyzing...' : 'Get Help'}
                    </button>
                </form>
                {response && (
                    <div className="response-container">
                        <h2>Response:</h2>
                        <p>{response}</p>
                    </div>
                )}
            </header>
        </div>
    );
}

export default App;