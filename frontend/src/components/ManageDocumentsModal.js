import React, { useState, useEffect } from 'react';

const translations = {
    en: {
        manageDocuments: 'Manage Documents',
        uploadDocument: 'Upload Document',
        machineName: 'Machine Name',
        selectFile: 'Select PDF File',
        upload: 'Upload',
        existingDocuments: 'Existing Documents',
        delete: 'Delete',
        close: 'Close',
        uploading: 'Uploading...',
        noDocuments: 'No documents uploaded yet',
        confirmDelete: 'Are you sure you want to delete this document?',
        uploadSuccess: 'Document uploaded successfully!',
        uploadError: 'Error uploading document',
        deleteSuccess: 'Document deleted successfully!',
        deleteError: 'Error deleting document',
    },
    ro: {
        manageDocuments: 'Gestionează Documente',
        uploadDocument: 'Încarcă Document',
        machineName: 'Numele Mașinii',
        selectFile: 'Selectează Fișier PDF',
        upload: 'Încarcă',
        existingDocuments: 'Documente Existente',
        delete: 'Șterge',
        close: 'Închide',
        uploading: 'Se încarcă...',
        noDocuments: 'Nu au fost încărcate documente încă',
        confirmDelete: 'Sigur vrei să ștergi acest document?',
        uploadSuccess: 'Document încărcat cu succes!',
        uploadError: 'Eroare la încărcarea documentului',
        deleteSuccess: 'Document șters cu succes!',
        deleteError: 'Eroare la ștergerea documentului',
    }
};

const ManageDocumentsModal = ({ isOpen, onClose, language, onDocumentChange }) => {
    const [machines, setMachines] = useState([]);
    const [machineName, setMachineName] = useState('');
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [message, setMessage] = useState('');

    const t = translations[language];

    useEffect(() => {
        if (isOpen) {
            fetchMachines();
        }
    }, [isOpen]);

    const fetchMachines = async () => {
        try {
            const response = await fetch('http://127.0.0.1:8000/api/machines');
            const data = await response.json();
            if (data.machines) {
                setMachines(data.machines);
            }
        } catch (error) {
            console.error('Failed to fetch machines:', error);
        }
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file && file.type === 'application/pdf') {
            setSelectedFile(file);
        } else {
            setMessage('Please select a PDF file');
            setTimeout(() => setMessage(''), 3000);
        }
    };

    const handleUpload = async () => {
        if (!machineName.trim() || !selectedFile) {
            setMessage('Please provide a machine name and select a file');
            setTimeout(() => setMessage(''), 3000);
            return;
        }

        setUploading(true);
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(`http://127.0.0.1:8000/api/machines/upload?name=${encodeURIComponent(machineName)}`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                setMessage(t.uploadSuccess);
                setMachineName('');
                setSelectedFile(null);
                fetchMachines();
                onDocumentChange();
            } else {
                const error = await response.json();
                setMessage(t.uploadError + ': ' + error.detail);
            }
        } catch (error) {
            setMessage(t.uploadError + ': ' + error.message);
        } finally {
            setUploading(false);
            setTimeout(() => setMessage(''), 3000);
        }
    };

    const handleDelete = async (machineName) => {
        if (!window.confirm(t.confirmDelete)) {
            return;
        }

        try {
            const response = await fetch(`http://127.0.0.1:8000/api/machines/${encodeURIComponent(machineName)}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                setMessage(t.deleteSuccess);
                fetchMachines();
                onDocumentChange();
            } else {
                const error = await response.json();
                setMessage(t.deleteError + ': ' + error.detail);
            }
        } catch (error) {
            setMessage(t.deleteError + ': ' + error.message);
        } finally {
            setTimeout(() => setMessage(''), 3000);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal manage-documents-modal" onClick={(e) => e.stopPropagation()}>
                <h3>{t.manageDocuments}</h3>
                
                {message && (
                    <div className={`message ${message.includes('success') ? 'success' : 'error'}`}>
                        {message}
                    </div>
                )}

                <div className="upload-section">
                    <h4>{t.uploadDocument}</h4>
                    <div className="form-group">
                        <label>{t.machineName}</label>
                        <input 
                            type="text" 
                            value={machineName} 
                            onChange={(e) => setMachineName(e.target.value)}
                            placeholder="e.g. Siemens CNC 840D"
                        />
                    </div>
                    <div className="form-group">
                        <label>{t.selectFile}</label>
                        <input 
                            type="file" 
                            accept=".pdf" 
                            onChange={handleFileChange}
                        />
                        {selectedFile && <span className="file-name">{selectedFile.name}</span>}
                    </div>
                    <button 
                        onClick={handleUpload} 
                        disabled={uploading || !machineName.trim() || !selectedFile}
                        className="upload-btn"
                    >
                        {uploading ? t.uploading : t.upload}
                    </button>
                </div>

                <div className="existing-documents">
                    <h4>{t.existingDocuments}</h4>
                    {machines.length === 0 ? (
                        <p>{t.noDocuments}</p>
                    ) : (
                        <div className="documents-list">
                            {machines.map((machine, index) => (
                                <div key={index} className="document-item">
                                    <span className="machine-name">{machine}</span>
                                    <button 
                                        onClick={() => handleDelete(machine)}
                                        className="delete-btn"
                                    >
                                        {t.delete}
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                <button onClick={onClose} className="close-btn">{t.close}</button>
            </div>
        </div>
    );
};

export default ManageDocumentsModal;
