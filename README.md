## About the Application

This application aims to provide factory employees with an intelligent assistant for troubleshooting machine problems. By leveraging Retrieval-Augmented Generation (RAG), it allows users to select a specific machine and input an error code or problem description. The AI then processes this input against the machine's dedicated documentation (without relying on external internet sources) to provide a structured explanation of the problem, probable causes, and step-by-step solutions.

Its primary goal is to streamline the troubleshooting process, reduce downtime, and empower employees with immediate, accurate information directly from the relevant technical manuals.

# Factory Maintenance RAG Application

This project is a web-based application designed to assist factory employees by providing quick, AI-driven solutions to problems with machinery. It uses a Retrieval-Augmented Generation (RAG) model to pull information from specific machine documentation, ensuring the advice is accurate and relevant.

## Features

- **Chat-based Interface:** Interact with the assistant in a familiar chat window.
- **Machine Selection:** Choose from a list of supported machines in a dedicated sidebar.
- **Customizable Settings:**
    - **Theme:** Switch between light and dark mode.
    - **Language:** Change the application and model response language (English/Romanian).
    - **System Prompt:** Modify the AI's system prompt to tailor its behavior.

## Project Status

**Phase 2: UI Overhaul (In Progress)**

- [X] Redesign the UI to a chat-based interface.
- [X] Implement sidebars for machine selection and settings.
- [X] Add theme, language, and system prompt customization.
- [ ] Update backend to handle new settings.
- [ ] Refine the chat experience and message display.

**Phase 1: MVP (Complete)**

- [X] Initial project setup and planning.
- [X] Backend: Create FastAPI server and RAG pipeline using Google's Gemini 2.5 Flash model.
- [X] Frontend: Build React UI for machine selection and problem input.
- [X] Integration: Connect frontend to backend and resolve CORS issues.
- [X] Documentation: Initial `README.md` with setup instructions.

## Technology Stack

*   **Backend:**
    *   **Framework:** Python with FastAPI
    *   **LLM Integration:** `langchain` with `langchain-google-genai`
    *   **Model:** `gemini-2.5-flash`
    *   **Document Processing:** `langchain` with `FAISS` for vector search, `HuggingFaceEmbeddings` for text embeddings, and `pypdf` for PDF loading.
*   **Frontend:**
    *   **Framework:** React.js (`create-react-app`)
    *   **Styling:** CSS

## How to Run

### Prerequisites
- Python 3.7+
- Node.js and npm

### 1. Backend Setup

1.  **Open a new terminal.**
2.  Navigate to the backend directory:
    ```bash
    cd C:\Users\Tempest\Desktop\RAG\backend
    ```
3.  Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```
4.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
5.  Start the FastAPI server:
    ```bash
    uvicorn app.main:app --reload
    ```
The backend server will now be running at `http://127.0.0.1:8000`.

### 2. Frontend Setup

1.  **Open another new terminal.**
2.  Navigate to the frontend directory:
    ```bash
    cd C:\Users\Tempest\Desktop\RAG\frontend
    ```
3.  Install the dependencies:
    ```bash
    npm install
    ```
4.  Start the React development server:
    ```bash
    npm start
    ```
This will open a new browser tab with the application at `http://localhost:3000`. You can now use the application.

## Available Machines

- Yaskawa Alarm 380500
- General Error Codes
- Fagor CNC 8055
- FC-GTR V2.1
- NUM CNC
