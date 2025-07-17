# Factory Maintenance RAG Assistant

An intelligent assistant for troubleshooting machine problems using Retrieval-Augmented Generation (RAG) and local/cloud LLMs.

**Tech Stack:** Python | FastAPI | React | LangChain | Docker

<br>

![Screenshot_1](https://github.com/user-attachments/assets/b4631fcf-c62a-4975-acbf-bd4f1010bcfe)

![Screenshot_2](https://github.com/user-attachments/assets/eb3c4e2b-bfd2-4263-b0d9-d57b2e034e35)

<br>

## 🎯 Project Overview

This application provides factory employees with a powerful tool for diagnosing and resolving machinery issues. It leverages a Retrieval-Augmented Generation (RAG) pipeline to deliver precise, context-aware answers based **exclusively** on uploaded technical manuals. Users can select a machine, describe a problem or error code, and receive a structured, actionable response without relying on external internet sources.

**Objective:**
The primary goal is to minimize equipment downtime and empower on-site personnel by transforming dense technical documents into an interactive and intelligent conversational assistant.

**Key Features:**
*   **Conversational QA:** A clean, intuitive chat interface for asking questions and receiving AI-generated solutions.
*   **Multi-LLM Support:** Seamlessly switch between local models (via Llama.cpp) and major API providers (Google Gemini, OpenAI, Anthropic).
*   **Intelligent Fallback:** Automatically cycles through available models if the primary choice fails, ensuring high availability.
*   **Dynamic Document Management:** A simple UI to upload, index, and delete machine-specific PDF manuals. The backend automatically creates and manages a dedicated vector store for each document.
*   **Optimized Retrieval:** Utilizes a specialized document chunking strategy and metadata filtering to provide highly accurate results for numeric error codes.
*   **Customizable UI:**
    *   **Theme:** Toggle between light and dark modes.
    *   **Language:** Switch between English and Romanian.
    *   **System Prompt:** Edit the core instructions given to the AI to tailor its response style and personality.
    *   **Drag-and-Drop:** Reorder machines and settings in the sidebars to your preference.

## 🛠️ Technology Stack

*   **Backend:**
    *   **Framework:** Python 3.8+ with **FastAPI**
    *   **LLM Orchestration:** **LangChain**
    *   **Vector Database:** **ChromaDB** for efficient similarity search.
    *   **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace.
    *   **Document Loading:** `PyMuPDF` for robust PDF parsing.
    *   **Local LLMs:** `llama-cpp-python` for running GGUF models.

*   **Frontend:**
    *   **Framework:** **React.js** (v18+)
    *   **UI Components:** Modern functional components with Hooks.
    *   **Styling:** Plain CSS with theme support.
    *   **Drag & Drop:** `@dnd-kit` for a customizable user experience.

*   **Deployment:**
    *   **Containerization:** **Docker** & **Docker Compose** for easy, reproducible deployment.

## ⚙️ Manual Setup and Run

### Prerequisites
*   Python 3.8+ and `pip`
*   Node.js 16+ and `npm`

### 1. Backend Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv .venv

    # Activate it (Windows)
    .venv\Scripts\activate

    # Activate it (macOS/Linux)
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Configure API Keys:**
    Create a `.env` file in the `backend` directory. API keys can also be set later in the UI.
    ```env
    GEMINI_API_KEY="your_google_api_key"
    OPENAI_API_KEY="your_openai_api_key"
    ANTHROPIC_API_KEY="your_anthropic_api_key"
    ```

5.  **Start the server:**
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    The backend is now running at `http://127.0.0.1:8000`.

### 2. Frontend Setup

1.  **Open a new terminal.**

2.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

3.  **Install dependencies:**
    ```bash
    npm install
    ```

4.  **Start the development server:**
    ```bash
    npm start
    ```
    This will open the application in a new browser tab at `http://localhost:3000`.

## 🐳 Docker Deployment

For the most convenient setup, you can run the application using a pre-built Docker image.

**Note:** The Docker image comes pre-packaged with the **`gemma-3-4b-it-Q8_K_M.gguf`** model, allowing you to run the application completely offline out-of-the-box.

1.  **Pull the latest image from Docker Hub:**
    ```bash
    docker pull radubalaita/rag-app:latest
    ```

2.  **Run the container:**
    ```bash
    docker run -p 8080:8080 radubalaita/rag-app:latest
    ```
    
3.  **Access the application:**
    Open your browser and navigate to **`http://localhost:8080`**.

## 🚀 Usage Guide

1.  **Open the Application:** Access the UI at `http://localhost:3000` (for manual setup) or `http://localhost:8080` (for Docker).
2.  **Upload a Document:**
    *   Click the settings icon (⚙️) to open the right sidebar.
    *   Select "Manage Documents".
    *   Give your machine a name (e.g., "Siemens CNC 840D"), select its PDF manual, and click "Upload".
3.  **Select Your Machine:** Click the menu icon (☰) to open the left sidebar and select the machine you just uploaded.
4.  **Ask a Question:** Type an error code into the input bar and press Enter.
5.  **Get a Solution:** The AI will generate a structured response based on the manual's content.
