## About the Application

This application aims to provide factory employees with an intelligent assistant for troubleshooting machine problems. By leveraging Retrieval-Augmented Generation (RAG), it allows users to select a specific machine and input an error code or problem description. The AI then processes this input against the machine's dedicated documentation (without relying on external internet sources) to provide a structured explanation of the problem, probable causes, and step-by-step solutions.

Its primary goal is to streamline the troubleshooting process, reduce downtime, and empower employees with immediate, accurate information directly from the relevant technical manuals.

# Factory Maintenance RAG Application

This project is a web-based application designed to assist factory employees by providing quick, AI-driven solutions to problems with machinery. It uses a Retrieval-Augmented Generation (RAG) model to pull information from specific machine documentation, ensuring the advice is accurate and relevant.

## Features

- **Chat-based Interface:** Interact with the assistant in a familiar chat window.
- **Machine Selection:** Choose from a list of supported machines in a dedicated sidebar.
- **Multiple LLM Providers:** Support for local models (LlamaCpp) and API providers (Google Gemini, OpenAI, Anthropic).
- **Intelligent Fallback System:** Automatic failover to alternative models when primary selection fails.
- **Document Management:** Upload, index, and delete machine documentation via web interface.
- **Customizable Settings:**
    - **Model Configuration:** Switch between local and cloud-based AI models.
    - **API Key Management:** Configure API keys for different providers directly in the UI.
    - **Theme:** Switch between light and dark mode.
    - **Language:** Change the application and model response language (English/Romanian).
    - **System Prompt:** Modify the AI's system prompt to tailor its behavior.

## Advanced Features

### LLM Provider Support
- **Local Models:** Privacy-first LlamaCpp integration with GGUF model files
- **Google Gemini:** Fast and efficient API-based processing
- **OpenAI GPT:** Industry-standard language model integration  
- **Anthropic Claude:** Advanced reasoning capabilities

### Intelligent Fallback System
The application automatically tries alternative models when the primary selection fails:
1. **User's chosen model** (primary)
2. **Local model** (first fallback - privacy & cost benefits)
3. **Google Gemini** (second fallback)
4. **OpenAI GPT** (third fallback)
5. **Anthropic Claude** (final fallback)

### Enhanced Error Code Retrieval
- **SelfQueryRetriever:** Exact metadata matching for precise error code lookup
- **Unicode Support:** Handles various quote characters in technical documentation
- **Optimized Chunking:** Smart document parsing specifically designed for error code manuals

## Project Status

**✅ COMPLETE: Full-Featured RAG Application**

- [X] **UI Overhaul:** Modern chat-based interface with sidebars
- [X] **Multi-LLM Support:** Local and API model integration
- [X] **Fallback System:** Automatic model switching for reliability
- [X] **Document Management:** Upload/delete functionality with enhanced Windows file handling
- [X] **Settings Management:** Theme, language, system prompt, and model configuration
- [X] **Error Code Optimization:** Precise retrieval using SelfQueryRetriever
- [X] **API Key Management:** Frontend-based configuration with environment fallback
- [X] **Enhanced File Handling:** Robust Windows-compatible file deletion with process management

## Technology Stack

*   **Backend:**
    *   **Framework:** Python with FastAPI
    *   **LLM Integration:** `langchain` with multiple providers:
        - `langchain-google-genai` (Google Gemini)
        - `langchain-openai` (OpenAI GPT)
        - `langchain-anthropic` (Anthropic Claude)
        - `llama-cpp-python` (Local LlamaCpp models)
    *   **Vector Database:** ChromaDB (migrated from FAISS for better performance)
    *   **Document Processing:** PyMuPDFLoader, HuggingFace embeddings, SelfQueryRetriever
    *   **Process Management:** psutil for enhanced file handle management
*   **Frontend:**
    *   **Framework:** React.js with modern component architecture
    *   **Styling:** CSS with dark/light theme support
    *   **State Management:** localStorage for persistent settings

## How to Run

### Prerequisites
- Python 3.8+
- Node.js 16+ and npm
- (Optional) Local GGUF model files for offline operation

### Environment Variables
Create a `.env` file in the backend directory with your API keys:
```bash
GEMINI_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Optional
```

**Note:** API keys can also be configured directly in the application's UI under Settings > Change Model.

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
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
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
This will open a new browser tab with the application at `http://localhost:3000`.

### 3. Local Model Setup (Optional)

For offline operation with local models:

1. Create a `.models` directory in the project root:
    ```bash
    mkdir C:\Users\Tempest\Desktop\RAG\.models
    ```
2. Download GGUF model files and place them in the `.models` directory
3. Local models will automatically appear in the model selection dropdown

## Docker Deployment

For easy deployment using Docker:

### Build and Run with Docker

1. **Build the Docker image:**
    ```bash
    docker build -t rag-app .
    ```

2. **Run the container (using built-in volume defaults):**
    ```bash
    docker run -p 8080:8080 rag-app
    ```

3. **Or run with custom volume paths:**
    ```bash
    docker run -p 8080:8080 \
      -v ./docs:/app/docs \
      -v ./backend/faiss_index:/app/backend/faiss_index \
      -v ./.models:/app/.models \
      rag-app
    ```

4. **Access the application:**
    Open your browser to `http://localhost:8080`

### Data Persistence Options
- **Default (built-in volumes):** Data persists in Docker-managed volumes
- **Custom paths:** Mount your own directories for easier access:
  - `-v ./docs:/app/docs` - Uploaded PDF documents
  - `-v ./backend/faiss_index:/app/backend/faiss_index` - Vector databases
  - `-v ./.models:/app/.models` - Local GGUF model files

### Environment Variables (Optional)
You can pass API keys as environment variables:
```bash
docker run -p 8080:8080 \
  -e GEMINI_API_KEY=your_key_here \
  rag-app
```

## Usage

1. **Upload Documents:** Use "Manage Documents" in the settings to upload machine manuals (PDF format)
2. **Select Machine:** Choose your machine from the sidebar
3. **Configure Model:** Select your preferred LLM provider in Settings > Change Model
4. **Ask Questions:** Enter error codes or describe problems in the chat interface
5. **Get Solutions:** Receive structured responses with problem descriptions, causes, and step-by-step solutions

## API Endpoints

- `GET /api/machines` - List available machines
- `POST /api/machines/upload` - Upload new machine documentation
- `DELETE /api/machines/{name}` - Delete machine and its documentation
- `GET /models` - List available local models
- `POST /query` - Process error code queries
- `GET/POST /prompt_template` - Manage system prompt templates
- `GET /debug/error_codes/{machine}` - Debug endpoint for indexed error codes

## Available Machines

The application supports any machine with uploaded PDF documentation. Examples include:
- CNC Machines (Fagor, NUM, etc.)
- Industrial Controllers  
- Manufacturing Equipment
- Custom machinery with technical manuals
