# Streamlit RAG App

This project implements a Retrieval-Augmented Generation (RAG) system using a local Ollama setup and a local vector database for storing embeddings. The application allows users to upload documents, which are stored in the vector database, and it can answer queries based on the context from the vector database.

## Features

- Document upload interface
- Query interface for retrieving answers based on uploaded documents
- Integration with Ollama for generating embeddings and responses
- Local vector database for efficient storage and retrieval of embeddings

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd streamlit-rag-app
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env` and fill in the necessary values.

## Usage

To run the Streamlit application, use the following command:

```bash
streamlit run src/app.py
```

## Directory Structure

- `src/`: Contains the main application code.
- `data/`: Directory for storing the local vector database files.
- `requirements.txt`: Lists the dependencies required for the project.
- `.env.example`: Example environment variables needed for the application.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.