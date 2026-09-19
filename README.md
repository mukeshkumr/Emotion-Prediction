# Emotion Prediction API

**A FastAPI-based web application that uses a BiGRU deep learning model to predict emotions from text input.** The application provides a REST API for emotion prediction and a simple web UI for interacting with the model.

---

## Table of Contents

- [Emotion Prediction API](#emotion-prediction-api)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Server](#running-the-server)
    - [`/predict` Response](#predict-response)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Change History](#change-history)
  - [Contributing](#contributing)
  - [License](#license)

---

## Project Overview

This project implements a **BiGRU (Bidirectional Gated Recurrent Unit)** neural network for text-based emotion classification. The model is loaded at startup using FastAPI's `lifespan` context manager, and the API exposes three primary endpoints:

- `/` – Root endpoint returning a welcome message.
- `/health` – Health check that confirms server status and model loading.
- `/predict` – POST endpoint that accepts a text payload and returns the predicted emotion, confidence score, and probability distribution.

The frontend consists of static HTML/CSS/JS files located in the `static/` directory, providing a simple UI for typing text and receiving emotion predictions.

---

## Features

- FastAPI backend with CORS support.
- BiGRU model inference with tokenizer.
- RESTful API endpoints for health checks and emotion prediction.
- Web UI for interactive use.
- Text preprocessing pipeline (lower‑casing, removal of apostrophes, special characters, and extra spaces).
- Easily extensible data models (`TextInput`, `PredictionResponse`, `HealthResponse`).

---

## Prerequisites

- **Python 3.12+** (recommended).
- A virtual environment (highly recommended).
- Git (for cloning/fetching the repository).

---

## Installation

1. **Clone or download the project:**

   ```bash
   git clone <repository-url>
   cd Emotion-Prediction
   ```

2. **Create and activate a virtual environment:**

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS / Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the dependencies:**

   ```bash
   # Using the project's pyproject.toml
   pip install -r pyproject.toml

   # Or install individual packages:
   pip install fastapi uvicorn keras numpy tensorflow
   ```

4. **Verify the model artifacts:**

   - Ensure `artifacts/BiGRU_Model.keras` exists (the file was renamed from `BiGRU_Modle.keras`).
   - Ensure `artifacts/tokenizer.pkl` is present.

---

## Running the Server

Start the FastAPI development server from the project root (with the virtual environment activated):

```bash
uvicorn main:app --reload

The server will be available at `http://127.0.0.1:8000`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Returns a welcome message. |
| `GET` | `/health` | Returns server status and whether the model is loaded. |
| `POST` | `/predict` | Predicts the emotion for the provided text. |

### `/predict` Request

```json
{
  "text": "I feel so happy and excited"
}
```

**Headers:** `Content-Type: application/json`

### `/predict` Response

```json
{
  "predicted_emotion": "joy",
  "confidence": 0.93,
  "all_probabilities": {
    "joy": 0.93,
    "sadness": 0.04,
    "anger": 0.01,
    "fear": 0.01,
    "neutral": 0.01
  }
}
```

---

## Usage

- **Via CLI with curl:**

  ```bash
  curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d '{"text":"I feel so happy and excited"}'
  ```

- **Via the web UI:** Open a browser and navigate to `http://localhost:8000`. Type a sentence and click **Read the mood** or press **Ctrl+Enter**.

- **Health check:**

  ```bash
  curl http://localhost:8000/health
  # Expected: {"status":"Server is running","model_loaded":true}
  ```

---

## Project Structure

```
Emotion-Prediction/
├── artifacts/           # Model files (BiGRU_Model.keras, tokenizer.pkl)
├── main.py              # FastAPI application, models, and endpoints
├── pyproject.toml       # Project dependencies and configuration
├── start_server.py      # Helper script to launch the server
├── static/
│   ├── index.html       # Web UI entry point
│   ├── script.js        # Frontend logic
│   └── style.css        # Styling
├── changes_happen.md    # Detailed change‑history file
├── .venv/               # Virtual environment
├── .gitignore
└── README.md            # This file
```

---

## Change History

All notable changes to this project are documented in **`changes_happen.md`**. This file records:

- Initial setup and feature implementation.
- Model filename fix (`BiGRU_Modle.keras` → `BiGRU_Model.keras`).
- Dependency installations and environment setup.
- Any bug fixes, refactors, or additions.

> **Note:** For day‑to‑day development, ensure every file change is also reflected in `changes_happen.md` (or the project‑specific `file_changes.md` if your workflow uses that naming).

---

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/foo`).
3. Make your changes, adding or updating `changes_happen.md` as appropriate.
4. Commit your changes (`git commit -m "Add foo feature"`).
5. Push to your fork (`git push origin feature/foo`).
6. Open a Pull Request.

Please follow the existing code style and ensure all new endpoints or modifications are reflected in the change‑history file.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file (if present) for full terms.

---