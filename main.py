import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


from ast import List
from dotenv import load_dotenv
from database import init_db, save_prediction, get_history, get_history_count, cleanup_old_predictions

init_db()

from typing import List
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from keras.models import load_model
import numpy as np
import pickle
import string
import re


load_dotenv()


"""
1. We are going to make some constants like:
A. Model Path (BiGRU)
B. Tokenizer Path
C. Max Sequence Length
D. Emotion Labels
E. Emotion emojis
"""
#A. Model Path (BiGRU)
model_path = os.getenv("MODEL_PATH", "artifacts/BiGRU_Model.keras")

#B. Tokenizer Path
tokenizer_path = os.getenv("TOKENIZER_PATH", "artifacts/tokenizer.pkl")

#C. Max Sequence Length
max_sequence_length = int(os.getenv("MAX_SEQ_LENGTH", 50))

#D. Emotion Labels
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

#E. Emotion emojis
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}



"""
2. Preprocess the upcoming text
Cleans raw text so it matches the format used while training.
A. Convert the text to lowercase. -done
B. Remove apostrophes (e.g can't -> cant). -done
C. Remove Special Characters and Punctuation. -done
D. Remove extra spaces -done
"""


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)  # Remove apostrophes
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # Remove special chars
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    
    # NEW: Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # NEW: Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)
    
    # NEW: Keep emojis if desired (or remove them)
    # text = re.sub(r":\w+:", "", text)  # Remove text emojis
    
    return text


"""
3. Request and Response Schemas
A. Text Input -> Input schema the text sent by user. -done
B. Prediction Response -> Output schema the emotion to predict. -done
C. Health Response (Server health check)
"""



class TextInput(BaseModel):
    text : str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"}
        )

class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence : float
    all_probabilites: dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class PredictionHistoryEntry(BaseModel):
    id: int
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float] | None = None
    created_at: str

# class TextsInput(BaseModel):
#     texts: List[str] = Field(
#         ...,
#         min_length=1,
#         max_length=10,
#         description="List of sentences to analyze",
#     )

"""
4. Model Loading and LifeSpan Management
Load the model and toknizer once the server starts up.
"""
dl_model = {} #{1. BiGRU, 2. Tokenizer}-> True , {} -> False

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading the model and tokenizer...')
    dl_model["BiGRU"] = load_model(model_path)                      #BiGRU Model
    with open(tokenizer_path, 'rb') as file:
        dl_model["Tokenizer"] = pickle.load(file)
    print('Model are loaded successfully...')   

    yield #Pause, model is laoded and server is running and at this point model wait karega for request

    dl_model.clear() #Ek baar server band ho gaya uske baad model ko memory se hata do.
               

"""
5. Mount the static files to the FastAPI app
A. Enable CORS (Cross-Origin Resource Sharing) to allow requests from different origins.
"""
app = FastAPI(
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount('/static', StaticFiles(directory="static"), name="static")



"""
6. API Endpoints.
A. Server UI at homepage ('/')
B. Health Check Endpoint ('/health')
C. Predict Emotion Endpoint ('/predict')
"""



#A. Server UI at homepage ('/')
@app.get('/', include_in_schema=False)
def server_ui():
    return FileResponse('static/index.html')

@app.get('/v1/models')
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "my-model",
                "object": "model",
                "owned_by": "local"
            }
        ]
        
    }
    

#B. Health Check Endpoint ('/health')
@app.get('/health', response_model=HealthResponse)
def health_check():
    return HealthResponse(status="Server is running", model_loaded=bool(dl_model))




#C. Predict Emotion Endpoint ('/predict')
@app.post('/predict', response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    """
    1. Cleans the input sentences.
    2. Convert the words into numeric using tokenizer.
    3. Pad the sequences to ensure uniform length.
    4. Run prediction using the BiGRU model.
    5. Return the top emotion and full probability breakdown.
    """

    BiGRU_model     = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

    #1. 
    cleaned_text = preprocess_text(text_input.text)

    #2. and 3. 
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    probabilites     = BiGRU_model.predict(padded_sequence)[0]

    top_emotion_index = int(np.argmax(probabilites)) # 4
    all_probabilites =  {
        label: float(prob) for prob, label in zip(probabilites, emotion_labels)
          
    }
    
    MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 0.3))
    if probabilites[top_emotion_index] < MIN_CONFIDENCE:
       predicted_emotion = "uncertain"
       confidence = float(probabilites[top_emotion_index])
    else:
       predicted_emotion = emotion_labels[top_emotion_index]
       confidence = float(probabilites[top_emotion_index])

    # Save to prediction history
    save_prediction(
        text=text_input.text,
        predicted_emotion=predicted_emotion,
        confidence=confidence,
        all_probabilities=all_probabilites,
    )

    # Cleanup if history exceeds limit
    cleanup_old_predictions(max_count=500)


    return PredictionResponse(
        text = text_input.text,
        predicted_emotion = predicted_emotion,
        confidence = confidence, 
        all_probabilites = all_probabilites
    )


#D. Prediction History Endpoint
@app.get('/history', response_model=list[PredictionHistoryEntry])
def prediction_history(
    limit: int = 50,
    offset: int = 0,
):
    """Retrieve prediction history with pagination."""
    items = get_history(limit=limit, offset=offset)
    total = get_history_count()
    return {"items": items, "total": total}


#E. Clear History Endpoint
@app.delete('/history')
def clear_history():
    """Clear all prediction history."""
    from database import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return {"detail": "History cleared successfully"}