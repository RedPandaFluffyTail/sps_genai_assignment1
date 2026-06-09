from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy

from app.bigram_model import BigramModel


app = FastAPI(
    title="Assignment 1 API",
    description="FastAPI app with a bigram text generator and spaCy word embeddings.",
    version="1.0.0",
)


corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas",
    "It tells the story of Edmond Dantes who is falsely imprisoned and later seeks revenge",
    "This is another example sentence",
    "We are generating text based on bigram probabilities",
    "Bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    query_word: str


nlp = spacy.load("en_core_web_lg")


@app.get("/")
def read_root():
    return {
        "message": "Assignment 1 API is running",
        "endpoints": ["/generate", "/embeddings"],
    }


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length,
    )

    return {
        "generated_text": generated_text,
    }


@app.post("/embeddings")
def get_embeddings(request: EmbeddingRequest):
    word = request.query_word.strip()

    if not word:
        raise HTTPException(
            status_code=400,
            detail="query_word cannot be empty",
        )

    doc = nlp(word)

    if len(doc) != 1:
        raise HTTPException(
            status_code=400,
            detail="Please provide one word only",
        )

    token = doc[0]

    if not token.has_vector:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding found for word: {word}",
        )

    return {
        "word": token.text,
        "dimension": len(token.vector),
        "embedding": token.vector.tolist(),
    }