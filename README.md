# Assignment 1: FastAPI Implementation

This project implements a FastAPI application for Assignment 1.

The API includes two endpoints:

1. `POST /generate`
   - Generates simple text using a bigram model.

2. `POST /embeddings`
   - Takes a query word and returns its spaCy word embedding.
   - The embedding is generated using the `en_core_web_lg` model.

## Setup

Install dependencies:

```bash
uv sync
```

Download the spaCy language model:

```bash
uv run python -m spacy download en_core_web_lg
```

## Run the API

```bash
uv run fastapi dev main.py
```

Open the API documentation in your browser:

```text
http://127.0.0.1:8000/docs
```

## Endpoint 1: Text Generation

POST `/generate`

Example request:

```json
{
  "start_word": "the",
  "length": 10
}
```

Example response:

```json
{
  "generated_text": "the count of monte cristo is another example sentence"
}
```

## Endpoint 2: Word Embeddings

POST `/embeddings`

Example request:

```json
{
  "query_word": "king"
}
```

Example response:

```json
{
  "word": "king",
  "dimension": 300,
  "embedding": [...]
}
```

## Notes

The `/embeddings` endpoint uses the spaCy `en_core_web_lg` model to return a 300-dimensional word embedding vector for the query word.