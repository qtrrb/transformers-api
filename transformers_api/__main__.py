import uvicorn


if __name__ == "__main__":
    uvicorn.run("transformers_api.api.main:app", host="0.0.0.0")
