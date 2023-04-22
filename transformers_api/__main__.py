import uvicorn
import logging

if __name__ == "__main__":
    logging.basicConfig()
    uvicorn.run("transformers_api.api.main:app", host="0.0.0.0")
