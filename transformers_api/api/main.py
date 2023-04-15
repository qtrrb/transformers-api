from fastapi import FastAPI, APIRouter
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers_api.generation.model import load_model, generate, stream_generate
from transformers_api.util import get_model_names
from transformers_api.api.schemas import CompletionArgs

app = FastAPI(
    title="transformers-api",
    description="An API to interact with Large Language Models",
)
router = APIRouter(prefix="/v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    load_model("gpt2")


@router.get("/models")
def models():
    return get_model_names()


@router.post("/completion")
def generate_text(args: CompletionArgs):
    load_model(args.model)

    generated_text = generate(
        prompt=args.prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        stopping_criteria=args.stopping_criteria,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    return Response(content=generated_text, media_type="text/html")


@router.post("/completion_stream")
async def stream_generate_text(args: CompletionArgs):
    load_model(args.model)

    def generate_tokens():
        for token in stream_generate(
            prompt=args.prompt,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            stopping_criteria=args.stopping_criteria,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ):
            yield token

    return StreamingResponse(generate_tokens(), media_type="text/event-stream")


app.include_router(router)
