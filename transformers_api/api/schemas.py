from pydantic import BaseModel


class CompletionArgs(BaseModel):
    model: str = "gpt2"
    prompt: str
    temperature: float = 1.0
    max_new_tokens: int = 512
    stopping_criteria: str | None = None
    top_p: float = 1.0
    repetition_penalty: float = 1.0
