import torch
import os
import gc
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    StoppingCriteriaList,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers_api.generation.stopping_criteria import TextStoppingCriteria

MODEL_ROOT_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../models/"
)


tokenizer = None
model = None
model_id = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str) -> bool:
    """
    Loads a pre-trained model and tokenizer from the specified model_name.

    If the specified model_name matches the current loaded model_id, the function returns True, indicating the model is already loaded.
    Otherwise, it attempts to load the model from the MODEL_ROOT_FOLDER directory. If the model is sharded, it calls load_sharded_model to load the model,
    otherwise it calls load_unsharded_model. If successful, the function returns True, otherwise it returns False.

    Args:
        model_name (str): The name of the model directory.

    Returns:
        bool: True if the model is loaded successfully or is already loaded, False otherwise.
    """  # noqa: E501

    global tokenizer, model

    if model_name == model_id:
        return True
    else:
        if tokenizer is not None:
            tokenizer = None
        if model is not None:
            model = None

        gc.collect()
        torch.cuda.empty_cache()

        model_path = os.path.join(MODEL_ROOT_FOLDER, model_name)

        for filename in os.listdir(model_path):
            if filename.endswith(".index.json"):
                return load_sharded_model(model_name)
        return load_unsharded_model(model_name)


def load_unsharded_model(model_name: str) -> bool:
    """
    Loads an unsharded pre-trained model and tokenizer from the specified model_name directory.

    Attempts to load the model using AutoModelForCausalLM.from_pretrained with torch_dtype=torch.float16 and device_map="auto".
    If successful, sets the global variables tokenizer, model, and model_id to the loaded tokenizer, model, and model_name, respectively,
    and returns True. Otherwise, returns False.

    Args:
        model_name (str): The name of the model directory.

    Returns:
        bool: True if the model is loaded successfully, False otherwise.
    """  # noqa: E501

    global tokenizer, model, model_id

    try:
        model_path = os.path.join(MODEL_ROOT_FOLDER, model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

        model_id = model_name
        return True
    except Exception as e:
        print(e)
        return False


def load_sharded_model(model_name: str) -> bool:
    """
    Loads a sharded pre-trained model and tokenizer from the specified model_name directory.

    Attempts to load the model using AutoModelForCausalLM.from_config with an empty weight initialization, followed by
    load_checkpoint_and_dispatch with device_map="auto".
    If successful, sets the global variables tokenizer, model, and model_id to the loaded tokenizer, model, and model_name, respectively,
    and returns True. Otherwise, returns False.

    Args:
        model_name (str): The name of the model directory.

    Returns:
        bool: True if the model is loaded successfully, False otherwise.
    """  # noqa: E501

    global tokenizer, model, model_id

    try:
        model_path = os.path.join(MODEL_ROOT_FOLDER, model_name)

        config = AutoConfig.from_pretrained(model_path)

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        model.tie_weights()

        model = load_checkpoint_and_dispatch(
            model, model_path, device_map="auto", dtype=torch.float16
        )

        if type(model) is LlamaForCausalLM:
            tokenizer = LlamaTokenizer.from_pretrained(model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

        model_id = model_name
        return True
    except Exception as e:
        print(e)
        return False


def generate(
    prompt: str,
    temperature=1.0,
    max_new_tokens=512,
    stopping_criteria: str | None = None,
    top_p=1.0,
    repetition_penalty=1.0,
):
    """
    Generates text based on the input prompt using a pre-trained language model.

    Args:
        prompt (str): The input text prompt to generate text from.
        temperature (float): Controls the "creativity" of the generated text. Lower temperature values result in more conservative choices, while higher temperature values result in more diverse choices. Default is 1.0.
        max_new_tokens (int): The maximum number of new tokens to generate. Default is 512.
        stopping_criteria (str): The stopping criteria for the generated text. The generation will stop when the stopping criteria is met. Default is None.
        top_p (float): Controls the diversity of the generated text by restricting the probability of sampling less probable words. Default is 1.0.
        repetition_penalty (float): Controls the degree of repetition in the generated text. Higher values result in less repetition. Default is 1.0.

    Returns:
        str: The generated text.
    """  # noqa: E501

    global tokenizer, model

    assert isinstance(model, PreTrainedModel), "Model is not loaded."
    assert isinstance(tokenizer, PreTrainedTokenizerBase), "Model is not loaded."

    stopping_criteria_ids = (
        tokenizer(stopping_criteria, return_tensors="pt").input_ids.to(device)[0]
        if stopping_criteria is not None and stopping_criteria != ""
        else None
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList(
            [TextStoppingCriteria(stopping_criteria_ids)]
        ),
    )

    generated_text = tokenizer.decode(
        output[:, input_ids.shape[1] :][0], skip_special_tokens=True
    )

    # Remove all instances of the Unicode unknown character (�)
    generated_text = generated_text.replace("\uFFFD", "")

    return generated_text


# Based on the generate function of
# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
@torch.no_grad()
def stream_generate(
    prompt: str,
    temperature=1.0,
    max_new_tokens=512,
    stopping_criteria: str | None = None,
    top_p=1.0,
    repetition_penalty=1.0,
):
    """
    Generates text based on the input prompt using a pre-trained language model in a streaming fashion.

    Args:
        prompt (str): The input text prompt to generate text from.
        temperature (float): Controls the "creativity" of the generated text. Lower temperature values result in more conservative choices, while higher temperature values result in more diverse choices. Default is 1.0.
        max_new_tokens (int): The maximum number of new tokens to generate. Default is 512.
        stopping_criteria (str): The stopping criteria for the generated text. The generation will stop when the stopping criteria are met. Default is None.
        top_p (float): Controls the diversity of the generated text by restricting the probability of sampling less probable words. Default is 1.0.
        repetition_penalty (float): Controls the degree of repetition in the generated text. Higher values result in less repetition. Default is 1.0.

    Yields:
        str: The generated text one token at a time.
    """  # noqa: E501

    global tokenizer, model

    assert isinstance(model, PreTrainedModel), "Model is not loaded."
    assert isinstance(tokenizer, PreTrainedTokenizerBase), "Model is not loaded."

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    stopping_criteria_ids = (
        tokenizer(stopping_criteria, return_tensors="pt").input_ids.to(device)[0]
        if stopping_criteria is not None and stopping_criteria != ""
        else None
    )

    stop = TextStoppingCriteria(stopping_criteria_ids)

    original_size = len(input_ids[0])

    logits_processor = LogitsProcessorList(
        [
            TemperatureLogitsWarper(temperature=temperature),
            RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty),
            TopPLogitsWarper(top_p=top_p),
        ]
    )

    while True:
        # Get logits for the next token
        logits = model(input_ids).logits[:, -1, :]
        # Apply logits processor
        next_tokens_scores = logits_processor(input_ids, logits)

        probs = torch.nn.functional.softmax(next_tokens_scores, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # Note: This is done to handle Sentencepiece based tokenizers,
        # as they don't preprend the prefix space to the start of a word
        tokens_previous = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        input_ids = torch.cat((input_ids, next_token_id), dim=1)
        tokens = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        new_tokens = tokens[len(tokens_previous) :]

        # Remove all instances of the Unicode unknown character (�)
        new_tokens = new_tokens.replace("\uFFFD", "")

        yield new_tokens

        if stop(input_ids, next_tokens_scores):
            break

        if len(input_ids[0]) >= original_size + max_new_tokens:
            break

        if input_ids[0][-1].item() == tokenizer.eos_token_id:
            break
