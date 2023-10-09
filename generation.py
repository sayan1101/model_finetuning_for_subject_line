import os
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)
from finetuned_model import FinetunedModel
import torch
from typing import List, Union, Dict


REPO_ID = "chats-bug/multiple-subject-gen-finetuned"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

finetuned_model = FinetunedModel(
    REPO_ID,
    quantization_config = quantization_config,
    load_in_4bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Nous-Hermes-Llama2-13b",
    trust_remote_code=True
)

def inference(message: str) -> Dict[str, Union[bool, List[Dict[str, str]]]]:
    responses = []
    for i in range (4):
        model_responses = finetuned_model.generate(
            # model=finetuned_model,
            message=message,
            # tokenizer=tokenizer,
            max_new_tokens=100,
            temperature=0.9,
            num_return_sequences=1,
            top_p=0.85,
            device="cuda"
        )
        for response in model_responses:
            # add the response to "text"
            responses.append({"text": response.strip()[3:]})

    output = {
        "status": True,
        "message": responses
    }
    
    return output

        

