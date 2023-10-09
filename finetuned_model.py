from peft import AutoPeftModelForCausalLM
import torch
import os
from transformers import AutoTokenizer, StoppingCriteria
from typing import List, Union, Optional


class FinetunedModel:
    def __init__(self, model_id: str, quantization_config=None, device_map="auto", load_in_4bit: bool = False, load_in_8bit: bool = False):
        self.model_id = model_id
        self.quantization_config = quantization_config
        self.device_map = device_map
        self.model_loaded = False
        self._load_model_and_tokenizer(load_in_4bit, load_in_8bit)


    def _load_model_and_tokenizer(self, load_in_4bit: bool = False, load_in_8bit: bool = False):
        try:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                self.model_id,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                device_map=self.device_map,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "NousResearch/Nous-Hermes-Llama2-13b",
                trust_remote_code=True,
            )
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            # Suppress fast_tokenizer warning
            self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

            self.model_loaded = True
        except Exception as e:
            print(f"Error loading the model: \n{e}")
            self.model = None
            self.tokenizer = None
            self.model_loaded = False

    def generate(
        self,
        message: Union[str, list[str]],
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 0.7,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        num_return_sequences: int = 1,
        num_beams: Optional[int] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: float = 0.0,
        stopping_criteria: Optional[StoppingCriteria] = None,
        seed: Optional[List[int]] = None,
        device = "auto"
    ) -> List[str]:
        generation_config = self.model.generation_config
        generation_config.max_new_tokens = max_new_tokens
        generation_config.temperature = temperature
        generation_config.do_sample = temperature > 0.0
        generation_config.top_p = top_p
        generation_config.num_return_sequences = num_return_sequences
        generation_config.pad_token_id = self.tokenizer.eos_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.repetition_penalty = repetition_penalty
        generation_config.no_repeat_ngram_size = no_repeat_ngram_size
        generation_config.diversity_penalty = diversity_penalty
        generation_config.num_beams = 1
        generation_config.num_beam_groups = 1
        generation_config.seed = None
        
        if stopping_criteria is not None:
            generation_config.stopping_criteria = stopping_criteria
        
        if num_return_sequences > 1:
            # Use beam search if we want more than one sequence
            assert num_beams is not None, "num_beams must be set if num_return_sequences > 1"
            assert num_beams % num_return_sequences == 0, "num_beams must be divisible by num_return_sequences"
            assert num_beam_groups is not None, "num_beam_groups must be set if num_return_sequences > 1"
            assert num_beam_groups >= 1, "num_beam_groups must be divisible by num_return_sequences"
        
            generation_config.num_beams = num_beams
            generation_config.num_beam_groups = num_beam_groups
            generation_config.do_sample = False

        if generation_config.do_sample and seed:
            generation_config.seed = seed
        
        
        encoding = self.tokenizer(message, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                generation_config=generation_config
            )
            texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            texts = [t[len(message):] for t in texts]
            return texts
    