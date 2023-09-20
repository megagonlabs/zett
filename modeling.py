from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from fire import Fire
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          IntervalStrategy, TrainingArguments,
                          set_seed, PreTrainedModel, PreTrainedTokenizerFast)

from transformer_base import run_summarization
from utils import DynamicModel


class TextGenerator(DynamicModel):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerFast
    scores: Optional[List[Tensor]] = None
    max_length: int

    def tokenize(self, texts: List[str], **kwargs):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs,
        ).to(self.model.device)

    def run(
        self,
        texts: List[str],
        do_sample=False,
        top_k=None,
        temperature=1.0,
        num_return: int = 4,
        prompt: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None,
        multi_prompt_ids: Optional[List[List[int]]] = None,
        decoder_input_ids: Optional[Tensor] = None,
        save_scores: bool = False,
        **kwargs,
    ) -> List[str]:
        # https://huggingface.co/transformers/v4.7.0/main_classes/model.html#generation
        tok = self.tokenizer
        eos, bos = tok.eos_token_id, tok.bos_token_id

        if prompt is not None:
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        if prompt_ids is not None:
            prompt_ids = [eos, bos] + prompt_ids
            decoder_input_ids = torch.tensor([prompt_ids])
        if multi_prompt_ids is not None:
            assert len(texts) == len(multi_prompt_ids)
            multi_prompt_ids = [[eos, bos] + lst for lst in multi_prompt_ids]
            decoder_input_ids = torch.tensor(multi_prompt_ids)
        if decoder_input_ids is not None:
            kwargs.update(decoder_input_ids=decoder_input_ids.to(self.model.device))

        outputs = self.model.generate(
            **self.tokenize(texts),
            do_sample=do_sample,
            top_k=top_k,
            temperature=temperature,
            num_return_sequences=num_return,
            return_dict_in_generate=True,
            output_scores=save_scores,
            max_length=self.max_length,
            **kwargs,
        )
        
        return outputs.scores, self.decode(outputs.sequences)

    def decode(self, outputs) -> List[str]:
        tok = self.tokenizer
        texts = tok.batch_decode(
            outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        # Manually remove <bos><eos><pad> in case we have custom special tokens
        # special_tokens = [tok.eos_token, tok.bos_token, tok.pad_token] # for T5 case, this is ['</s>', None, '<pad>']
        special_tokens = [tok.eos_token, tok.unk_token, tok.pad_token]
        for i, t in enumerate(texts):
            for token in special_tokens:
                t = t.replace(token, "")
                texts[i] = t
        return texts


class RelationModel(DynamicModel):
    model_dir: str
    data_dir: str
    model_name: str
    do_pretrain: bool
    batch_size: int = 64
    grad_accumulation: int = 2
    random_seed: int = 42
    warmup_ratio: float = 0.2
    lr_pretrain: float = 3e-5
    lr_finetune: float = 3e-5
    epochs_pretrain: int = 3
    epochs_finetune: int = 5
    train_fp16: bool = True

    def fit(self, path_train: str, path_dev: Optional[str] = None):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def get_lr(self) -> float:
        return self.lr_pretrain if self.do_pretrain else self.lr_finetune

    def get_epochs(self) -> int:
        return self.epochs_pretrain if self.do_pretrain else self.epochs_finetune

    def get_train_args(self, do_eval: bool) -> TrainingArguments:
        return TrainingArguments(
            seed=self.random_seed,
            do_train=True,
            do_eval=do_eval or None,
            overwrite_output_dir=True,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accumulation,
            warmup_ratio=self.warmup_ratio,
            output_dir=self.model_dir,
            save_strategy=IntervalStrategy.EPOCH,
            save_total_limit=3,
            evaluation_strategy=IntervalStrategy.EPOCH
            if do_eval
            else IntervalStrategy.NO,
            learning_rate=self.get_lr(),
            num_train_epochs=self.get_epochs(),
            load_best_model_at_end=True,
            fp16=self.train_fp16,
        )


class ZETTTripletExtractor(RelationModel):
    model_name: str = "t5-base"
    max_source_length: int = 128
    max_target_length: int = 64

    def fit(self, path_train: str, path_dev: Optional[str] = None):
        kwargs = {}

        data_args = run_summarization.DataTrainingArguments(
            train_file=path_train,
            validation_file=path_dev,
            overwrite_cache=True,
            max_target_length=self.max_target_length,
            max_source_length=self.max_source_length,
            **kwargs,
        )
        train_args = self.get_train_args(do_eval=path_dev is not None)
        #   -> per_device_train_batch_size=self.batch_size,
        #   -> gradient_accumulation_steps=self.grad_accumulation,
        kwargs = {
            k: v for k, v in train_args.to_dict().items() if not k.startswith("_")
        }

        train_args = run_summarization.Seq2SeqTrainingArguments(**kwargs)
        model_args = run_summarization.ModelArguments(
            model_name_or_path=self.model_name
        )
        run_summarization.main(
            model_args=model_args, training_args=train_args, data_args=data_args
        )

    def load_generator(self, device: torch.device) -> TextGenerator:
        gen = TextGenerator(
            model=AutoModelForSeq2SeqLM.from_pretrained(self.model_dir),
            tokenizer=AutoTokenizer.from_pretrained(self.model_dir),
            max_length=self.max_target_length,
        )
        gen.model = gen.model.to(device)
        return gen


if __name__ == "__main__":
    Fire()
