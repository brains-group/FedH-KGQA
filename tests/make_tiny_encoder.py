"""Build a tiny random RoBERTa-style encoder and tokenizer for offline smoke tests.

The smoke test uses it to exercise the Hugging Face + LoRA code path without a download.
  python tests/make_tiny_encoder.py --data-dir data/processed/toy/2hop/C1_rho0_balanced_s0 --out data/tiny_encoder
"""
import argparse
import json
import os

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="data/tiny_encoder")
    a = ap.parse_args()
    texts = []
    with open(os.path.join(a.data_dir, "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["question_masked"])
    specials = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=specials))
    from tokenizers.processors import TemplateProcessing
    tok.post_processor = TemplateProcessing(single="<s> $A </s>",
                                            special_tokens=[("<s>", tok.token_to_id("<s>")),
                                                            ("</s>", tok.token_to_id("</s>"))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                   unk_token="<unk>", mask_token="<mask>", cls_token="<s>", sep_token="</s>")
    os.makedirs(a.out, exist_ok=True)
    fast.save_pretrained(a.out)
    cfg = RobertaConfig(vocab_size=fast.vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=128, max_position_embeddings=130, pad_token_id=fast.pad_token_id)
    RobertaModel(cfg).save_pretrained(a.out)
    print(f"tiny encoder ({fast.vocab_size} tokens) written to {a.out}")


if __name__ == "__main__":
    main()
