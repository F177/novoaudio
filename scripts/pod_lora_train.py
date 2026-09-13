"""Fine-tuning LoRA do MOSS-TTS-v1.5 pra pt-BR, rodando no Pod (venv "tts").

Baseado em duas fontes reais do repositório oficial (`OpenMOSS/MOSS-TTS`,
clonado em `/root/moss-tts-repo` no Pod pra esta investigação):

- `moss_tts_delay/finetuning/{dataset.py,common.py}` — carregamento de dado
  e empacotamento teacher-forcing (`MossTTSSFTDataset`), reaproveitado
  direto porque já validamos que funciona com nosso `train_with_codes.jsonl`
  (rodou até o backward() num teste de fine-tuning completo, que só faltou
  por OOM — confirmando que a parte de dado está certa, só o fine-tuning
  completo não cabe em 24GB).
- `community/norwegian-lora/train_lora.py` — a lógica de aplicar LoRA
  (via `peft`) num `MossTTSDelayModel`, incluindo dois workarounds reais
  que doeria descobrir sozinho: `get_input_embeddings()` do modelo exige
  `input_ids` mas o `peft` chama sem argumento nenhum durante o setup; e
  `forward()` não aceita `output_hidden_states` nomeado, mas o `peft`
  manda esse kwarg mesmo assim. Esse script usava uma versão antiga de
  `dataset.py` (classe `MossTTSDataset`, já renomeada/reestruturada pra
  `MossTTSSFTDataset`) — não dava pra importar direto, então a parte de
  dado foi reescrita aqui usando a API atual, mantendo os workarounds de
  LoRA que continuam válidos (mesma classe de modelo).

Fine-tuning completo (via `moss_tts_delay/finetuning/sft.py` oficial) deu
CUDA OOM até com `--per-device-batch-size 1 --gradient-checkpointing` na
GPU de 24GB do Pod — LoRA existe justamente pra caber: só uns poucos
milhões de parâmetros treináveis, não os 8B inteiros.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, get_scheduler

MOSS_TTS_REPO = "/root/moss-tts-repo"
if MOSS_TTS_REPO not in sys.path:
    sys.path.insert(0, MOSS_TTS_REPO)

from moss_tts_delay.finetuning.common import load_jsonl, resolve_jsonl_paths  # noqa: E402
from moss_tts_delay.finetuning.dataset import MossTTSSFTDataset  # noqa: E402
from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel  # noqa: E402
from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor  # noqa: E402

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="OpenMOSS-Team/MOSS-TTS-v1.5")
    parser.add_argument("--codec-path", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--n-vq", type=int, default=None)
    parser.add_argument("--channelwise-loss-weight", default="1,32")
    parser.add_argument("--resume-adapter-path", default=None)
    return parser.parse_args()


def parse_channelwise_loss_weight(spec: str, n_heads: int) -> list[float]:
    values = [float(v) for v in spec.split(",") if v.strip()]
    if len(values) == n_heads:
        return values
    if len(values) == 2:
        text_weight, total_audio_weight = values
        per_audio = total_audio_weight / max(1, n_heads - 1)
        return [text_weight] + [per_audio] * (n_heads - 1)
    raise ValueError(f"channelwise_loss_weight precisa de {n_heads} ou 2 valores")


def apply_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    """Aplica LoRA no backbone de linguagem do MossTTSDelayModel.

    Os dois monkey-patches abaixo são workarounds reais (achados no exemplo
    norueguês da comunidade) pra incompatibilidades entre a API do `peft`
    (feita pra `transformers` genérico) e as particularidades desta classe
    de modelo — sem eles, `get_peft_model` ou o forward duram quebram.
    """
    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # peft chama model.get_input_embeddings() sem argumento no setup, mas
    # o método real desta classe exige `input_ids`.
    original_get_input_embeddings = type(model).get_input_embeddings
    type(model).get_input_embeddings = lambda self, input_ids=None: (
        original_get_input_embeddings(self, input_ids)
        if input_ids is not None
        else self.language_model.get_input_embeddings()
    )

    # peft (versão mais nova que a testada no exemplo norueguês) exige que
    # o modelo base tenha `prepare_inputs_for_generation` só pra existir o
    # atributo — nunca é chamado de verdade aqui, só usamos forward() pra
    # treino, não o generate() customizado desta classe.
    if not hasattr(type(model), "prepare_inputs_for_generation"):
        type(model).prepare_inputs_for_generation = lambda self, *a, **k: {}

    if args.resume_adapter_path:
        model = PeftModel.from_pretrained(model, args.resume_adapter_path, is_trainable=True)
    else:
        model = get_peft_model(model, lora_config)

    # peft manda output_hidden_states/return_dict como kwarg mesmo o
    # forward() desta classe não aceitando esses nomes explicitamente.
    base_cls = type(model.get_base_model())
    original_forward = base_cls.forward

    def patched_forward(self, *fargs, output_hidden_states=None, return_dict=None, **fkwargs):
        return original_forward(self, *fargs, **fkwargs)

    base_cls.forward = patched_forward

    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    if not trainable:
        raise RuntimeError("Nenhum parâmetro LoRA treinável encontrado.")
    total = sum(trainable.values())
    print(f"[lora] {len(trainable)} tensores treináveis, {total:,} parâmetros")
    return model


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = MossTTSDelayProcessor(
        tokenizer=tokenizer, audio_tokenizer=None, model_config=config
    )

    paths = resolve_jsonl_paths(args.train_jsonl)
    records = [r for p in paths for r in load_jsonl(p)]
    if not records:
        raise ValueError(f"nenhum registro em {args.train_jsonl}")
    print(f"[data] {len(records)} exemplos de {len(paths)} arquivo(s)")

    dataset = MossTTSSFTDataset(records=records, processor=processor, n_vq=args.n_vq)

    model = MossTTSDelayModel.from_pretrained(
        args.model_path, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    model.gradient_checkpointing_enable()
    model = apply_lora(model, args)

    channelwise_loss_weight = parse_channelwise_loss_weight(
        args.channelwise_loss_weight, model.get_base_model().config.n_vq + 1
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate)

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or (args.num_epochs * steps_per_epoch)
    warmup_steps = max(1, math.ceil(max_train_steps * args.warmup_ratio))
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    micro_step = 0
    accum_loss = 0.0
    start = time.time()
    losses: list[float] = []

    for epoch in range(args.num_epochs):
        if global_step >= max_train_steps:
            break
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                channelwise_loss_weight=torch.tensor(
                    channelwise_loss_weight, dtype=torch.float32, device=device
                ),
            )
            loss = outputs.loss
            (loss / args.gradient_accumulation_steps).backward()
            accum_loss += loss.detach().float().item()
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            step_loss = accum_loss / args.gradient_accumulation_steps
            accum_loss = 0.0
            losses.append(step_loss)

            if global_step % args.logging_steps == 0:
                elapsed = time.time() - start
                print(
                    f"step={global_step}/{max_train_steps} epoch={epoch} "
                    f"loss={step_loss:.4f} lr={lr_scheduler.get_last_lr()[0]:.2e} "
                    f"elapsed={elapsed:.1f}s"
                )
                sys.stdout.flush()

            if global_step >= max_train_steps:
                break

    ckpt_dir = output_dir / "final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    (output_dir / "losses.json").write_text(json.dumps(losses), encoding="utf-8")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[done] peak_vram_allocated={peak:.2f}GB steps={global_step} losses={losses}")
    print(f"[done] adapter salvo em {ckpt_dir}")


if __name__ == "__main__":
    main()
