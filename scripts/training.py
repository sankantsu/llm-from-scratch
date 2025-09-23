import argparse
import dataclasses
import logging
import math
import os
import matplotlib.pyplot as plt
import tiktoken
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from llm_from_scratch.dataset import create_dataloader_v1, get_verdict_txt
from llm_from_scratch.generate_text import (
    text_to_token_ids,
    token_ids_to_text,
    generate_text_simple,
)
from llm_from_scratch.gpt_config import GPT_CONFIG_124M
from llm_from_scratch.gpt_model import GPTModel


@dataclasses.dataclass
class TrainingArgs:
    peak_lr: float
    initial_lr: float
    min_lr: float
    weight_decay: float
    warmup_ratio: float
    num_epochs: int
    batch_size: int
    eval_freq: int
    eval_iter: int
    eval_context: str


@dataclasses.dataclass
class TrainResult:
    n_epoch: int
    train_losses: list[float]
    validation_losses: list[float]
    tokens_seen: list[int]


def get_verdict_txt_cached() -> str:
    file_path = "the-verdict.txt"

    # Cache on file_path
    if not os.path.exists(file_path):
        text_data = get_verdict_txt()
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    return text_data


def calc_loss_loader(
    data_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(input_batch)
            loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model_simple(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: tiktoken.Encoding,
    training_args: TrainingArgs,
) -> TrainResult:
    num_epochs = training_args.num_epochs
    eval_freq = training_args.eval_freq
    eval_iter = training_args.eval_iter
    total_training_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_training_steps * training_args.warmup_ratio)
    peak_lr = training_args.peak_lr
    initial_lr = training_args.initial_lr
    min_lr = training_args.min_lr
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    device = next(model.parameters()).device
    global_step = 0
    tokens_seen = 0
    train_losses, val_losses, track_tokens_seen = [], [], []
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()

            # Adjust the learning rate based on the current phase (warmup or cosine annealing)
            if global_step < warmup_steps:
                # Linear warmup
                lr = initial_lr + global_step * lr_increment
            else:
                # Cosine annealing after warmup
                progress = ((global_step - warmup_steps) /
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            # Apply the calculated learning rate to the optimizer
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            logits = model(input_batch)
            loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
            loss.backward()

            # Gradient clipping
            if global_step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            tokens_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}",
                    flush=True,
                )
            global_step += 1

        # Print generated text sample
        model.eval()
        start_context = training_args.eval_context
        context_size = model.pos_emb.weight.shape[0]
        tokens = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            token_ids = generate_text_simple(
                model, tokens, max_new_tokens=50, context_size=context_size
            )
            decoded_text = token_ids_to_text(token_ids, tokenizer)
            decoded_text = decoded_text.replace("\n", r"\n")
            print(
                f"Sample text generation (Epoch {epoch+1}, Context: {start_context}):"
            )
            print(decoded_text)
    return TrainResult(
        n_epoch=num_epochs,
        train_losses=train_losses,
        validation_losses=val_losses,
        tokens_seen=track_tokens_seen,
    )


def plot_losses(train_result: TrainResult, filename: str = "loss_curve.png") -> None:
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    n_epochs = train_result.n_epoch
    train_losses = train_result.train_losses
    val_losses = train_result.validation_losses
    tokens_seen = train_result.tokens_seen
    epochs_seen = torch.linspace(0, n_epochs, len(train_losses))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig(filename)


def main() -> None:
    logging.basicConfig(level="INFO")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--device", default="cpu", help="Device to use for training."
    )
    args = parser.parse_args()

    torch.manual_seed(123)

    device = torch.device(args.device)
    logging.info(f"Using device {device}")

    text_data = get_verdict_txt_cached()

    # Setup model and optimizer
    gpt_config = GPT_CONFIG_124M.copy()
    gpt_config["context_length"] = 256  # Shortened context length (orig: 1024)
    model = GPTModel(gpt_config)
    model.to(device)

    default_context = "Every effort moves you"
    training_args = TrainingArgs(
        peak_lr=0.001,
        initial_lr=1e-5,
        min_lr=1e-5,
        warmup_ratio=0.2,
        num_epochs=10,
        batch_size=2,
        weight_decay=0.1,
        eval_freq=5,
        eval_iter=1,
        eval_context=default_context,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.peak_lr,
        weight_decay=training_args.weight_decay,
    )

    # Setup data loader
    train_ratio = 0.9
    split_idx = int(len(text_data) * train_ratio)
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=training_args.batch_size,
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
    )
    validation_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=training_args.batch_size,
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
    )

    # Train model
    tokenizer = tiktoken.get_encoding("gpt2")
    train_result = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        optimizer=optimizer,
        tokenizer=tokenizer,
        training_args=training_args,
    )
    plot_losses(train_result)


if __name__ == "__main__":
    main()
