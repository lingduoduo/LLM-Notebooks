import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# --- Configuration: use an ultra-small model to fit M1 CPU ---
MODEL_NAME = "prajjwal1/bert-tiny"
NUM_LABELS = 2
MAX_LENGTH = 32
ADAPTER_DIM = 8
DEVICE = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class AdapterLayer(nn.Module):
    """A lightweight bottleneck adapter with a residual connection."""

    def __init__(self, input_dim: int, adapter_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapter_output = self.up_project(self.activation(self.down_project(x)))
        return x + self.dropout(adapter_output)


class ModelWithAdapter(nn.Module):
    """Frozen classifier model with trainable adapters after each encoder layer."""

    def __init__(self, base_model: AutoModelForSequenceClassification, adapter_dim: int = 16) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.loss_fct = nn.CrossEntropyLoss()

        for param in self.base_model.parameters():
            param.requires_grad = False

        hidden_size = base_model.config.hidden_size
        num_layers = base_model.config.num_hidden_layers
        self.adapters = nn.ModuleList(AdapterLayer(hidden_size, adapter_dim) for _ in range(num_layers))

        print("✓ Base model parameters frozen")
        print(f"✓ Added {len(self.adapters)} Adapter layers")
        print(f"✓ Params per Adapter layer: {adapter_dim * hidden_size * 2 + adapter_dim + hidden_size}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: dict,
    ) -> SequenceClassifierOutput:
        bert_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if token_type_ids is not None:
            bert_inputs["token_type_ids"] = token_type_ids

        outputs = self.base_model.bert(**bert_inputs)

        last_hidden_state = None
        for hidden_state, adapter in zip(outputs.hidden_states[1:], self.adapters):
            last_hidden_state = adapter(hidden_state)

        if last_hidden_state is None:
            raise RuntimeError("No hidden states were returned from the base model.")

        pooled_output = self.base_model.dropout(last_hidden_state[:, 0, :])
        logits = self.base_model.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


def build_dataset() -> Dataset:
    data = {
        "text": ["Very useful", "Terrible", "Not bad", "Awful", "Excellent", "Very disappointed"],
        "labels": [1, 0, 1, 0, 1, 0],
    }
    return Dataset.from_dict(data)


def tokenize_function(examples: dict) -> dict:
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def build_trainer(model: nn.Module, train_dataset: Dataset) -> Trainer:
    training_args = TrainingArguments(
        output_dir="./adapter_output",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        learning_rate=1e-3,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        use_cpu=True,
    )
    return Trainer(model=model, args=training_args, train_dataset=train_dataset)


def run_inference(model: nn.Module, texts: list[str]) -> None:
    model.eval()
    model.to(DEVICE)

    print("\n=== Adapter inference demo ===")
    print("What to watch: how the trained Adapter changes the model's predictions")

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            prediction = probabilities.argmax(dim=-1).item()
            confidence = probabilities.max().item()

        result = "Positive" if prediction == 1 else "Negative"
        print(f"Text: '{text}' -> Prediction: {result} (confidence: {confidence:.3f})")


def main() -> None:
    print("=== Core differences: Adapter fine-tuning vs LoRA fine-tuning ===")
    print("1. Adapter: insert small neural network modules between model layers")
    print("2. LoRA: modify existing weight matrices via low-rank matrix decomposition")
    print("3. Adapter adds new layers; LoRA changes the weights of existing layers")

    dataset = build_dataset()
    tokenized_dataset = dataset.map(tokenize_function, batched=True, desc="Tokenizing dataset").remove_columns(["text"])
    tokenized_dataset.set_format("torch")

    print("\n=== Creating Adapter model ===")
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    adapter_model = ModelWithAdapter(base_model, adapter_dim=ADAPTER_DIM)
    adapter_model.to(DEVICE)

    total_params, trainable_params = count_parameters(adapter_model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    print("Key observation: training only a tiny fraction of parameters is the core advantage of Adapter fine-tuning.")

    trainer = build_trainer(adapter_model, tokenized_dataset)

    print("\n=== Start Adapter fine-tuning ===")
    print("Core idea: only the inserted Adapter module parameters are trained; the base model stays frozen.")
    trainer.train()

    run_inference(adapter_model, ["Excellent", "Very disappointed", "It's okay"])


if __name__ == "__main__":
    main()
