from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Vérifier si CUDA est disponible et définir le device sur GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Charger les données
dataset = load_dataset("json", data_files={"train": "dataset/recipes_train.jsonl", "validation": "dataset/recipes_val.jsonl"})

# Charger le modèle GPT-2 avec LoRA
model_name = "gpt2"  # Modèle plus léger que Mistral-7B
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Définir le pad_token à l'eos_token pour résoudre l'erreur de padding
tokenizer.pad_token = tokenizer.eos_token

# Charger le modèle GPT-2 et l'envoyer sur le GPU
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Configuration LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05
)

# Appliquer LoRA au modèle et l'envoyer sur GPU
model = get_peft_model(model, lora_config).to(device)

# Fonction de prétraitement des données
def preprocess_function(examples):
    inputs = [instruction + "\n" + response for instruction, response in zip(examples["instruction"], examples["response"])]

    # Tokenisation avec ajout des labels pour le calcul de la perte
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

    # Envoyer les tenseurs sur GPU
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    model_inputs["labels"] = model_inputs["input_ids"].clone()

    return model_inputs

# Tokeniser les données d'entraînement et de validation
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Configuration de l'entraînement avec support du GPU et du mixed precision (fp16)
training_args = TrainingArguments(
    output_dir="./gpt2_recipes",  # Répertoire de sortie
    per_device_train_batch_size=4,  # Taille de batch d'entraînement
    per_device_eval_batch_size=4,  # Taille de batch d'évaluation
    eval_strategy="epoch",  # Évaluation après chaque époque
    save_strategy="epoch",  # Sauvegarde après chaque époque
    logging_dir="./logs",  # Répertoire des logs
    learning_rate=1e-4,  # Taux d'apprentissage
    num_train_epochs=3,  # Nombre d'époques
    weight_decay=0.01,  # Décroissance de poids
    push_to_hub=False,  # Ne pas pousser sur le hub de Hugging Face
    gradient_accumulation_steps=4,  # Accumulation des gradients pour gérer la mémoire
    logging_steps=500,  # Intervalle des étapes de logging
    save_total_limit=2,  # Limite du nombre de checkpoints à conserver
    evaluation_strategy="steps",  # Évaluation par étapes
    eval_steps=500,  # Nombre d'étapes entre chaque évaluation
    fp16=True,  # Active le mixed precision pour accélérer l'entraînement sur GPU
    dataloader_pin_memory=True,  # Optimisation mémoire pour GPU
    report_to="none",  # Désactiver les logs vers Weights & Biases
)

# Créer l'objet Trainer pour l'entraînement
trainer = Trainer(
    model=model,  # Modèle à entraîner
    args=training_args,  # Arguments d'entraînement
    train_dataset=tokenized_datasets["train"],  # Dataset d'entraînement
    eval_dataset=tokenized_datasets["validation"],  # Dataset de validation
    tokenizer=tokenizer,  # Tokenizer utilisé pour la prétraitement
)

# Démarrer l'entraînement
trainer.train()

# Sauvegarder le modèle fine-tuné
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
