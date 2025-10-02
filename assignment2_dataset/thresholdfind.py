from datasets import load_dataset, concatenate_datasets
from transformers import pipeline
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# ------------------------
# 1. Load Pokémon test set
# ------------------------
pokemon_dataset = load_dataset("JJMack/pokemon-classification-gen1-9")
pokemon_test = pokemon_dataset["test"]
pokemon_test = pokemon_test.remove_columns([c for c in pokemon_test.column_names if c != "image_data"])
pokemon_test = pokemon_test.add_column("binary_label", [1] * len(pokemon_test))

# Resize Pokémon images
def preprocess_pokemon(example):
    img = example["image_data"].convert("RGB").resize((224, 224))
    example["image_data"] = img
    return example

pokemon_test = pokemon_test.map(preprocess_pokemon)

# ------------------------
# 2. Load Digimon dataset
# ------------------------
digimon_dataset = load_dataset("imagefolder", data_dir="./not_pokemon", split="train")

def preprocess_digimon(example):
    img = example["image"].convert("RGB").resize((224, 224))
    example["image_data"] = img
    return example

digimon = digimon_dataset.map(preprocess_digimon)
digimon = digimon.remove_columns([c for c in digimon.column_names if c not in ["image_data"]])
digimon = digimon.add_column("binary_label", [0] * len(digimon))

# ------------------------
# 3. Combine & shuffle
# ------------------------
combined_test = concatenate_datasets([pokemon_test, digimon]).shuffle(seed=42)

# ------------------------
# 4. Load model (GPU + batch)
# ------------------------
pipe = pipeline(
    "image-classification",
    model="skshmjn/Pokemon-classifier-gen9-1025",
    device=0,       # GPU
    batch_size=16
)

# ------------------------
# 5. Run model & collect scores
# ------------------------
all_scores = []
all_labels = combined_test["binary_label"]

# Using batched processing
batch_size = 16
for i in range(0, len(combined_test), batch_size):
    batch_imgs = combined_test["image_data"][i:i+batch_size]
    preds = pipe(batch_imgs)
    # extract best score per image
    for pred in preds:
        best = max(pred, key=lambda x: x['score'])
        all_scores.append(best['score'])

all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

# ------------------------
# 6. Sweep thresholds
# ------------------------
thresholds = np.arange(0.1, 1.00, 0.05)
results = []

for t in thresholds:
    y_pred = (all_scores >= t).astype(int)
    
    acc = accuracy_score(all_labels, y_pred)
    prec = precision_score(all_labels, y_pred)
    rec = recall_score(all_labels, y_pred)
    f1 = f1_score(all_labels, y_pred)
    
    # confusion matrix
    cm = confusion_matrix(all_labels, y_pred)
    sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    
    results.append({
        "threshold": t,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity
    })

# ------------------------
# 7. Print results
# ------------------------
print("Threshold sweep results:")
for r in results:
    print(f"t={r['threshold']:.2f} | Acc={r['accuracy']:.4f} | Prec={r['precision']:.4f} | Recall={r['recall']:.4f} | F1={r['f1']:.4f} | Sens={r['sensitivity']:.4f} | Spec={r['specificity']:.4f}")

# ------------------------
# 8. Pick best threshold
# ------------------------
# Example: pick threshold that maximizes F1
best = max(results, key=lambda x: x["f1"])
print(f"\nBest threshold by F1: {best['threshold']:.2f} with F1={best['f1']:.4f}")
