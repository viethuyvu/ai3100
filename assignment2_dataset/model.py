from datasets import load_dataset, concatenate_datasets
from transformers import pipeline
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load Pokemon dataset test split
pokemon_dataset = load_dataset("JJMack/pokemon-classification-gen1-9")
pokemon_test = pokemon_dataset["test"]
pokemon_test = pokemon_test.remove_columns([c for c in pokemon_test.column_names if c != "image_data"])
pokemon_test = pokemon_test.add_column("binary_label", [1] * len(pokemon_test))
# Resize and convert all images to RGB
def preprocess_pokemon(example):
    img = example["image_data"].convert("RGB")
    img = img.resize((224, 224))
    example["image_data"] = img
    return example

pokemon_test = pokemon_test.map(preprocess_pokemon)
# print("Pokemon test set:", pokemon_test)

# Load Digimon dataset from local directory
digimon_dataset = load_dataset("imagefolder", data_dir="./not_pokemon", split="train")
# Resize and convert all images to RGB
def preprocess_digimon(example):
    img = example["image"].convert("RGB")
    img = img.resize((224, 224))
    example["image_data"] = img
    return example

digimon = digimon_dataset.map(preprocess_digimon)
# print("Original Digimon dataset:", digimon)

# Keep only 'image_data' and add binary label = 0
digimon = digimon.remove_columns([c for c in digimon.column_names if c not in ["image_data"]])
digimon = digimon.add_column("binary_label", [0] * len(digimon))

# print("Simplified Digimon dataset:", digimon)

combined_test = concatenate_datasets([pokemon_test, digimon]).shuffle(seed=13)
# print("Combined test set:", combined_test)

# Load model and create pipeline
pipe = pipeline("image-classification", model="skshmjn/Pokemon-classifier-gen9-1025", device=0,batch_size=16)

# Predict function
def predict_batch(batch, threshold=0.4):
    preds = pipe(batch["image_data"])
    results = []
    for pred in preds:
        best = max(pred, key=lambda x: x['score'])
        if best['score'] >= threshold:
            results.append({"pred_label":1,"pred_name":"Pokemon","pred_score":best['score']})
        else:
            results.append({"pred_label":0,"pred_name":"Not Pokemon","pred_score":best['score']})
    return {k:[dic[k] for dic in results] for k in results[0]}

results = combined_test.map(predict_batch, batched=True, batch_size=16)

# Show metrics
y_true = results["binary_label"]
y_pred = results["pred_label"]

print("\n====Classification Report====")
print(classification_report(y_true, y_pred, target_names=["Not Pokemon", "Pokemon"],digits=4))

print("====Confusion Matrix====")
cm = confusion_matrix(y_true, y_pred)
print(cm)

sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
print(f"Specificity (True Negative Rate): {specificity:.4f}")