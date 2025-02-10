from datasets import load_dataset
import json
import random
import re

def parse_recipe_text(text):
    """Parse le texte brut d'une recette pour extraire les composants."""
    # Trouver le titre (tout ce qui est avant "Ingredients:")
    title_match = re.match(r"(.*?)\s*Ingredients:", text)
    title = title_match.group(1).strip() if title_match else "Recette"
    
    # Séparer les sections principales
    parts = text.split("Ingredients:")
    if len(parts) > 1:
        ingredients_and_rest = parts[1].split("Directions:")
        ingredients_text = ingredients_and_rest[0].strip()
        directions_text = ingredients_and_rest[1].strip() if len(ingredients_and_rest) > 1 else ""
    else:
        ingredients_text = ""
        directions_text = ""
    
    # Nettoyer les ingrédients
    ingredients = [ing.strip().strip('- ') for ing in ingredients_text.split('\n') if ing.strip()]
    
    # Nettoyer les instructions
    directions = [step.strip().strip('- ') for step in directions_text.split('\n') if step.strip()]
    
    # Extraire le nombre de portions s'il existe
    servings = ""
    if directions and "Serves" in directions[-1]:
        servings = directions[-1]
        directions = directions[:-1]
    
    return {
        "title": title,
        "ingredients": ingredients,
        "directions": directions,
        "servings": servings
    }

def generate_instruction(ingredients):
    """Génère une instruction interactive basée sur les ingrédients."""
    # Liste d'ingrédients courants pour créer des variations
    common_ingredients = ["tomates", "oignons", "ail", "poivrons", "carottes", "pommes de terre",
                         "riz", "pâtes", "poulet", "boeuf", "poisson", "lentilles", "haricots"]
    allergens = ["cacahuètes", "fruits de mer", "gluten", "lactose", "soja", "fruits à coque",
                 "oeufs", "moutarde", "sésame"]
    
    # Sélectionner aléatoirement 2-3 ingrédients souhaités
    wanted = random.sample(common_ingredients, random.randint(2, 3))
    # Sélectionner 1-2 allergènes à éviter
    unwanted = random.sample(allergens, random.randint(1, 2))
    
    return f"Propose moi une recette avec {', '.join(wanted[:-1])} et {wanted[-1]}, elle ne doit pas contenir de {' ni de '.join(unwanted)}"

def load_and_convert_recipes():
    # Charger le dataset
    print("Chargement du dataset...")
    dataset = load_dataset("corbt/all-recipes")
    
    # Créer la liste pour stocker les données converties
    converted_data = []
    
    # Parcourir les recettes
    print("Conversion des recettes...")
    for recipe in dataset['train']:
        # Parser le texte de la recette
        parsed_recipe = parse_recipe_text(recipe['input'])
        
        # Créer l'instruction interactive
        instruction = generate_instruction(parsed_recipe['ingredients'])
        
        # Formater la réponse
        ingredients_list = "\n".join([f"- {ing}" for ing in parsed_recipe['ingredients']])
        steps_list = "\n".join([f"{i+1}. {step}" for i, step in enumerate(parsed_recipe['directions'])])
        
        response = f"""Voici une recette de {parsed_recipe['title']} :

Ingrédients nécessaires :
{ingredients_list}

Étapes de préparation :
{steps_list}

{parsed_recipe['servings']}"""
        
        # Créer l'entrée au format attendu par LM Studio
        entry = {
            "instruction": instruction,
            "response": response
        }
        
        converted_data.append(entry)
        
        # Afficher un exemple pour vérification (première recette uniquement)
        if len(converted_data) == 1:
            print("\nExemple de conversion:")
            print("------------------------")
            print(json.dumps(entry, ensure_ascii=False, indent=2))
            print("------------------------\n")
    
    # Limiter le nombre de recettes pour les tests initiaux
    converted_data = converted_data[:10000]
    
    # Mélanger les données
    random.shuffle(converted_data)
    
    # Diviser en train/validation (90/10)
    split_index = int(len(converted_data) * 0.9)
    train_data = converted_data[:split_index]
    val_data = converted_data[split_index:]
    
    print(f"Nombre de recettes converties:")
    print(f"- Training: {len(train_data)}")
    print(f"- Validation: {len(val_data)}")
    
    # Sauvegarder les fichiers
    print("\nSauvegarde des fichiers...")
    with open('recipes_train.jsonl', 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    with open('recipes_val.jsonl', 'w', encoding='utf-8') as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print("\nFichiers créés avec succès:")
    print("- recipes_train.jsonl")
    print("- recipes_val.jsonl")

if __name__ == "__main__":
    load_and_convert_recipes()
