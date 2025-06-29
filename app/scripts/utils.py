import json
import re
import json
from groq import Groq
from dotenv import load_dotenv
import os, sys
from pathlib import Path

# Definim entorn on s'executarà aquest script (com si fos el root)
base_dir = Path(__file__).resolve().parent.parent.parent  # Aquí, puja un nivell més alt per arribar a l'arrel
sys.path.append(str(base_dir))

from django.conf import settings




def get_user_folder_path(username, subfolder='data'):
    base_path = os.path.join(settings.BASE_DIR, 'users', username, subfolder)
    os.makedirs(base_path, exist_ok=True)
    return base_path

def get_user_db_path(username):
    return os.path.join(get_user_folder_path(username, 'data'), 'user_database.db')

def get_clean_table_name(file_path):
    name = os.path.splitext(os.path.basename(file_path))[0]
    return name.strip().replace(' ', '_').replace('-', '_').lower()

def extract_json_from_text(text: str) -> dict:
    """
    Intenta extreure i deserialitzar el primer bloc JSON vàlid d'un text (possiblement brut) retornat per un LLM.
    """

    # 🧠 Busca el primer bloc entre claus { ... }
    matches = re.finditer(r"\{.*?\}", text, re.DOTALL)

    for match in matches:
        try:
            raw_json = match.group()
            return json.loads(raw_json)
        except json.JSONDecodeError:
            continue  # Prova amb el següent match

    raise ValueError("No s'ha pogut extreure un bloc JSON vàlid del text.")

def prompt_AI(prompt: str, model: str = "llama3-70b-8192") -> str:
    """
    Fa una crida a Groq amb el prompt indicat i retorna el contingut de la resposta com a string.
    """

    # Carreguem el .env
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Puja 3 nivells
    dotenv_path = os.path.join(base_dir, '.env')
    
    # Carregar el .env
    load_dotenv(dotenv_path)
    
    # Comprovem si la clau està carregada
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise RuntimeError("Falta la clau API de Groq (GROQ_API_KEY) al fitxer .env")
    else:
        print("Clau API carregada correctament.")  # Depuració per veure que està carregada

    # Groq client
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    prompt_AI("Genera contingut per una oferta especial de llançament d'un nou producte tecnològic.")