import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
import groq

# Carregar variables d'entorn
load_dotenv()

# Constants
DEFAULT_DB_NAME = 'user_database'
DATABASE_DIR = 'app/users/'

def prompt_AI(query):
    """
    Envia una consulta a l'API de Groq i retorna la resposta
    """
    try:
        # Obtenir clau API
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY no trobada a les variables d'entorn")

        # Crear client
        client = groq.Groq(api_key=api_key)

        # Fer la consulta
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": query,
                }
            ],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=2048,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"Error en prompt_AI: {e}")
        return f"Error: {str(e)}"

def get_clean_table_name(file_path):
    """
    Genera un nom de taula net a partir del nom del fitxer
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # Neteja el nom per fer-lo compatible amb SQL
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)
    # Assegurar que comença amb una lletra
    if clean_name and not clean_name[0].isalpha():
        clean_name = 'table_' + clean_name
    return clean_name or 'user_table'

def get_user_db_path(username, db_name=DEFAULT_DB_NAME):
    """
    Retorna el camí de la base de dades per l'usuari
    """
    user_dir = os.path.join(DATABASE_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, f'{db_name}.db')

def get_user_folder_path(username, subfolder=None):
    """
    Retorna el camí de la carpeta de l'usuari
    """
    user_dir = os.path.join(DATABASE_DIR, username)
    if subfolder:
        user_dir = os.path.join(user_dir, subfolder)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def extract_json_from_text(text: str) -> dict:
    """
    Intenta extreure i deserialitzar el primer bloc JSON vàlid d'un text (possiblement brut) retornat per un LLM.
    """
    # Busca el primer bloc entre claus { ... }
    matches = re.finditer(r"\{.*?\}", text, re.DOTALL)

    for match in matches:
        try:
            raw_json = match.group()
            return json.loads(raw_json)
        except json.JSONDecodeError:
            continue  # Prova amb el següent match

    raise ValueError("No s'ha pogut extreure un bloc JSON vàlid del text.")
