# generate_file_context.py

import os
import json
import groq
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


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

EXCLUSION = {".venv", ".env", "requirements.txt", "__pycache__", ".git", ".DS_Store"}

def is_valid_file(path: str) -> bool:
    for excl in EXCLUSION:
        if excl in path:
            return False
    return os.path.isfile(path)

def read_files_and_describe(folder: str) -> list:
    file_descriptions = []

    for root, _, files in os.walk(folder):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, folder)

            if not is_valid_file(relative_path):
                continue

            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"⚠️ Could not read {relative_path}: {e}")
                continue

            prompt = f"Give a concise description (1-3 sentences) of what this file does:\n\n{content[:5000]}"
            description = prompt_AI(prompt)  # Calls your AI description function

            file_descriptions.append({
                "file": relative_path,
                "description": description.strip(),
                "char_count": len(content)
            })

    return file_descriptions

def save_context_as_json(folder: str, output_path: str = "file_context.json"):
    result = read_files_and_describe(folder)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"📄 Context saved to {output_path} with {len(result)} files.")

if __name__ == "__main__":
    save_context_as_json(".")
