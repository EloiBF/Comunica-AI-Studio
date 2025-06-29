import json
import re
import json
from groq import Groq
from dotenv import load_dotenv
import os

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