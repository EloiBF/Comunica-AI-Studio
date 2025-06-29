import json
import os, sys
from pathlib import Path
# Definim entorn on s'executarà aquest script (com si fos el root)
base_dir = Path(__file__).resolve().parent.parent.parent  # Aquí, puja un nivell més alt per arribar a l'arrel
sys.path.append(str(base_dir))

# Importem funcions necessàries
from app.scripts.utils import prompt_AI


def generar_prompt_clusters(cluster_summary: dict) -> str:
    prompt = (
        "Tens la següent informació resumida de clústers de clients. "
        "Per a cada clúster, crea un nom breu i una descripció útil per màrqueting. "
        "El nom hauria de capturar l’essència del segment (per ex., 'Famílies Estalviadores'), "
        "i la descripció ha d’explicar què caracteritza el segment. "
        "Respon en format JSON com una llista de clústers amb els camps 'cluster', 'nom' i 'descripció'.\n\n"
        "Aquí tens les dades:\n"
    )
    prompt += json.dumps(cluster_summary, indent=2, ensure_ascii=False)
    prompt += "\n\nRespon només amb el JSON, si us plau."
    return prompt

def interpretar_clusters_amb_ai(cluster_summary: dict) -> dict:
    prompt = generar_prompt_clusters(cluster_summary)
    resposta = prompt_AI(prompt)
    try:
        return json.loads(resposta)
    except json.JSONDecodeError:
        print("⚠️ No s'ha pogut parsejar la resposta de l'AI. Resposta bruta:")
        print(resposta)
        return {}
