import json
import os, sys
from pathlib import Path
import re

# Definim entorn on s'executarà aquest script (com si fos el root)
base_dir = Path(__file__).resolve().parent.parent.parent  # Aquí, puja un nivell més alt per arribar a l'arrel
sys.path.append(str(base_dir))

# Importem funcions necessàries
from app.scripts.utils import prompt_AI, get_user_folder_path


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


# Configurar path
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))

from app.scripts.utils import prompt_AI

def interpretar_clusters_amb_ai(cluster_summary: dict) -> dict:
    """
    Genera descripcions dels clusters utilitzant IA
    """
    prompt = f"""
Ets un expert en segmentació de clients i análisis de dades.

Analitza el següent resum de clusters i genera una descripció clara i útil per cada cluster:

{json.dumps(cluster_summary, indent=2, ensure_ascii=False)}

Per cada cluster, proporciona:
1. Un nom descriptiu del segment
2. Una descripció breu dels trets principals
3. Recomanacions d'acció o estratègia

Retorna la resposta en format JSON amb aquesta estructura:
{{
  "cluster_0": {{
    "name": "Nom del segment",
    "description": "Descripció dels trets principals",
    "recommendations": "Recomanacions estratègiques"
  }},
  ...
}}

Retorna només el JSON, sense explicacions addicionals.
"""

    try:
        resposta = prompt_AI(prompt)
        # Intentar extreure JSON de la resposta
        import re
        json_match = re.search(r'\{.*\}', resposta, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "No s'ha pogut generar descripcions válides"}
    except Exception as e:
        print(f"Error generant descripcions: {e}")
        return {"error": str(e)}

def generar_descripcions_clusters(username: str, table_name: str = None) -> dict:
    """
    Genera descripcions dels clusters utilitzant IA
    """
    user_dir = os.path.join('app/users', username)

    # Buscar fitxers de clustering summary
    if table_name:
        clusters_summary_path = os.path.join(user_dir, f'{table_name}_clustering_summary.json')
        clusters_desc_path = os.path.join(user_dir, f'{table_name}_clustering_descriptions.json')
    else:
        # Buscar qualsevol fitxer de clustering summary
        files = [f for f in os.listdir(user_dir) if f.endswith('_clustering_summary.json')]
        if not files:
            print("No s'ha trobat cap fitxer de clustering summary")
            return {}
        clusters_summary_path = os.path.join(user_dir, files[0])
        base_name = files[0].replace('_clustering_summary.json', '')
        clusters_desc_path = os.path.join(user_dir, f'{base_name}_clustering_descriptions.json')

    # Llegir el cluster summary
    try:
        with open(clusters_summary_path, 'r', encoding='utf-8') as f:
            cluster_summary = json.load(f)
    except FileNotFoundError:
        print(f"No s'ha trobat el fitxer {clusters_summary_path}")
        return {}

    # Generar descripcions amb AI
    descripcions = interpretar_clusters_amb_ai(cluster_summary)

    # Guardar les descripcions
    with open(clusters_desc_path, 'w', encoding='utf-8') as f:
        json.dump(descripcions, f, indent=2, ensure_ascii=False)

    print(f"✅ Descripcions guardades a: {clusters_desc_path}")
    return descripcions

def generar_descripcions_clusters(username: str, table_name: str) -> dict:
    """
    Genera descripcions dels clusters utilitzant IA
    """
    clusters_folder = get_user_folder_path(username, 'clusters')

    # Carregar el resum de clusters
    summary_path = os.path.join(clusters_folder, f"{table_name}_clustering_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No s'ha trobat el fitxer de resum: {summary_path}")

    with open(summary_path, 'r', encoding='utf-8') as f:
        cluster_summary = json.load(f)

    # Generar descripcions amb IA
    descripcions = interpretar_clusters_amb_ai(cluster_summary)

    # Guardar les descripcions a la carpeta clusters
    desc_filename = f"{table_name}_clustering_descriptions.json"
    desc_path = os.path.join(clusters_folder, desc_filename)

    with open(desc_path, 'w', encoding='utf-8') as f:
        json.dump(descripcions, f, indent=2, ensure_ascii=False)

    print(f"✅ Descripcions guardades a: {desc_filename}")

    return descripcions