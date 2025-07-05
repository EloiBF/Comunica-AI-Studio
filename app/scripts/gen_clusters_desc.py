import os
import json
import re
from app.scripts.utils import prompt_AI, get_user_folder_path

def _generar_prompt_clusters(cluster_summary: dict) -> str:
    return f"""
    Ets un expert en segmentació de clients i anàlisi de dades.

    Analitza el següent resum de clústers i genera per a cada clúster:
    1. Un nom descriptiu
    2. Una descripció de les principals característiques
    3. Recomanacions o estratègies útils

    Respon només en format JSON amb aquesta estructura:
    {{
      "cluster_0": {{
        "name": "Nom del segment",
        "description": "Descripció de les característiques",
        "recommendations": "Recomanacions estratègiques"
      }},
      ...
    }}

    Aquí tens les dades:
    {json.dumps(cluster_summary, indent=2, ensure_ascii=False)}
    """

def _interpretar_clusters_amb_ai(cluster_summary: dict) -> dict:
    try:
        prompt = _generar_prompt_clusters(cluster_summary)
        resposta = prompt_AI(prompt)
        match = re.search(r'{.*}', resposta, re.DOTALL)
        return json.loads(match.group()) if match else {"error": "No s'ha pogut obtenir JSON vàlid"}
    except Exception as e:
        return {"error": str(e)}

def generar_descripcions_clusters(username: str, table_name: str) -> dict:
    """
    Genera descripcions per als clústers d'un dataset de l'usuari i les desa
    """
    carpeta_clusters = get_user_folder_path(username, 'clusters')
    path_resum = os.path.join(carpeta_clusters, f"{table_name}_clustering_summary.json")
    path_desc = os.path.join(carpeta_clusters, f"{table_name}_clustering_descriptions.json")

    if not os.path.exists(path_resum):
        raise FileNotFoundError(f"No s'ha trobat el fitxer de resum: {path_resum}")

    with open(path_resum, 'r', encoding='utf-8') as f:
        resum_clusters = json.load(f)

    descripcions = _interpretar_clusters_amb_ai(resum_clusters)

    with open(path_desc, 'w', encoding='utf-8') as f:
        json.dump(descripcions, f, indent=2, ensure_ascii=False)

    print(f"✅ Descripcions guardades a: {path_desc}")
    return descripcions
