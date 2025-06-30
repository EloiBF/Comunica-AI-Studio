# Importar llibreries necessàries
from pathlib import Path 
import os, sys

# Definim entorn on s'executarà aquest script (com si fos el root)
base_dir = Path(__file__).resolve().parent.parent.parent  # Aquí, puja un nivell més alt per arribar a l'arrel
sys.path.append(str(base_dir))

# Importem funcions necessàries
from app.scripts.utils import extract_json_from_text
from app.scripts.utils import prompt_AI


# Definir entorn de Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

from django.conf import settings

# Funcions per rutes dinàmiques d'usuari
def get_user_html_dir(username):
    """Retorna la carpeta HTML de l'usuari"""
    return settings.BASE_DIR / 'app' / 'users' / username / 'htmls'

# Accedir a les rutes configurades a settings.py
TEMPLATE_DIR = settings.TEMPLATE_DIR
IMAGES_DIR = settings.IMAGES_DIR
IMAGES_JSON = settings.IMAGES_JSON



def load_template(template_name: str) -> str:
    path = TEMPLATE_DIR / template_name
    print(f"🔍 Carregant plantilla des de: {path}")  # Afegeix aquesta línia per veure el camí complet
    if not path.exists():
        raise FileNotFoundError(f"No s'ha trobat la plantilla: {template_name}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_base_prompt(template_name: str) -> str:
    """
    Retorna el prompt base que es fa servir per generar el contingut, 
    depenent de la plantilla seleccionada (ex: sale, tips).
    """
    if template_name == "sale.html":
        return (
            "Ets un expert en màrqueting digital i has de generar contingut per una comunicació HTML per vendre un producte."
            " El contingut ha de ser atractiu, concís i enfocat a la conversió."
        )
    elif template_name == "tips.html":
        return (
            "Ets un expert en consells pràctics i has de generar contingut per una comunicació HTML amb diversos consells útils."
            " El contingut ha de ser directe, fàcil de llegir i orientat a l'aplicació pràctica dels consells."
        )
    else:
        raise ValueError(f"Plantilla no reconeguda: {template_name}")

def get_html_prompt(template_name: str) -> str:
    """
    Retorna el prompt específic per generar un objecte JSON que descriu el contingut esperat per cada plantilla.
    Ara inclou un exemple clar de l'estructura JSON esperada en format vàlid.
    """
    if template_name == "sale.html":
        return (
            "Retorna un JSON vàlid amb els següents camps:\n"
            "{\n"
            "title: Títol atractiu de l'oferta,\n"
            "content: Descripció breu i atractiva del producte o oferta,\n"
            "image_url: URL d'una imatge representativa de l'oferta\n"
            "}\n"
            "\nNo afegeixis cap explicació ni text fora del JSON."
        )
    elif template_name == "tips.html":
        return (
            "Retorna un JSON vàlid amb els següents camps:\n"
            "{\n"
            "title: Títol general dels consells,\n"
            "tip_1_title: Títol del primer consell,\n"
            "tip_1_content: Contingut del primer consell,\n"
            "tip_2_title: Títol del segon consell,\n"
            "tip_2_content: Contingut del segon consell,\n"
            "tip_3_title: Títol del tercer consell (opcional),\n"
            "tip_3_content: Contingut del tercer consell (opcional)\n"
            "}\n"
            "\nNo afegeixis cap explicació ni text fora del JSON."
        )
    else:
        raise ValueError(f"Plantilla no reconeguda: {template_name}")

def build_prompt(template_name: str, prompt: str) -> str:
    """
    Construeix el prompt final, combinant el prompt base amb el prompt específic de la plantilla.
    Ara inclou un exemple clar de la resposta esperada en format JSON.
    """
    base_prompt = get_base_prompt(template_name)
    html_prompt = get_html_prompt(template_name)

    # Final prompt combinat
    full_prompt = (
        f"{base_prompt}\n\n"
        f"Genera el contingut seguint aquests requeriments específics per la plantilla '{template_name}':\n"
        f"{html_prompt}\n\n"
        f"El prompt base és:\n{prompt}"
    )
    return full_prompt

def generate_content_from_prompt(prompt: str, template_name: str) -> dict:
    """
    Genera contingut utilitzant el prompt creat i retorna un diccionari amb els camps corresponents.
    """
    full_prompt = build_prompt(template_name, prompt)
    print(f"📝 Generant contingut amb el prompt:\n{full_prompt}\n")

    raw_content = prompt_AI(full_prompt)
    return extract_json_from_text(raw_content)

def render_template(template_str: str, data: dict) -> str:
    rendered = template_str
    for key, value in data.items():
        rendered = rendered.replace(f"[{key}]", value)
    return rendered


# Script principal per generar HTML a partir d'un prompt i una plantilla --> no usat directament a l'aplicació, però útil per proves i generació manual
def main(prompt: str, username: str, template_name: str = "sale.html") -> None:
    print("🔄 Generant contingut amb Groq...")
    data = generate_content_from_prompt(prompt, template_name)
    # Afegir imatge temporalment genèrica
    data["image_url"] = "https://via.placeholder.com/600x300.png?text=Oferta+Especial"
    print("📄 Carregant plantilla...")
    template = load_template(template_name)
    print("🧠 Renderitzant plantilla amb contingut...")
    html_final = render_template(template, data)
    # Guardar el resultat generat a la ruta configurada per a l'usuari
    output_path = get_user_html_dir(username) / f'{prompt}.html'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_final)
    print(f"✅ HTML generat i guardat a: {output_path}")

if __name__ == "__main__":
    prompt = "Vols generar una oferta especial de descompte per un producte popular."
    template_name = "sale.html"  # o "tips.html"
    main(prompt, template_name, name_html="test")
