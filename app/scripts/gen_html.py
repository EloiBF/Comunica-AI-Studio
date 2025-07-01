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
    Returns the base prompt used to generate content,
    depending on the selected template (e.g., sale, tips).
    """
    if template_name == "sale.html":
        return (
            "You are a digital marketing expert and need to generate content for an HTML communication to sell a product."
            " The content must be engaging, concise, and focused on conversion."
        )
    elif template_name == "tips.html":
        return (
            "You are a practical advice expert and need to generate content for an HTML communication with several useful tips."
            " The content should be direct, easy to read, and oriented toward practical application."
        )
    elif template_name == "newsletter.html":
        return (
            "You are a corporate communications expert and need to generate content for an HTML newsletter."
            " The content should be informative, engaging, and maintain reader interest."
        )
    elif template_name == "welcome.html":
        return (
            "You are a user experience expert and need to generate content for a welcome HTML message."
            " The content should be warm, welcoming, and make the customer feel valued."
        )
    elif template_name == "event.html":
        return (
            "You are an event planning expert and need to generate content for an HTML invitation."
            " The content should be exciting, informative, and create anticipation for the event."
        )
    elif template_name == "survey.html":
        return (
            "You are a market research expert and need to generate content for an HTML survey."
            " The content should be clear, motivating, and explain the importance of participation."
        )
    elif template_name == "thank_you.html":
        return (
            "You are a customer relations expert and need to generate content for a thank-you HTML message."
            " The content should be sincere, personalized, and strengthen the relationship with the client."
        )
    elif template_name == "reminder.html":
        return (
            "You are an assertive communication expert and need to generate content for an HTML reminder."
            " The content should be helpful, non-intrusive, and action-oriented."
        )
    elif template_name == "announcement.html":
        return (
            "You are a corporate communications expert and need to generate content for an HTML announcement."
            " The content should be clear, impactful, and convey the information effectively."
        )
    elif template_name == "seasonal.html":
        return (
            "You are a seasonal marketing expert and need to generate content for a themed HTML communication."
            " The content should capture the spirit of the season and create an emotional connection."
        )
    else:
        raise ValueError(f"Unrecognized template: {template_name}")

def get_html_prompt(template_name: str) -> str:
    """
    Returns the specific prompt to generate a JSON object that describes
    the expected content structure for each template.
    Now includes a clear example of the expected JSON format.
    """
    if template_name == "sale.html":
        return (
            "Return a valid JSON with the following fields:\n"
            "{\n"
            "title: Catchy offer title,\n"
            "content: Brief and appealing description of the product or offer,\n"
            "image_url: URL of a representative image of the offer\n"
            "}\n"
            "\nDo not include any explanation or text outside the JSON."
        )
    elif template_name == "tips.html":
        return (
            "Return a valid JSON with the following fields:\n"
            "{\n"
            "title: General title for the tips,\n"
            "tip_1_title: Title of the first tip,\n"
            "tip_1_content: Content of the first tip,\n"
            "tip_2_title: Title of the second tip,\n"
            "tip_2_content: Content of the second tip,\n"
            "tip_3_title: Title of the third tip (optional),\n"
            "tip_3_content: Content of the third tip (optional)\n"
            "}\n"
            "\nDo not include any explanation or text outside the JSON."
        )
    elif template_name == "newsletter.html":
        return (
            "Return a valid JSON with the following fields:\n"
            "{\n"
            "title: Newsletter title,\n"
            "intro: Introduction or greeting,\n"
            "article_1_title: Title of the first article,\n"
            "article_1_content: Content of the first article,\n"
            "article_2_title: Title of the second article,\n"
            "article_2_content: Content of the second article,\n"
            "closing: Closing message\n"
            "}\n"
            "\nDo not include any explanation or text outside the JSON."
        )
    elif template_name == "welcome.html":
        return (
            "Return a valid JSON with the following fields:\n"
            "{\n"
            "title: Welcome title,\n"
            "welcome_message: Main welcome message,\n"
            "next_steps: Suggested next steps or actions,\n"
            "support_info: Support or contact information\n"
            "}\n"
            "\nDo not include any explanation or text outside the JSON."
        )
    elif template_name == "event.html":
        return (
            "Return a valid JSON with the following fields:\n"
            "{\n"
            "event_title: Event title,\n"
            "event_description: Description of the event,\n"
            "event_date: Date of the event,\n"
            "event_location: Event location,\n"
            "registration_info: Registration information\n"
            "}\n"
            "\nDo not include any explanation or text outside the JSON."
        )
    elif template_name == "survey.html":
        return (
            "Return a valid JSON with the following fields:\n"
            "{\n"
            "survey_title: Survey title,\n"
            "survey_intro: Introduction explaining the survey,\n"
            "participation_benefit: Benefit of participating,\n"
            "estimated_time: Estimated time to complete,\n"
            "survey_link: Link to the survey\n"
            "}\n"
            "\nDo not include any explanation or text outside the JSON."
        )
    elif template_name == "thank_you.html":
        return (
            "Return a valid JSON with the following fields:\n"
            "{\n"
            "thank_you_title: Thank you title,\n"
            "thank_you_message: Main thank you message,\n"
            "appreciation_details: Specific details of what is appreciated,\n"
            "future_relationship: Message about future relationship\n"
            "}\n"
            "\nDo not include any explanation or text outside the JSON."
        )
    elif template_name == "reminder.html":
        return (
            "Return a valid JSON with the following fields:\n"
            "{\n"
            "reminder_title: Reminder title,\n"
            "reminder_message: Main reminder message,\n"
            "action_required: Required action,\n"
            "deadline: Deadline or due date,\n"
            "contact_info: Contact info for questions\n"
            "}\n"
            "\nDo not include any explanation or text outside the JSON."
        )
    elif template_name == "announcement.html":
        return (
            "Return a valid JSON with the following fields:\n"
            "{\n"
            "announcement_title: Announcement title,\n"
            "announcement_content: Main content of the announcement,\n"
            "key_points: Key points or highlights,\n"
            "effective_date: Effective date,\n"
            "additional_info: Additional information\n"
            "}\n"
            "\nDo not include any explanation or text outside the JSON."
        )
    elif template_name == "seasonal.html":
        return (
            "Return a valid JSON with the following fields:\n"
            "{\n"
            "seasonal_title: Seasonal theme title,\n"
            "seasonal_greeting: Seasonal greeting,\n"
            "seasonal_content: Themed content,\n"
            "seasonal_offer: Seasonal offer or promotion,\n"
            "seasonal_wishes: Seasonal wishes\n"
            "}\n"
            "\nDo not include any explanation or text outside the JSON."
        )
    else:
        raise ValueError(f"Unrecognized template: {template_name}")

def build_prompt(template_name: str, prompt: str) -> str:
    """
    Builds the final prompt by combining the base prompt with the specific template instructions.
    Now includes a clear example of the expected JSON output format.
    """
    base_prompt = get_base_prompt(template_name)
    html_prompt = get_html_prompt(template_name)

    full_prompt = (
        f"{base_prompt}\n\n"
        f"Generate content following these specific requirements for the '{template_name}' template:\n"
        f"{html_prompt}\n\n"
        "Always give answer in the same language as the original user prompt.\n\n"
        f"The original user prompt is:\n{prompt}"
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
