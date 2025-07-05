import os
import sys
import json
import logging
from pathlib import Path
from django.conf import settings
from django.contrib.auth import get_user_model

# Set up Django environment
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
import django
django.setup()

# Import Django models
from app.models import UserContext, GeneratedCommunication

# Import necessary functions
from app.scripts.utils import extract_json_from_text, prompt_AI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_user_context(username, context_data):
    """
    Saves the user's context to the database.
    
    Args:
        username (str): The username
        context_data (list): List of dictionaries with the structure:
            [
                {
                    "type": "company" | "products" | "target_audience" | "communication_style",
                    "title": "Section title",
                    "content": "Detailed content"
                },
                ...
            ]
    """
    try:
        # Get the user
        User = get_user_model()
        user = User.objects.get(username=username)
        
        # Convert list of dictionaries to structured data
        structured_data = {
            "company": {},
            "products": {},
            "target_audience": {},
            "communication_style": {}
        }
        
        for entry in context_data:
            section_type = entry.get("type", "company")
            if section_type in structured_data:
                structured_data[section_type][entry["title"]] = entry["content"]
        
        # Create or update the context
        context, created = UserContext.objects.update_or_create(
            user=user,
            context_type='user_preferences',
            defaults={
                'name': 'User Preferences',
                'context_data': structured_data,
                'is_active': True
            }
        )
        
        logger.info(f"User context saved for {username}")
        return context
        
    except Exception as e:
        logger.error(f"Error saving user context: {str(e)}")
        raise

def get_user_context(username):
    """
    Retrieves the user's context from the database.
    
    Args:
        username (str): The username
        
    Returns:
        dict: Context data in {"title": "content"} format
    """
    try:
        User = get_user_model()
        user = User.objects.get(username=username)
        
        # Get the most recent active context
        context = UserContext.objects.filter(
            user=user, 
            is_active=True
        ).order_by('-updated_at').first()
        
        return context.context_data if context else {}
        
    except User.DoesNotExist:
        logger.error(f"User {username} not found")
        return {}
    except Exception as e:
        logger.error(f"Error retrieving user context: {str(e)}")
        return {}

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

def get_context_prompt() -> str:
    """
    Retorna el prompt per generar el context inicial de l'usuari.
    """
    return (
        "You are a context expert and need to generate a comprehensive description of a company's context.\n"
        "The context should include:\n"
        "1. Company information: sector, mission, values, target audience, communication style\n"
        "2. Products and services: main offerings, unique features, competitive advantages\n"
        "3. Target audience: demographics, preferences, communication channels\n"
        "4. Communication preferences: tone, style, preferred channels\n"
        "\n"
        "Format the response as a JSON object with these keys:\n"
        "- company_info: Detailed company description\n"
        "- products: Product/service information\n"
        "- target_audience: Audience characteristics\n"
        "- communication_style: Preferred communication style\n"
    )

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
            "message: Brief and appealing description of the product or offer,\n"
            "cta_text: Text for the call-to-action button,\n"
            "cta_url: URL for the call-to-action button,\n"
            "list_item_1: First benefit or feature,\n"
            "list_item_2: Second benefit or feature,\n"
            "list_item_3: Third benefit or feature,\n"
            "color_primary: Main color for gradients and accents,\n"
            "color_secondary: Secondary color for gradients\n"
            "}\n"
            "\nDo not include any explanation or text outside the JSON."
        )
    elif template_name == "tips.html":
        return (
            "Return a valid JSON with the following fields:\n"
            "{\n"
            "title: General title for the tips,\n"
            "feature_title_1: Title of the first tip,\n"
            "feature_description_1: Content of the first tip,\n"
            "feature_title_2: Title of the second tip,\n"
            "feature_description_2: Content of the second tip,\n"
            "feature_title_3: Title of the third tip (optional),\n"
            "feature_description_3: Content of the third tip (optional),\n"
            "cta_text: Text for the call-to-action button,\n"
            "cta_url: URL for the call-to-action button\n"
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
            "message: Main welcome message,\n"
            "list_item_1: First benefit or feature,\n"
            "list_item_2: Second benefit or feature,\n"
            "list_item_3: Third benefit or feature,\n"
            "cta_text: Text for the call-to-action button,\n"
            "cta_url: URL for the call-to-action button\n"
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
            "title: Thank you title,\n"
            "message: Main thank you message,\n"
            "feature_title_1: Title for appreciation section,\n"
            "feature_description_1: Content for appreciation section,\n"
            "list_item_1: First next step,\n"
            "list_item_2: Second next step,\n"
            "list_item_3: Third next step,\n"
            "contact_email: Email address for contact,\n"
            "website_url: Website URL\n"
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
    """
    Render a template string with the given data.
    
    Args:
        template_str (str): The template string with {variable} placeholders
        data (dict): Dictionary of variables to replace in the template
        
    Returns:
        str: Rendered template with variables replaced
    """
    try:
        for key, value in (data or {}).items():
            if value is not None:  # Only replace if value is not None
                template_str = template_str.replace(f'{{{key}}}', str(value))
        return template_str
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        raise

def save_generated_communication(username: str, template_name: str, prompt: str, content: dict, html_content: str):
    """
    Save the generated communication to the database.
    
    Args:
        username (str): The username
        template_name (str): Name of the template used
        prompt (str): The original prompt
        content (dict): The generated content
        html_content (str): The final rendered HTML
        
    Returns:
        GeneratedCommunication: The saved communication object
    """
    try:
        User = get_user_model()
        user = User.objects.get(username=username)
        
        # Create a name for the communication
        name = f"{template_name} - {content.get('title', 'Untitled')}"
        
        # Save to database
        communication = GeneratedCommunication.objects.create(
            user=user,
            name=name,
            template_name=template_name,
            prompt=prompt,
            content=content,
            html_content=html_content,
            status='completed'
        )
        
        logger.info(f"Generated communication saved for user {username}")
        return communication
        
    except Exception as e:
        logger.error(f"Error saving generated communication: {str(e)}")
        raise

def main(prompt: str, username: str, template_name: str = "sale.html"):
    """
    Main function to generate HTML from a prompt and template.
    
    Args:
        prompt (str): The user's prompt
        username (str): The username
        template_name (str): Name of the template to use
        
    Returns:
        str: The generated HTML content
    """
    try:
        # Load the template
        template = load_template(template_name)
        
        # Get user context
        context = get_user_context(username)
        
        # Build the prompt
        full_prompt = build_prompt(template_name, prompt, context)
        
        # Generate content
        content = generate_content_from_prompt(full_prompt, template_name)
        
        # Render the template with the generated content
        html_content = render_template(template, content)
        
        # Save the communication to the database
        save_generated_communication(
            username=username,
            template_name=template_name,
            prompt=prompt,
            content=content,
            html_content=html_content
        )
        
        return html_content
        
    except Exception as e:
        logger.error(f"Error in main generation: {str(e)}")
        raise

if __name__ == "__main__":
    prompt = "Vols generar una oferta especial de descompte per un producte popular."
    template_name = "sale.html"  # o "tips.html"
    main(prompt, template_name, name_html="test")
