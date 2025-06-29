import sys
import os
import django
from pathlib import Path
# Afegeix la carpeta arrel del projecte al PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Configura el DJANGO_SETTINGS_MODULE abans d'accedir a settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

# Inicialitza Django
django.setup()

# Ara pots accedir a les configuracions
from django.conf import settings

TEMPLATE_DIR = settings.TEMPLATE_DIR
IMAGES_DIR = settings.IMAGES_DIR
IMAGES_JSON = settings.IMAGES_JSON
HTML_OUTPUT = settings.HTML_OUTPUT

print("Ruta plantilles:", TEMPLATE_DIR)
print("Ruta imatges:", IMAGES_DIR)
print("Fitxer imatges JSON:", IMAGES_JSON)
print("Ruta de sortida HTML:", HTML_OUTPUT)
