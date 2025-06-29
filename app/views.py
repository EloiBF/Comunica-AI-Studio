from django.shortcuts import render, redirect
from django.conf import settings
import os
import json
from datetime import datetime
from app.scripts.gen_html import generate_content_from_prompt, render_template
from django.http import HttpResponse, Http404
from mimetypes import guess_type
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

# Landing page (pública)
def landing(request):
    return render(request, 'landing.html')

# Signup
def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')  # Redirige a la pàgina de panell
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

# Login
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('landing')  # Redirige a la página de inicio o donde desees

#  Serveix els fitxers generats
def generated_file(request, filename):
    HTML_OUTPUT = settings.HTML_OUTPUT
    file_path = os.path.join(HTML_OUTPUT, filename)

    if not os.path.exists(file_path):
        raise Http404("Fitxer no trobat")

    mime_type, _ = guess_type(file_path)
    with open(file_path, 'rb') as f:
        return HttpResponse(f.read(), content_type=mime_type)

# Pàgines per l'àrea autenticada
@login_required
def dashboard(request):
    return render(request, 'dashboard.html')


@login_required
def context_view(request):
    return render(request, 'context.html')


@login_required
def generate(request):
    TEMPLATE_DIR = settings.TEMPLATE_DIR
    HTML_OUTPUT = settings.HTML_OUTPUT

    # Si és POST i vol eliminar
    if request.method == "POST" and request.POST.get("action") == "delete":
        filename = request.POST.get("filename")
        if filename:
            base_name = os.path.splitext(filename)[0]
            html_path = os.path.join(HTML_OUTPUT, base_name + ".html")
            json_path = os.path.join(HTML_OUTPUT, base_name + ".json")

            if os.path.exists(html_path):
                os.remove(html_path)
            if os.path.exists(json_path):
                os.remove(json_path)

        return redirect("generate")  # Nom de la URL (assegura’t que coincideix amb el `name` de la teva ruta)

    # Si és POST i vol generar nova comunicació
    elif request.method == "POST":
        communication_name = request.POST.get("name")
        prompt = request.POST.get("prompt")
        template_name = request.POST.get("template")

        generated_data = generate_content_from_prompt(prompt, template_name)

        try:
            # Llegeix plantilla HTML
            template_path = os.path.join(TEMPLATE_DIR, template_name)
            with open(template_path, 'r', encoding='utf-8') as file:
                template = file.read()

            rendered_html = render_template(template, generated_data)

            file_base = communication_name.replace(' ', '_').lower()
            html_filename = f"{file_base}.html"
            json_filename = f"{file_base}.json"

            html_path = os.path.join(HTML_OUTPUT, html_filename)
            json_path = os.path.join(HTML_OUTPUT, json_filename)

            os.makedirs(HTML_OUTPUT, exist_ok=True)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(rendered_html)

            metadata = {
                "name": communication_name,
                "prompt": prompt,
                "template": template_name,
                "created_at": datetime.now().isoformat()
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

        except FileNotFoundError:
            return render(request, 'generate.html', {
                'generated_content': "❌ Plantilla no trobada.",
                'generated_metadata': [],
                'generated_html_files': []
            })

        return redirect("generate")

    # GET: carrega les metadades existents
    generated_metadata = []
    generated_html_files = [f for f in os.listdir(HTML_OUTPUT) if f.endswith('.html')]

    for filename in os.listdir(HTML_OUTPUT):
        if filename.endswith('.json'):
            json_path = os.path.join(HTML_OUTPUT, filename)
            try:
                with open(json_path, 'r', encoding='utf-8') as json_file:
                    metadata = json.load(json_file)
                    metadata['filename'] = filename.replace('.json', '.html')
                    generated_metadata.append(metadata)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error carregant metadades de {filename}: {e}")

    return render(request, 'generate.html', {
        'generated_content': None,
        'generated_html_files': generated_html_files,
        'generated_metadata': generated_metadata
    })

@login_required
def clusters(request):
    return render(request, 'clusters.html')

@login_required
def cluster_communications(request, cluster_id):
    return render(request, 'cluster_communications.html', {'cluster_id': cluster_id})

@login_required
def journey_builder(request):
    return render(request, 'journey.html')

@login_required
def profile(request):
    return render(request, 'profile.html')
