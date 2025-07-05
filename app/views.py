import os
import sys
from pathlib import Path
from datetime import datetime
from mimetypes import guess_type
import json
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse, HttpResponse, Http404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.utils.timezone import now

# Añadir el directorio raíz al path para imports
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))

# Importaciones de la aplicación
from app.scripts.gen_html import generate_content_from_prompt, render_template
from app.scripts.utils import (
    get_user_folder_path,
    get_user_communications_folder,
    load_communications,
    delete_communication,
    save_communication
)
from core.models import UserProfile


# Landing page (pública)
def landing(request):
    return render(request, 'landing.html')

# Signup
def signup_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            try:
                user = form.save()
                # El UserProfile es crea automàticament amb els signals
                login(request, user)
                return redirect('dashboard')
            except Exception as e:
                form.add_error(None, 'Error creant el compte. Si us plau, torna-ho a intentar.')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

# Login
def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            
            # Assegurar-se que el UserProfile existeix
            UserProfile.objects.get_or_create(user=user)
            
            # Redirigir a la pàgina següent o dashboard
            next_url = request.GET.get('next', 'dashboard')
            return redirect(next_url)
    else:
        form = AuthenticationForm()
    
    return render(request, 'login.html', {'form': form})
    
# Logout
def logout_view(request):
    logout(request)
    return redirect('landing')  # Redirigeix a la pàgina d'inici o on desitgis

#  Serveix els fitxers generats
@login_required
def generated_file(request, filename):
    """
    Sirve archivos generados desde la carpeta de comunicaciones del usuario
    """
    user_htmls_folder = get_user_communications_folder(request.user.username)
    file_path = os.path.join(user_htmls_folder, filename)

    if not os.path.exists(file_path):
        raise Http404("Fitxer no trobat")

    mime_type, _ = guess_type(file_path)
    with open(file_path, 'rb') as f:
        return HttpResponse(f.read(), content_type=mime_type)

# Pàgines per l'àrea autenticada
@login_required
def dashboard(request):
    """
    Muestra el dashboard con las comunicaciones generadas por el usuario
    """
    user_folder = get_user_communications_folder(request.user.username)
    generated_metadata = load_communications(user_folder)

    return render(request, 'dashboard.html', {
        'generated_metadata': generated_metadata
    })

@login_required
def context_view(request):
    from app.scripts.gen_html import get_user_context, save_user_context
    
    if request.method == 'POST':
        # Process form data
        context_data = []
        i = 1
        while True:
            title_key = f'title_{i}'
            content_key = f'content_{i}'
            type_key = f'type_{i}'  # New field to specify the type of context
            
            if title_key not in request.POST or content_key not in request.POST:
                break
                
            title = request.POST[title_key].strip()
            content = request.POST[content_key].strip()
            type = request.POST.get(type_key, 'company').strip().lower()  # Default to 'company'
            
            if title and content:  # Only add non-empty entries
                context_data.append({
                    'type': type,
                    'title': title,
                    'content': content
                })
                
            i += 1
        
        # Save the context data
        if context_data:
            save_user_context(request.user.username, context_data)
            
        # Redirect to prevent form resubmission
        return redirect('context')
    
    # Load existing context data
    context_data = get_user_context(request.user.username)
    
    # Prepare context for the template
    context_entries = []
    context_types = ['company', 'products', 'target_audience', 'communication_style']
    
    # Add existing entries
    for context_type in context_types:
        if context_type in context_data:
            for title, content in context_data[context_type].items():
                context_entries.append({
                    'id': len(context_entries) + 1,
                    'type': context_type,
                    'title': title,
                    'content': content
                })
    
    return render(request, 'context.html', {
        'context_entries': context_entries,
        'context_types': context_types,
        'next_id': len(context_entries) + 1 if context_entries else 1
    })

@login_required
def generate(request):
    """
    Vista para generar nuevas comunicaciones y gestionar las existentes
    """
    # Configuración de rutas
    TEMPLATE_DIR = settings.TEMPLATE_DIR
    username = request.user.username
    
    # Obtener carpeta de comunicaciones del usuario
    HTML_OUTPUT = get_user_communications_folder(username)
    os.makedirs(HTML_OUTPUT, exist_ok=True)

    # Cargar clusters disponibles
    clusters_folder = os.path.join(settings.BASE_DIR, 'app', 'users', username, 'clusters')
    os.makedirs(clusters_folder, exist_ok=True)
    clustering_files = [f for f in os.listdir(clusters_folder) if f.endswith('_clustering_summary.json')]

    # Definición de las plantillas disponibles
    TEMPLATES = [
        ('sale.html', '💰', 'Vendes', 'Promocions i ofertes comercials'),
        ('tips.html', '💡', 'Consells', 'Consells útils i pràctics'),
        ('welcome.html', '👋', 'Benvinguda', 'Missatges de benvinguda'),
        ('survey.html', '📊', 'Enquesta', "Recollida d'opinions"),
        ('thank_you.html', '🙏', 'Agraïment', 'Missatges de gratitud'),
        ('reminder.html', '⏰', 'Recordatori', 'Recordatoris i avisos'),
        ('announcement.html', '📢', 'Anunci', 'Comunicats oficials'),
    ]

    available_datasets = []
    for file_name in clustering_files:
        table_name = file_name.replace('_clustering_summary.json', '')
        clusters_desc_path = os.path.join(clusters_folder, f'{table_name}_clustering_descriptions.json')
        clusters_summary_path = os.path.join(clusters_folder, file_name)

        # Carrega descripcions i summary
        clusters_descriptions = None
        if os.path.exists(clusters_desc_path):
            with open(clusters_desc_path, 'r', encoding='utf-8') as f:
                clusters_descriptions = json.load(f)
        with open(clusters_summary_path, 'r', encoding='utf-8') as f:
            clusters_data = json.load(f)

        available_datasets.append({
            'name': table_name,
            'descriptions': clusters_descriptions,
            'num_clusters': len(clusters_data)
        })

    # Detectar subscripció
    try:
        user_profile, created = UserProfile.objects.get_or_create(user=request.user)
        subscription = user_profile.subscription
    except Exception:
        subscription = 'free'

    template_name = {
        'free': 'generate.html',
        'premium': 'generate_prem.html',
        'enterprise': 'generate_enterprise.html'
    }.get(subscription, 'generate.html')

    # POST: eliminar comunicació
    if request.method == "POST" and request.POST.get("action") == "delete":
        filename = request.POST.get("filename")
        if filename:
            delete_communication(HTML_OUTPUT, filename)
        return redirect("generate")

    # POST: generar nova comunicació
    elif request.method == "POST":
        communication_name = request.POST.get("name")
        prompt = request.POST.get("prompt")
        selected_template = request.POST.get("template")
        dataset_name = request.POST.get("dataset", "")
        cluster_id = request.POST.get("cluster", "")

        generated_data = generate_content_from_prompt(prompt, selected_template)

        try:
            template_path = os.path.join(TEMPLATE_DIR, selected_template)
            with open(template_path, 'r', encoding='utf-8') as file:
                template = file.read()

            rendered_html = render_template(template, generated_data)

            # Obtener colores del formulario o usar valores por defecto
            colors = {
                'primary_color': request.POST.get('primary_color', '#667eea'),
                'secondary_color': request.POST.get('secondary_color', '#764ba2'),
                'accent_color': request.POST.get('accent_color', '#4a90e2'),
                'background_color': request.POST.get('background_color', '#ffffff'),
                'text_color': request.POST.get('text_color', '#333333')
            }
            
            # Guardar comunicación con metadatos incluyendo colores
            metadata = {
                'name': communication_name,
                'template': selected_template,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dataset': dataset_name,
                'cluster': cluster_id,
                'prompt': prompt,
                'colors': colors  # Añadir los colores a los metadatos
            }
            
            # Guardar usando la función de utilidad
            html_filename, _ = save_communication(HTML_OUTPUT, rendered_html, metadata)
            return redirect('generated_file', filename=html_filename)
        except FileNotFoundError:
            return render(request, template_name, {
                'generated_content': "❌ Plantilla no trobada.",
                'generated_metadata': [],
                'generated_html_files': [],
                'subscription': subscription,
                'available_datasets': available_datasets,
                'templates': TEMPLATES,  # 
            })

    # GET: mostrar comunicacions existents
    generated_metadata = load_communications(HTML_OUTPUT)
    generated_html_files = [f for f in os.listdir(HTML_OUTPUT) if f.endswith('.html')]

    return render(request, template_name, {
        'generated_content': None,
        'generated_html_files': generated_html_files,
        'generated_metadata': generated_metadata,
        'available_datasets': available_datasets,
        'subscription': subscription,
        'templates': TEMPLATES,  # 👈 AFEGIT
    })

def _adjust_color(color, amount=30):
    """
    Lighten or darken a color by a given amount.
    Amount should be between -255 (darker) and 255 (lighter).
    """
    try:
        color = color.lstrip('#')
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        
        r = max(0, min(255, r + amount))
        g = max(0, min(255, g + amount))
        b = max(0, min(255, b + amount))
        
        return f"#{r:02x}{g:02x}{b:02x}"
    except:
        return color

@login_required
def template_preview(request, template_name):
    # Get the template path
    PREVIEWS_DIR = settings.PREVIEWS_DIR
    template_file = template_name if template_name.endswith(".html") else f"{template_name}.html"
    template_path = os.path.join(PREVIEWS_DIR, template_file)

    # Check if file exists
    if not os.path.exists(template_path):
        raise Http404("Template not found.")
    
    # Get the primary color from query parameters, default to #667eea (indigo)
    primary_color = request.GET.get('color', '#667eea')
    
    # Read the template content
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Generate color variants
        primary_hover = _adjust_color(primary_color, -10)
        primary_light = _adjust_color(primary_color, 40)
        
        # Add dynamic CSS for the primary color
        dynamic_css = f"""
        <style>
            :root {{
                --primary-color: {primary_color};
                --primary-hover: {primary_hover};
                --primary-light: {primary_light};
            }}
            
            /* Add any specific element styling that uses the primary color */
            .btn-primary {{
                background-color: var(--primary-color);
                border-color: var(--primary-color);
            }}
            .btn-primary:hover {{
                background-color: var(--primary-hover);
                border-color: var(--primary-hover);
            }}
            .text-primary {{
                color: var(--primary-color) !important;
            }}
            .bg-primary {{
                background-color: var(--primary-color) !important;
            }}
        </style>
        """
        
        # Insert the dynamic CSS right after the opening <head> tag
        if '</head>' in template_content:
            template_content = template_content.replace('</head>', f"{dynamic_css}</head>")
        else:
            # If no head tag, add one at the beginning
            template_content = f"<head>{dynamic_css}</head>{template_content}"
        
        # Create a response with the modified template
        response = HttpResponse(template_content, content_type='text/html')
        
        # Allow the iframe to be embedded
        response['X-Frame-Options'] = 'ALLOW-FROM *'
        response['Content-Security-Policy'] = "frame-ancestors *"
        
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise Http404(f"Error loading template: {str(e)}")


@login_required
def clusters(request):
    from app.scripts.gen_clusters import full_clustering_pipeline
    from app.scripts.gen_clusters_desc import generar_descripcions_clusters
    import os, json

    # Obtenir el directori de l'usuari
    user_folder = get_user_folder_path(request.user.username)
    message = None
    selected_dataset = request.GET.get('dataset', None)

    # Si el mètode és POST, gestionar el fitxer o l'eliminació del dataset
    if request.method == 'POST':
        if 'file' in request.FILES:
            # Gestió de la càrrega del fitxer
            uploaded_file = request.FILES['file']
            n_clusters = int(request.POST.get('n_clusters', 0))
            user_data_dir = os.path.join(user_folder, 'data_files')
            os.makedirs(user_data_dir, exist_ok=True)
            file_path = os.path.join(user_data_dir, uploaded_file.name)
            
            # Guardar el fitxer temporalment
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            
            try:
                # Generar clusters amb el pipeline
                cluster_summary, json_path = full_clustering_pipeline(file_path, request.user.username, n_clusters)
                json_filename = os.path.basename(json_path)
                table_name = json_filename.replace('_clustering_summary.json', '')

                # Generar descripcions
                clusters_descriptions = generar_descripcions_clusters(request.user.username, table_name)
                message = "✅ Clusters generats correctament!"
                selected_dataset = table_name
            except Exception as e:
                print(f"Error generant clusters: {str(e)}")  
                message = "❌ Error processant el fitxer. Si us plau, revisa el format i torna-ho a intentar."
        
        elif request.POST.get('delete_dataset'):
            # Eliminació del dataset
            dataset_to_delete = request.POST.get('delete_dataset')
            try:
                # Fitxers associats al dataset
                files_to_delete = [
                    f'{dataset_to_delete}_clustering_summary.json',
                    f'{dataset_to_delete}_clustering_descriptions.json'
                ]
                for file_name in files_to_delete:
                    file_path = os.path.join(user_folder, 'clusters', file_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)

                message = f"✅ Dataset '{dataset_to_delete}' eliminat correctament!"

                # Si estem visualitzant el dataset eliminat, netegem la selecció
                if selected_dataset == dataset_to_delete:
                    selected_dataset = None
            except Exception as e:
                print(f"Error eliminant dataset: {str(e)}")  
                message = "❌ Error eliminant el dataset. Si us plau, torna-ho a intentar."

    # Obtenir tots els datasets disponibles (màxim 3)
    clusters_folder = get_user_folder_path(request.user.username, 'clusters')
    clustering_files = [f for f in os.listdir(clusters_folder) if f.endswith('_clustering_summary.json')] if os.path.exists(clusters_folder) else []
    
    available_datasets = []
    for file_name in clustering_files:
        table_name = file_name.replace('_clustering_summary.json', '')
        clusters_summary_path = os.path.join(clusters_folder, file_name)
        
        # Obtenir data de modificació
        mod_time = os.path.getmtime(clusters_summary_path)

        # Carregar dades del dataset
        with open(clusters_summary_path, 'r', encoding='utf-8') as f:
            clusters_data = json.load(f)

        # Carregar descripcions si existeixen
        clusters_desc_path = os.path.join(clusters_folder, f'{table_name}_clustering_descriptions.json')
        clusters_descriptions = None
        if os.path.exists(clusters_desc_path):
            with open(clusters_desc_path, 'r', encoding='utf-8') as f:
                clusters_descriptions = json.load(f)

        available_datasets.append({
            'name': table_name,
            'data': clusters_data,
            'descriptions': clusters_descriptions,
            'modified_time': mod_time,
            'num_clusters': len(clusters_data)
        })

    # Ordenar per data de modificació (més recent primer) i limitar a 3
    available_datasets.sort(key=lambda x: x['modified_time'], reverse=True)
    available_datasets = available_datasets[:3]

    # Si no hi ha dataset seleccionat, seleccionar el més recent
    if not selected_dataset and available_datasets:
        selected_dataset = available_datasets[0]['name']

    # Obtenir dades del dataset seleccionat
    selected_data = None
    for dataset in available_datasets:
        if dataset['name'] == selected_dataset:
            selected_data = dataset
            break

    return render(request, 'clusters.html', {
        'available_datasets': available_datasets,
        'selected_dataset': selected_dataset,
        'selected_data': selected_data,
        'message': message,
        'max_datasets': 3
    })


@login_required
def delete_dataset(request):
    if request.method == 'POST':
        dataset_name = request.POST.get('delete_dataset')

        try:
            # Obtenir el dataset pel seu nom
            dataset = Dataset.objects.get(name=dataset_name)
        except Dataset.DoesNotExist:
            raise Http404("Dataset no trobat")

        # Ruta per eliminar els arxius JSON associats als clusters
        json_files = dataset.get_json_files()  # Suposant que tens un mètode per obtenir els arxius JSON

        # Eliminar els arxius JSON associats al dataset
        for json_file in json_files:
            file_path = os.path.join(settings.MEDIA_ROOT, json_file)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Eliminar el dataset de la base de dades
        dataset.delete()

        # Redirigir a la pàgina amb un missatge d'èxit
        return redirect('clusters_page')  # Redirigeix on correspongui

    return redirect('clusters_page')  # Si no és un POST, redirigeix


@login_required
def cluster_communications(request, cluster_id):
    return render(request, 'cluster_communications.html', {'cluster_id': cluster_id})

@login_required
def journey_builder(request):
    return render(request, 'journey.html')

@login_required
def profile(request):
    return render(request, 'profile.html')


@login_required
def user_details(request):
    user = request.user
    try:
        profile, created = UserProfile.objects.get_or_create(user=user)
        subscription = profile.get_subscription_display()
    except Exception:
        subscription = 'Free'

    return render(request, 'user_details.html', {
        'user': user,
        'subscription': subscription,
        'profile': profile if 'profile' in locals() else None,
    })