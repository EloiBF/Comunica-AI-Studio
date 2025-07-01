from django.shortcuts import render, redirect
from django.conf import settings
import os,sys
from pathlib import Path
import json
from django.http import JsonResponse
from datetime import datetime
from django.http import HttpResponse, Http404
from mimetypes import guess_type
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.utils.timezone import now

# Definim entorn on s'executarà aquest script (com si fos el root)
base_dir = Path(__file__).resolve().parent.parent.parent  # Aquí, puja un nivell més alt per arribar a l'arrel
sys.path.append(str(base_dir))

from app.scripts.utils import get_user_folder_path
from app.scripts.gen_html import generate_content_from_prompt, render_template
from core.models import UserProfile


# Landing page (pública)
def landing(request):
    return render(request, 'landing.html')


# Signup
def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Crear el perfil de l'usuari després del registre
            UserProfile.objects.create(user=user)
            login(request, user)
            return redirect('dashboard')  # Redirigeix a la pàgina de panell
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
            # Comprovar si el perfil ja existeix, sinó, crear-lo
            if not hasattr(user, 'profile'):
                UserProfile.objects.create(user=user)
            return redirect('dashboard')  # O redirigeix a la pàgina que desitgis
        else:
            return render(request, 'login.html', {'form': form})
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
    # Usar carpeta HTML específica per l'usuari
    user_htmls_folder = get_user_folder_path(request.user.username, 'htmls')
    file_path = os.path.join(user_htmls_folder, filename)

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
    username = request.user.username

    # Carpeta HTML per a l’usuari
    user_htmls_folder = get_user_folder_path(username, 'htmls')
    HTML_OUTPUT = user_htmls_folder
    os.makedirs(HTML_OUTPUT, exist_ok=True)

    # Carpeta Clusters
    clusters_folder = get_user_folder_path(username, 'clusters')
    clustering_files = [f for f in os.listdir(clusters_folder) if f.endswith('_clustering_summary.json')] if os.path.exists(clusters_folder) else []
    
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
        subscription = request.user.userprofile.subscription
    except UserProfile.DoesNotExist:
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
            base_name = os.path.splitext(filename)[0]
            html_path = os.path.join(HTML_OUTPUT, base_name + ".html")
            json_path = os.path.join(HTML_OUTPUT, base_name + ".json")
            if os.path.exists(html_path): os.remove(html_path)
            if os.path.exists(json_path): os.remove(json_path)
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

            file_base = communication_name.replace(' ', '_').lower()
            html_filename = f"{file_base}.html"
            json_filename = f"{file_base}.json"

            html_path = os.path.join(HTML_OUTPUT, html_filename)
            json_path = os.path.join(HTML_OUTPUT, json_filename)

            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(rendered_html)

            metadata = {
                "name": communication_name,
                "prompt": prompt,
                "template": selected_template,
                "dataset": dataset_name if dataset_name else None,
                "cluster": cluster_id if cluster_id else None,
                "created_at": now().isoformat()
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

        except FileNotFoundError:
            return render(request, template_name, {
                'generated_content': "❌ Plantilla no trobada.",
                'generated_metadata': [],
                'generated_html_files': [],
                'subscription': subscription,
                'available_datasets': available_datasets
            })

        return render(request, template_name, {
            'generated_content': rendered_html,
            'generated_metadata': [metadata],
            'generated_html_files': [html_filename],
            'subscription': subscription,
            'available_datasets': available_datasets
        })

    # GET: mostrar comunicacions existents
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

    return render(request, template_name, {
        'generated_content': None,
        'generated_html_files': generated_html_files,
        'generated_metadata': generated_metadata,
        'available_datasets': available_datasets,
        'subscription': subscription
    })


@login_required
def clusters(request):
    from app.scripts.gen_clusters import full_clustering_pipeline
    from app.scripts.gen_clusters_desc import generar_descripcions_clusters

    user_folder = get_user_folder_path(request.user.username)

    message = None
    selected_dataset = request.GET.get('dataset', None)

    if request.method == 'POST':
        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            n_clusters = int(request.POST.get('n_clusters', 0))

            # Guardar fitxer temporalment
            user_data_dir = os.path.join(user_folder, 'data_files')
            os.makedirs(user_data_dir, exist_ok=True)
            file_path = os.path.join(user_data_dir, uploaded_file.name)

            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            try:
                # Generar clusters amb nou pipeline
                cluster_summary, json_path = full_clustering_pipeline(file_path, request.user.username, n_clusters)

                # Extreure nom de taula del path del JSON
                json_filename = os.path.basename(json_path)
                table_name = json_filename.replace('_clustering_summary.json', '')

                # Generar descripcions
                clusters_descriptions = generar_descripcions_clusters(request.user.username, table_name)

                message = "✅ Clusters generats correctament!"
                selected_dataset = table_name

            except Exception as e:
                print(f"Error generant clusters: {str(e)}")  # Log per debug
                message = "❌ Error processant el fitxer. Si us plau, revisa el format i torna-ho a intentar."

        elif request.POST.get('delete_dataset'):
            dataset_to_delete = request.POST.get('delete_dataset')
            try:
                # Eliminar fitxers relacionats amb aquest dataset
                files_to_delete = [
                    f'{dataset_to_delete}_clustering_summary.json',
                    f'{dataset_to_delete}_clustering_descriptions.json'
                ]

                for file_name in files_to_delete:
                    file_path = os.path.join(user_folder, file_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)

                message = f"✅ Dataset '{dataset_to_delete}' eliminat correctament!"

                # Si estem visualitzant el dataset eliminat, netegem la selecció
                if selected_dataset == dataset_to_delete:
                    selected_dataset = None

            except Exception as e:
                print(f"Error eliminant dataset: {str(e)}")  # Log per debug
                message = "❌ Error eliminant el dataset. Si us plau, torna-ho a intentar."

    # Obtenir tots els datasets disponibles (màxim 3) des de la carpeta clusters
    clusters_folder = get_user_folder_path(request.user.username, 'clusters')
    clustering_files = [f for f in os.listdir(clusters_folder) if f.endswith('_clustering_summary.json')] if os.path.exists(clusters_folder) else []
    available_datasets = []

    for file_name in clustering_files:
        table_name = file_name.replace('_clustering_summary.json', '')
        clusters_summary_path = os.path.join(str(clusters_folder), file_name)

        # Obtenir data de modificació
        mod_time = os.path.getmtime(clusters_summary_path)

        # Carregar dades del dataset
        with open(clusters_summary_path, 'r', encoding='utf-8') as f:
            clusters_data = json.load(f)

        # Carregar descripcions si existeixen
        clusters_desc_path = os.path.join(str(clusters_folder), f'{table_name}_clustering_descriptions.json')
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
        profile = UserProfile.objects.get(user=user)
        subscription = profile.get_subscription_display()  # Obtenim el valor llegible de la subscripció
    except UserProfile.DoesNotExist:
        subscription = 'Free'  # Per defecte si el perfil no existeix

    return render(request, 'user_details.html', {
        'user': user,
        'subscription': subscription,  # Pasem el valor llegible de la subscripció a la plantilla
    })