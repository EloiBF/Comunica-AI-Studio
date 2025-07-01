from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Filtre personalitzat per obtenir un valor d'un diccionari per la seva clau."""
    return dictionary.get(key)

@register.filter
def lookup(dictionary, key):
    """Buscar un valor en el diccionari per la seva clau."""
    return dictionary.get(key)

@register.filter
def get_cluster_data(clusters_data, cluster_id):
    """Obtenir dades d'un cluster específic."""
    cluster_key = f"cluster_{cluster_id}"
    return clusters_data.get(cluster_key, {})

@register.filter
def get_cluster_desc(descriptions, cluster_id):
    """Obtenir descripció d'un cluster específic."""
    cluster_key = f"cluster_{cluster_id}"
    return descriptions.get(cluster_key, {})

@register.filter
def extract_cluster_id(cluster_key):
    """Extreure el número de cluster de la clau 'cluster_X'."""
    if cluster_key.startswith('cluster_'):
        return cluster_key.split('_')[1]  # Retorna només l'ID del cluster
    return cluster_key