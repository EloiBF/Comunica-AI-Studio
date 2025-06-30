# Applying the requested changes to use centralized route functions from settings in the save_clustering_summary function.
import os, sys
import pandas as pd
import numpy as np
import sqlite3
import json
import re
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

# Definim entorn on s'executarà aquest script (com si fos el root)
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))

# Importem funcions necessàries
from app.scripts.utils import prompt_AI, get_clean_table_name, get_user_db_path, get_user_folder_path

# Constants
DATABASE_DIR = 'app/users/'
DEFAULT_DB_NAME = 'user_database'

def create_user_table_from_file(file_path, username):
    """
    Carrega un fitxer CSV/Excel a una base de dades SQLite de l'usuari
    i retorna el nom de la taula creada
    """
    db_path = get_user_db_path(username)
    conn = sqlite3.connect(db_path)
    table_name = get_clean_table_name(file_path)

    # Determinar format del fitxer
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif ext == '.csv':
        # Provar diferents separadors
        for sep in [',', ';', '\t', '|']:
            try:
                df = pd.read_csv(file_path, sep=sep)
                if df.shape[1] > 1:
                    break
            except Exception:
                continue
        else:
            raise ValueError("No s'ha pogut detectar el separador del CSV.")
    else:
        raise ValueError("Format de fitxer no suportat.")

    # Neteja bàsica del DataFrame
    df.columns = df.columns.str.strip()
    df = df.loc[:, df.columns.notna()]
    df = df.loc[:, df.columns != '']

    # Netejar strings
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype(str).str.strip()

    # Guardar a SQLite
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

    print(f"✅ Taula '{table_name}' creada correctament")
    return table_name

def get_table_info(username, table_name):
    """
    Obté informació de la taula per generar prompts d'IA
    """
    db_path = get_user_db_path(username)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Obtenir informació de columnes
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns_info = cursor.fetchall()

    # Obtenir una mostra de dades
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
    sample_data = cursor.fetchall()

    # Crear text descriptiu
    headers = [col[1] for col in columns_info]
    headers_text = ", ".join(headers)

    sample_text = ""
    for row in sample_data:
        sample_text += ", ".join([str(val) for val in row]) + "\n"

    conn.close()

    return {
        'headers': headers,
        'headers_text': headers_text,
        'sample_data': sample_text,
        'columns_info': columns_info
    }

def generate_clustering_table_sql(username, original_table_name):
    """
    Genera SQL amb IA per crear una taula de clustering modificada
    """
    table_info = get_table_info(username, original_table_name)

    clustering_table_name = f"{original_table_name}_clustering"

    prompt = f"""
Ets un expert en SQL i análisis de dades. 

Tens una taula anomenada '{original_table_name}' amb les següents columnes:
{table_info['headers_text']}

Mostra de dades:
{table_info['sample_data']}

Genera una sentència SQL CREATE TABLE per crear una nova taula anomenada '{clustering_table_name}' 
que contingui NOMÉS les columnes més rellevants per fer clustering (análisis de segments).

Regles:
1. Selecciona només columnes numèriques o categòriques útils per clustering
2. Evita IDs, noms o columnes identificatives
3. Inclou columnes que puguin revelar patrons de comportament
4. La sentència ha de ser: CREATE TABLE {clustering_table_name} AS SELECT ... FROM {original_table_name};
5. No afegeixis comentaris, només el SQL

Retorna només la sentència SQL:
"""

    sql_response = prompt_AI(prompt)

    # Netejar la resposta
    sql_clean = re.search(r'CREATE TABLE.*?;', sql_response, re.DOTALL | re.IGNORECASE)
    if sql_clean:
        return sql_clean.group(0).strip()
    else:
        raise ValueError("No s'ha pogut generar SQL vàlid")

def generate_feature_engineering_sql(username, clustering_table_name):
    """
    Genera SQL amb IA per modificar la taula de clustering amb noves columnes calculades
    """
    table_info = get_table_info(username, clustering_table_name)

    prompt = f"""
Ets un expert en feature engineering per clustering.

Tens una taula de clustering anomenada '{clustering_table_name}' amb columnes:
{table_info['headers_text']}

Mostra de dades:
{table_info['sample_data']}

Genera sentències SQL ALTER TABLE per afegir MÀXIM 2 noves columnes calculades útils per clustering.
Exemples: ràtios entre variables, diferències, transformacions matemàtiques.

Regles:
1. Màxim 2 columnes noves
2. Només columnes realment útils per segmentació
3. Utilitza funcions SQL estàndard
4. Format: ALTER TABLE {clustering_table_name} ADD COLUMN nom_columna AS (càlcul);
5. Si no és possible calcular columnes útils, retorna només: -- NO NEW COLUMNS --

Retorna només les sentències SQL (una per línia):
"""

    sql_response = prompt_AI(prompt)

    if "-- NO NEW COLUMNS --" in sql_response:
        print("🔍 No es poden generar noves columnes calculades")
        return None

    # Extreure sentències ALTER TABLE
    alter_statements = re.findall(r'ALTER TABLE.*?;', sql_response, re.DOTALL | re.IGNORECASE)
    if alter_statements:
        return '\n'.join(alter_statements)
    else:
        print("🔍 No s'han trobat sentències ALTER TABLE vàlides")
        return None

def execute_sql_statements(username, sql_statements):
    """
    Executa sentències SQL a la base de dades de l'usuari
    """
    if not sql_statements:
        return True

    db_path = get_user_db_path(username)
    conn = sqlite3.connect(db_path)

    try:
        # Dividir en sentències individuals si hi ha múltiples
        statements = [stmt.strip() for stmt in sql_statements.split(';') if stmt.strip()]

        for statement in statements:
            print(f"🔄 Executant: {statement[:100]}...")
            conn.execute(statement + ';')

        conn.commit()
        conn.close()
        print("✅ SQL executat correctament")
        return True

    except Exception as e:
        conn.close()
        print(f"❌ Error executant SQL: {e}")
        return False

def prepare_clustering_data(username, clustering_table_name):
    """
    Prepara les dades per al clustering des de la taula SQL
    """
    db_path = get_user_db_path(username)
    conn = sqlite3.connect(db_path)

    try:
        df = pd.read_sql_query(f"SELECT * FROM {clustering_table_name}", conn)
        conn.close()
        print(f"✅ Dades carregades: {df.shape[0]} files, {df.shape[1]} columnes")
        return df
    except Exception as e:
        conn.close()
        print(f"❌ Error carregant dades: {e}")
        return None

def handle_high_cardinality_categoricals(df, max_card=20, top_n=10):
    """Agrupa categories poc freqüents en 'Other' i exclou variables amb cardinalitat excessiva."""
    df_result = df.copy()
    categoricals = [col for col in df_result.select_dtypes(include=['object', 'category']).columns]
    cols_to_use = []

    for col in categoricals:
        num_unique = df_result[col].nunique(dropna=False)
        if num_unique > max_card:
            continue
        top_cats = df_result[col].value_counts().nlargest(top_n).index
        df_result[col] = df_result[col].apply(lambda x: x if x in top_cats else 'Other')
        cols_to_use.append(col)

    return df_result, cols_to_use

def detect_optimal_clusters(X, min_k=2, max_k=6):
    """Detecta el nombre òptim de clusters"""
    if len(X) < min_k:
        return min_k

    # Mètode del colze
    costs = []
    for k in range(min_k, min(max_k + 1, len(X))):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X)
        costs.append(kmeans.inertia_)

    if len(costs) > 1:
        kl = KneeLocator(range(min_k, min_k + len(costs)), costs, curve="convex", direction="decreasing")
        if kl.elbow is not None:
            return max(min_k, min(kl.elbow, max_k))

    # Si no hi ha colze clar, usar silhouette
    silhouette_scores = []
    for k in range(min_k, min(max_k + 1, len(X))):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
        else:
            score = -1
        silhouette_scores.append(score)

    return np.argmax(silhouette_scores) + min_k

def perform_clustering(df, n_clusters=0):
    """
    Realitza el clustering sobre el DataFrame preparat
    """
    # Processar variables categòriques
    df_proc, categoricals = handle_high_cardinality_categoricals(df)
    numerics = df_proc.select_dtypes(include=['number']).columns.tolist()

    # Crear variables dummy per categòriques
    df_cat = pd.get_dummies(df_proc[categoricals], drop_first=False) if categoricals else pd.DataFrame(index=df_proc.index)

    # Escalar variables numèriques
    scaler = RobustScaler()
    X_num = scaler.fit_transform(df_proc[numerics]) if numerics else np.zeros((len(df_proc), 0))

    # Combinar tots els features
    X = np.concatenate([X_num, df_cat.values], axis=1) if df_cat.shape[1] > 0 else X_num

    # Determinar nombre de clusters
    if n_clusters == 0:
        n_clusters = detect_optimal_clusters(X)
        print(f"🎯 Nombre òptim de clusters detectat: {n_clusters}")

    # Aplicar KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_result = df.copy()
    df_result['cluster'] = kmeans.fit_predict(X)

    # Mostrar distribució de clusters
    cluster_counts = df_result['cluster'].value_counts(normalize=True).sort_index()
    cluster_percentages = (cluster_counts * 100).round(2)

    print("\n📊 Distribució de clusters:")
    for cl, perc in cluster_percentages.items():
        print(f"  Cluster {cl}: {perc}%")

    return df_result, n_clusters

def generate_cluster_summary(df, n_clusters):
    """
    Genera un resum dels clusters
    """
    numerics = df.select_dtypes(include=['number']).columns.tolist()
    numerics = [col for col in numerics if col != 'cluster']  # Excloure la columna cluster
    categoricals = [col for col in df.select_dtypes(include=['object', 'category']).columns]

    cluster_summary = {}

    # Estadístiques globals
    global_num_mean = {col: float(df[col].mean()) for col in numerics}
    global_cat_dist = {col: df[col].value_counts(normalize=True).to_dict() for col in categoricals}

    for i in range(n_clusters):
        group = df[df['cluster'] == i]
        if len(group) == 0:
            continue

        cluster_info = {
            'cluster_id': int(i),
            'size': int(len(group)),
            'percentage': float(round(len(group) / len(df) * 100, 1))
        }

        # Estadístiques numèriques
        if numerics:
            num_stats = {}
            for col in numerics:
                cluster_mean = float(group[col].mean())
                global_mean = global_num_mean[col]
                if abs(cluster_mean - global_mean) >= 0.01:
                    num_stats[col] = {
                        'cluster_mean': round(cluster_mean, 2),
                        'global_mean': round(global_mean, 2),
                        'difference': round(cluster_mean - global_mean, 2)
                    }
            if num_stats:
                cluster_info['numeric_features'] = num_stats

        # Estadístiques categòriques
        if categoricals:
            cat_stats = {}
            for col in categoricals:
                cluster_dist = group[col].value_counts(normalize=True)
                global_dist = global_cat_dist[col]

                significant_categories = {}
                for cat in cluster_dist.index:
                    cluster_pct = cluster_dist[cat] * 100
                    global_pct = global_dist.get(cat, 0) * 100
                    if abs(cluster_pct - global_pct) >= 5:  # Diferència significativa
                        significant_categories[cat] = {
                            'cluster_pct': round(cluster_pct, 1),
                            'global_pct': round(global_pct, 1),
                            'difference': round(cluster_pct - global_pct, 1)
                        }

                if significant_categories:
                    cat_stats[col] = significant_categories

            if cat_stats:
                cluster_info['categorical_features'] = cat_stats

        cluster_summary[f"cluster_{i}"] = cluster_info

    return cluster_summary

def save_clustering_summary(username, clustering_summary, table_name):
    """
    Guarda el resum de clustering com a JSON a la carpeta de l'usuari
    """
    from django.conf import settings
    user_clusters_folder = settings.get_user_clusters_dir(username)
    summary_path = os.path.join(str(user_clusters_folder), f'{table_name}_clustering_summary.json')

def save_clustering_results(username, table_name, df_clustered, cluster_summary):
    """
    Guarda els resultats del clustering
    """
    user_folder = get_user_folder_path(username)
    clusters_folder = get_user_folder_path(username, 'clusters')

    # Guardar DataFrame amb clusters a SQLite
    db_path = get_user_db_path(username)
    conn = sqlite3.connect(db_path)
    df_clustered.to_sql(f"{table_name}_clustering_results", conn, if_exists='replace', index=False)
    conn.close()

    # Guardar resum en JSON a la carpeta clusters
    json_filename = f"{table_name}_clustering_summary.json"
    json_path = os.path.join(clusters_folder, json_filename)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_summary, f, indent=2, ensure_ascii=False)

    print(f"✅ Resultats guardats:")
    print(f"   - Taula SQL: {table_name}_clustering_results")
    print(f"   - Resum JSON: {json_filename}")

    return json_path

def full_clustering_pipeline(file_path, username, n_clusters=0):
    """
    Pipeline complet de clustering amb generació SQL amb IA
    """
    print("🚀 Iniciant pipeline de clustering...")

    try:
        # 1. Crear taula original
        print("\n📁 Pas 1: Carregant fitxer a SQLite...")
        original_table_name = create_user_table_from_file(file_path, username)

        # 2. Generar taula de clustering amb IA
        print("\n🤖 Pas 2: Generant taula de clustering amb IA...")
        clustering_sql = generate_clustering_table_sql(username, original_table_name)
        print(f"SQL generat: {clustering_sql}")

        if not execute_sql_statements(username, clustering_sql):
            raise Exception("Error executant SQL de taula de clustering")

        clustering_table_name = f"{original_table_name}_clustering"

        # 3. Feature engineering amb IA
        print("\n🔧 Pas 3: Feature engineering amb IA...")
        feature_sql = generate_feature_engineering_sql(username, clustering_table_name)
        if feature_sql:
            print(f"Feature engineering SQL: {feature_sql}")
            execute_sql_statements(username, feature_sql)

        # 4. Carregar dades per clustering
        print("\n📊 Pas 4: Preparant dades per clustering...")
        df = prepare_clustering_data(username, clustering_table_name)
        if df is None:
            raise Exception("Error carregant dades de clustering")

        # 5. Realizar clustering
        print("\n🎯 Pas 5: Executant clustering...")
        df_clustered, n_clusters_used = perform_clustering(df, n_clusters)

        # 6. Generar resum
        print("\n📋 Pas 6: Generant resum de clusters...")
        cluster_summary = generate_cluster_summary(df_clustered, n_clusters_used)

        # 7. Guardar resultats
        print("\n💾 Pas 7: Guardant resultats...")
        json_path = save_clustering_results(username, original_table_name, df_clustered, cluster_summary)

        print(f"\n🎉 Pipeline completat exitosament!")
        print(f"📄 Resum guardat a: {json_path}")

        return cluster_summary, json_path

    except Exception as e:
        print(f"\n❌ Error en el pipeline: {str(e)}")
        raise e

# Funció per mantenir compatibilitat amb el codi existent
def full_clustering_flow(file_path, username, n_clusters=0):
    """Wrapper per compatibilitat amb el codi existent"""
    return full_clustering_pipeline(file_path, username, n_clusters)