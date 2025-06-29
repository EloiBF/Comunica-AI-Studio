import os, sys
import pandas as pd
import numpy as np
import sqlite3
import requests
import json
import re
import functools
import time
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from kneed import KneeLocator
import groq  
from sklearn.metrics import silhouette_score
import json
from pathlib import Path

# Definim entorn on s'executarà aquest script (com si fos el root)
base_dir = Path(__file__).resolve().parent.parent.parent  # Aquí, puja un nivell més alt per arribar a l'arrel
sys.path.append(str(base_dir))

# Importem funcions necessàries
from app.scripts.utils import prompt_AI, get_clean_table_name, get_user_db_path, get_user_folder_path



# Configura les rutes de les dades i arxius
DATA_DIR = 'input/data_example/'
DATABASE_DIR = 'users/data/'
DEFAULT_DB_NAME = 'user_database'

def get_user_db_path(username, db_name=DEFAULT_DB_NAME):
    """
    Retorna el camí de la base de dades per l'usuari
    """
    return os.path.join(DATABASE_DIR, f'{username}_{db_name}.db')

def load_user_file_to_sqlite(file_path, username):
    """
    Carrega un fitxer CSV a una base de dades SQLite de l'usuari
    """
    conn = sqlite3.connect(get_user_db_path(username))
    df = pd.read_csv(file_path)
    df.to_sql('NEW_TABLE', conn, if_exists='replace', index=False)
    conn.close()
    return 'NEW_TABLE'

# --- CLEAN AI SQL RESPONSE ---

def clean_sql_statements(text):
    statements = re.findall(r"(ALTER TABLE.+?;|SELECT.+?;|CREATE VIEW.+?;|DROP VIEW.+?;)", text, re.DOTALL | re.IGNORECASE)
    return "\n".join([s.strip() for s in statements])

# --- SQLITE & DATA UTILITIES ---

def load_user_file_to_sqlite(file_path, username):
    db_path = get_user_db_path(username)
    conn = sqlite3.connect(db_path)
    table_name = get_clean_table_name(file_path)

    # Determinar format
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif ext == '.csv':
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

    # Neteja bàsica
    df.columns = df.columns.str.strip()
    df = df.loc[:, df.columns.notna()]
    df = df.loc[:, df.columns != '']
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].str.strip()

    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    return table_name

def get_records_with_headers(cursor, table_name, n=5):
    cursor.execute(f"PRAGMA table_info({table_name})")
    headers = [info[1] for info in cursor.fetchall()]
    cursor.execute(f"SELECT * FROM {table_name} LIMIT {n}")
    records = cursor.fetchall()
    headers_text = ", ".join(headers)
    records_text = "\n".join([", ".join(map(str, r)) for r in records])
    return f"The table is '{table_name}'.\n{headers_text}\n{records_text}"

def get_numeric_statistics(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    statistics = {}
    for col in columns:
        col_name, col_type = col[1], col[2]
        if col_type.upper() in ["INTEGER", "REAL"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            cursor.execute(f"SELECT MIN({col_name}), MAX({col_name}), AVG({col_name}) FROM {table_name}")
            minimum, maximum, mean = cursor.fetchone()
            statistics[col_name] = {'count': count, 'min': minimum, 'max': maximum, 'mean': mean}
    result = f"Statistics of numeric columns in table '{table_name}':\n\n"
    for column, stats in statistics.items():
        result += (f"Column: {column}\n  Count: {stats['count']}\n  Min: {stats['min']}\n"
                   f"  Max: {stats['max']}\n  Mean: {stats['mean']}\n\n")
    return result

def get_categorical_catalog(cursor, table_name, max_elements=30):
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    catalog = {}
    for col in columns:
        col_name, col_type = col[1], col[2]
        if col_type.upper() not in ["INTEGER", "REAL"]:
            cursor.execute(f"SELECT COUNT(DISTINCT \"{col_name}\") FROM \"{table_name}\";")
            distinct_count = cursor.fetchone()[0]
            if distinct_count < max_elements:
                cursor.execute(f"SELECT DISTINCT \"{col_name}\" FROM \"{table_name}\";")
                values = [row[0] for row in cursor.fetchall()]
                catalog[col_name] = values
    result = "Catalog of categorical variables:\n"
    for column, values in catalog.items():
        result += f"{column}: {', '.join(map(str, values))}\n"
    return result

# --- AI SQL PROMPTS ---

def prompt_new_column_ideas(reg_txt, num_stats, cat_cat):
    query = (
        "As an expert in data analysis, examine the following table and suggest which new calculated columns could be useful for a clustering analysis. "
        "Only consider columns that are true transformations of existing data, such as: differences or ratios between numbers, duration between dates, etc. "
        "DO NOT include variable categorizations, recodings, or groupings. "
        "Do not include already existing or duplicated columns."
        "You must be restrictive and give a maximum of 3 new columns, the most useful and meaningful for the analysis. Never more."
        "If it is not possible to calculate any useful column, state this clearly. "
        "Return only a list of new column ideas, without SQL code or additional comments.\n"
        f"{reg_txt}\n{num_stats}\nCategorical variables:{cat_cat}"
    )
    return prompt_AI(query)

def prompt_sql_new_columns(ideas, reg_txt, num_stats, cat_cat):
    query = (
        "Based on these ideas for new calculated columns for clustering:\n"
        f"{ideas}\n"
        "Generate only the SQL code to implement them in the table using 'ALTER TABLE ... ADD COLUMN ... AS (...'. "
        "Do not return comments or explanations. "
        "If there is no possible new column, return only '-- NO NEW COLUMNS POSSIBLE --'.\n"
        f"{reg_txt}\n{num_stats}\nCategorical variables:{cat_cat}"
    )
    ai_response = prompt_AI(query)
    sql_statement = clean_sql_statements(ai_response)
    print("\n--- SQL GENERATED BY AI (Feature Engineering) ---")
    print(sql_statement)
    print("------------------------------------------------\n")
    return sql_statement

def prompt_feature_engineering(reg_txt, ideas, num_stats, cat_cat):
    print("\n--- IDEAS FOR NEW COLUMNS ---")
    print(ideas)
    print("--------------------------------\n")
    sql_statement = prompt_sql_new_columns(ideas, reg_txt, num_stats, cat_cat)
    # If the response is the signal that there are no new columns, return None
    if not sql_statement or '-- NO NEW COLUMNS POSSIBLE --' in sql_statement:
        return None
    return sql_statement

def prompt_column_selection_clustering(reg_txt, num_stats, cat_cat):
    query = (
        "You are an expert SQL code generator for clustering analysis. "
        "Select the most relevant columns of the table for a k-means clustering model, "
        "including numeric and categorical variables. Return only the SQL statement selecting those columns, for example: "
        "SELECT column1, column2, column3 FROM TABLA;"
        "Do not write comments or explanations, only the SQL query.\n"
        f"The table is as follows: {reg_txt}\n{num_stats}\nCategorical variables:{cat_cat}"
    )
    ai_response = prompt_AI(query)
    sql_statement = clean_sql_statements(ai_response)
    print("\n--- SQL GENERATED BY AI (Column Selection) ---")
    print(sql_statement)
    print("-----------------------------------------------------\n")
    return sql_statement

def prompt_important_columns(variables):
    query = (
        "You are an expert in clustering."
        "We are a travel agency and our goal is to achieve effective segmentation of our customers so we can provide personalized recommendations to them."
        "From a list of variables, you must select a maximum of three variables that you consider most relevant to achieve segmentation that meets our objective,such as the customer's age."
        "You should return the list of variables each written in quotation marks and separated by commas, all enclosed in square brackets, WITHOUT ANY OTHER CHARACTER \n"
        f"The list of variables is as follows:{variables}"
    )
    return prompt_AI(query)


# --- MAIN PIPELINE FLOW ---

def prepare_sqlite_and_view(file, username, max_retries=10):
    db_file = get_user_db_path(username)  # Asegurat que utilitzes el path correcte per la base de dades de l'usuari
    table_name = 'NEW_TABLE'
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Ara utilitzem la ruta del fitxer directament, passant el camí absolut o relatiu
    file_path = file  # Si 'file' ja és una ruta del fitxer, no cal canviar-ho
    
    # Càrrega de fitxer a SQLite
    table_name = load_user_file_to_sqlite(file_path, username)  # Utilitza la funció definida prèviament

    # 1. Informació inicial per a feature engineering
    reg_txt = get_records_with_headers(cursor, table_name)
    num_stats = get_numeric_statistics(cursor, table_name)
    cat_cat = get_categorical_catalog(cursor, table_name)

    # 2. Feature engineering (nous columns) amb millora de reintents
    attempts = 0
    last_error = ""
    while attempts < max_retries:
        if attempts == 0:
            # Primer intent: prompt normal
            ideas = prompt_new_column_ideas(reg_txt, num_stats, cat_cat)
        else:
            # Nous intents: inclou l'error en el prompt per ajudar la IA
            error_message = f"\nNote: The previous SQL code failed with the following SQLite error: {last_error}. Please avoid using unsupported functions or syntax."
            ideas = prompt_new_column_ideas(reg_txt, num_stats, cat_cat) + error_message

        sql_statement_columns = prompt_feature_engineering(reg_txt, ideas, num_stats, cat_cat)
        if not sql_statement_columns:
            print("No new columns generated, skipping this step.")
            break
        
        # Validació abans d'executar
        try:
            cursor.executescript(sql_statement_columns)
            conn.commit()
            print("Feature engineering executed successfully.")
            break  # Exit si l'operació és exitosa
        except Exception as e:
            last_error = str(e)
            attempts += 1
            print(f"Error executing feature engineering SQL (attempt {attempts}/{max_retries}): {e}")
            if attempts >= max_retries:
                print("Permanent failure executing feature engineering SQL after several attempts.")
                break  # O raise, segons la preferència

    # Continua el procés per aplicar altres operacions després...
    # ...
    conn.close()
    return db_file


def handle_high_cardinality_categoricals(df, max_card=20, top_n=10):
    """Groups infrequent categories into 'Other' and excludes variables with excessive cardinality."""
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

def read_clustering_data(db_file='data/base_sql.db'):
    db_file = db_file  # ← Aquí
    conn = sqlite3.connect(db_file)
    try:
        try:
            df = pd.read_sql_query("SELECT * FROM clustering_table;", conn)
            print("Leído desde la vista clustering_table.")
        except Exception as e:
            print(f"clustering_table no existe o no se puede leer ({e}), usando NEW_TABLE.")
            df = pd.read_sql_query("SELECT * FROM NEW_TABLE;", conn)
            print("Leído desde la tabla NEW_TABLE.")
    finally:
        conn.close()
    return df

def detect_columns(df):
    categoricals = [col for col in df.select_dtypes(include=['object', 'category']).columns 
                   if df[col].nunique() < min(20, 0.5*len(df))]
    numerics = df.select_dtypes(include=['number']).columns.tolist()
    return numerics, categoricals

def balanced_kmeans(X, n_clusters, max_iter=100):
    n_samples = X.shape[0]
    size_min = n_samples // n_clusters
    size_max = size_min + (n_samples % n_clusters > 0)
    # Inicialización de centroides
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    centroids = kmeans.fit(X).cluster_centers_
    labels = np.full(n_samples, -1)
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        idx_sorted = np.argsort(np.min(distances, axis=1))
        assigned = [0]*n_clusters
        labels.fill(-1)
        for i in idx_sorted:
            centroid_order = np.argsort(distances[i])
            for c in centroid_order:
                if assigned[c] < size_max:
                    labels[i] = c
                    assigned[c] += 1
                    break
        new_centroids = np.array([X[labels == c].mean(axis=0) if np.any(labels==c) else centroids[c] for c in range(n_clusters)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels

def assign_clusters_df(df, n_clusters=0, max_card=20, top_n=10, min_k=3, max_k=6):
    df_proc, categoricals = handle_high_cardinality_categoricals(df, max_card=max_card, top_n=top_n)
    numerics = df_proc.select_dtypes(include=['number']).columns.tolist()
    df_cat = pd.get_dummies(df_proc[categoricals], drop_first=False) if categoricals else pd.DataFrame(index=df_proc.index)
    scaler = RobustScaler()
    X_num = scaler.fit_transform(df_proc[numerics]) if numerics else np.zeros((len(df_proc), 0))
    X = np.concatenate([X_num, df_cat.values], axis=1) if df_cat.shape[1] > 0 else X_num

    feature_names = numerics + list(df_cat.columns)
    vars_importantes = prompt_important_columns(feature_names)
    importantes_idx = [i for i, name in enumerate(feature_names) if name in vars_importantes or any(name.startswith(var + "_") for var in vars_importantes)]
    if importantes_idx:
        X_importantes = X[:, importantes_idx]
        feature_names_dup = [name + "_dup" for name in feature_names]
        X = np.concatenate([X, X_importantes], axis=1)
        feature_names = feature_names + feature_names_dup

    # Selección automática del número de clústers dentro del rango [min_k, max_k]
    n_clusters_used = n_clusters
    metodo_usado = ""
    if n_clusters == 0:
        # 1. Método del codo (KneeLocator sobre inercia)
        costs = []
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
            kmeans.fit(X)
            costs.append(kmeans.inertia_)
        kl = KneeLocator(range(min_k, max_k + 1), costs, curve="convex", direction="decreasing")
        elbow = kl.elbow
        if elbow is not None:
            n_clusters_used = max(min_k, min(elbow, max_k))
            metodo_usado = "Método del codo (inercia)"
        else:
            # 2. Si no hay codo claro, usamos coeficiente de silueta
            silhouette_scores = []
            for k in range(min_k, max_k + 1):
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
                labels = kmeans.fit_predict(X)
                if len(set(labels)) > 1:
                    score = silhouette_score(X, labels)
                else:
                    score = -1
                silhouette_scores.append(score)
            n_clusters_used = np.argmax(silhouette_scores) + min_k
            metodo_usado = "Coeficiente de silueta"
        print(f"El número óptimo de clústers es {n_clusters_used} (Método utilizado: {metodo_usado})")
    else:
        n_clusters_used = n_clusters
        metodo_usado = "Valor especificado por el usuario"

    # KMeans normal
    kmeans = KMeans(n_clusters=n_clusters_used, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    cluster_counts = df['Cluster'].value_counts(normalize=True).sort_index()
    cluster_percentages = (cluster_counts * 100).round(2)

    print("\nPorcentaje de registros por clúster:")
    for cl, perc in cluster_percentages.items():
        print(f"Cluster {cl}: {perc}%")

    min_percentage = cluster_percentages.min()
    if min_percentage < 5:
        print("\nAl menos un clúster tiene menos del 5% de los registros. Se utilizará Balanced KMeans.")
        labels = balanced_kmeans(X, n_clusters_used)
        df['Cluster'] = labels
        # Recalcular los porcentajes
        cluster_counts = df['Cluster'].value_counts(normalize=True).sort_index()
        cluster_percentages = (cluster_counts * 100).round(2)
        print("\nNuevo porcentaje de registros por clúster (balanceado):")
        for cl, perc in cluster_percentages.items():
            print(f"Cluster {cl}: {perc}%")
    else:
        print("\nNo es necesario balancear los clústers.")

    print(f"\nMétodo utilizado para determinar el número de clústers: {metodo_usado}")

    return df, n_clusters_used


def generate_clusters_dict(
    df, 
    n_clusters=4,
    top_n_cat=2,
    cat_threshold=2,
    decimals=1
):
    numerics, categoricals = detect_columns(df)
    segments_dict = {}
    global_num_mean = {col: float(df[col].mean()) for col in numerics}
    global_cat_dist = {col: df[col].value_counts(normalize=True).to_dict() for col in categoricals}
    for i in range(n_clusters):
        group = df[df['Cluster'] == i]
        if len(group) == 0:
            continue
        cluster_info = {
            'Cluster': int(i),
            'Size': int(len(group)),
            'Share': float(round(len(group) / len(df) * 100, decimals))
        }
        num_stats = {}
        for col in numerics:
            if col.lower() == "cluster":
                continue
            clus_mean = float(group[col].mean())
            global_mean = global_num_mean[col]
            diff = abs(clus_mean - global_mean)
            if diff >= 0.01:
                num_stats[col] = {
                    'c': float(round(clus_mean, decimals)),
                    'g': float(round(global_mean, decimals))
                }
        if num_stats:
            cluster_info['num'] = num_stats
        cat_stats = {}
        for col in categoricals:
            clus_counts = group[col].value_counts(normalize=True)
            global_counts = global_cat_dist[col]
            values = set(clus_counts.index).union(global_counts.keys())
            diffs = [
                (v, abs(clus_counts.get(v, 0)*100 - global_counts.get(v, 0)*100))
                for v in values
            ]
            diffs.sort(key=lambda x: x[1], reverse=True)
            selected = []
            for v, diff in diffs:
                if diff >= cat_threshold or len(selected) < top_n_cat:
                    selected.append(v)
            val_stats = {
                v: {
                    'c': float(round(clus_counts.get(v, 0)*100, decimals)),
                    'g': float(round(global_counts.get(v, 0)*100, decimals))
                }
                for v in selected
            }
            cat_stats[col] = val_stats
        if cat_stats:
            cluster_info['cat'] = cat_stats
        segments_dict[i] = cluster_info
    return segments_dict

def full_clustering_flow(file, n_clusters=0):
    print("=== 1. prepare_sqlite_and_view ===")
    db_file = prepare_sqlite_and_view(file)
    print("=== 2. read_clustering_data ===")
    df = read_clustering_data(db_file)
    print("=== 3. assign_clusters_df ===")
    df_clusters, n_clusters_used = assign_clusters_df(df, n_clusters)
    print("=== 4. generate_clusters_dict ===")
    cluster_summary = generate_clusters_dict(df_clusters, n_clusters_used)

    # Guardar la columna 'Cluster' en la tabla NEW_TABLE
    print("=== 5. Guardando nueva tabla en SQLite ===")
    conn = sqlite3.connect(db_file)
    df_clusters = df_clusters.rename(columns={'Cluster': 'cluster'})
    df_clusters.to_sql('NEW_TABLE', conn, if_exists='replace', index=False)
    conn.close()

    print("=== 6. Clustering flow terminado ===")
    print(cluster_summary)
    return cluster_summary

full_clustering_flow('input/data_example/clients_variats.csv', 0)
