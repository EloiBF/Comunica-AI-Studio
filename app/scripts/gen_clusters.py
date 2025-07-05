# GEN_CLUSTERS_IMPROVED.PY

import os
import sys
import pandas as pd
import numpy as np
import sqlite3
import re
import shutil
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the environment where this script will run
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))

# Import necessary functions
from app.scripts.utils import prompt_AI, get_clean_table_name, get_user_db_path, get_user_folder_path

# Constants
DATABASE_DIR = 'app/users/'
DEFAULT_DB_NAME = 'user_database'
MAX_RETRIES = 3
SUPPORTED_EXTENSIONS = ['.xlsx', '.xls', '.csv']
CSV_SEPARATORS = [',', ';', '\t', '|']

@dataclass
class ClusteringConfig:
    """Configuration for clustering parameters"""
    n_clusters: int = 0
    min_k: int = 3
    max_k: int = 6
    max_card: int = 20
    top_n: int = 10
    cat_threshold: float = 2.0
    decimals: int = 1
    max_retries: int = MAX_RETRIES

class ClusteringError(Exception):
    """Custom exception for clustering errors"""
    pass

class DataProcessor:
    """Handles data loading and preprocessing"""
    
    @staticmethod
    def load_file_to_dataframe(file_path: str) -> pd.DataFrame:
        """Load file data into a pandas DataFrame"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {ext}")
        
        try:
            if ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif ext == '.csv':
                df = DataProcessor._load_csv_with_separator_detection(file_path)
            
            return DataProcessor._clean_dataframe(df)
            
        except Exception as e:
            raise ClusteringError(f"Error loading file {file_path}: {e}")
    
    @staticmethod
    def _load_csv_with_separator_detection(file_path: str) -> pd.DataFrame:
        """Load CSV with automatic separator detection"""
        for sep in CSV_SEPARATORS:
            try:
                df = pd.read_csv(file_path, sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
        
        # If no separator worked, try default
        return pd.read_csv(file_path)
    
    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare DataFrame"""
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Remove empty columns
        df = df.loc[:, df.columns.notna()]
        df = df.loc[:, df.columns != '']
        
        # Clean string columns
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).str.strip()
        
        return df

class SQLiteManager:
    """Manages SQLite database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def create_table_from_dataframe(self, df: pd.DataFrame, table_name: str):
        """Create table from DataFrame"""
        df.to_sql(table_name, self.conn, if_exists='replace', index=False)
        self.conn.commit()
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute query and return DataFrame"""
        return pd.read_sql_query(query, self.conn)
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive table information"""
        return {
            'records': self._get_sample_records(table_name),
            'numeric_stats': self._get_numeric_statistics(table_name),
            'categorical_catalog': self._get_categorical_catalog(table_name)
        }
    
    def _get_sample_records(self, table_name: str, n: int = 5) -> str:
        """Get sample records with headers"""
        self.cursor.execute(f"SELECT * FROM {table_name} LIMIT {n}")
        records = self.cursor.fetchall()
        headers = [desc[0] for desc in self.cursor.description]
        
        headers_text = ", ".join(headers)
        records_text = "\n".join([", ".join(map(str, r)) for r in records])
        return f"Table: {table_name}\nHeaders: {headers_text}\nRecords:\n{records_text}"
    
    def _get_numeric_statistics(self, table_name: str) -> str:
        """Get numeric column statistics"""
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        columns = self.cursor.fetchall()
        
        stats = {}
        for col in columns:
            col_name, col_type = col[1], col[2]
            if col_type.upper() in ["INTEGER", "REAL"]:
                try:
                    self.cursor.execute(f"""
                        SELECT COUNT(*), MIN({col_name}), MAX({col_name}), AVG({col_name})
                        FROM {table_name}
                    """)
                    count, minimum, maximum, mean = self.cursor.fetchone()
                    stats[col_name] = {
                        'count': count, 'min': minimum, 'max': maximum, 'mean': mean
                    }
                except Exception as e:
                    logger.warning(f"Error getting stats for {col_name}: {e}")
        
        result = f"Numeric statistics for {table_name}:\n"
        for col, stat in stats.items():
            result += f"{col}: count={stat['count']}, min={stat['min']}, max={stat['max']}, mean={stat['mean']:.2f}\n"
        
        return result
    
    def _get_categorical_catalog(self, table_name: str, max_elements: int = 30) -> str:
        """Get categorical column catalog"""
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        columns = self.cursor.fetchall()
        
        catalog = {}
        for col in columns:
            col_name, col_type = col[1], col[2]
            if col_type.upper() not in ["INTEGER", "REAL"]:
                try:
                    self.cursor.execute(f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table_name}"')
                    distinct_count = self.cursor.fetchone()[0]
                    
                    if distinct_count <= max_elements:
                        self.cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{table_name}"')
                        values = [row[0] for row in self.cursor.fetchall()]
                        catalog[col_name] = values
                except Exception as e:
                    logger.warning(f"Error getting catalog for {col_name}: {e}")
        
        result = "Categorical variables:\n"
        for col, values in catalog.items():
            result += f"{col}: {', '.join(map(str, values))}\n"
        
        return result

class AIPromptManager:
    """Manages AI prompts with robust error handling"""
    
    def __init__(self, max_retries: int = MAX_RETRIES):
        self.max_retries = max_retries
    
    def _retry_prompt(self, prompt_func, *args, **kwargs):
        """Generic retry mechanism for AI prompts"""
        for attempt in range(self.max_retries):
            try:
                result = prompt_func(*args, **kwargs)
                if result and str(result).strip():
                    return result
                logger.warning(f"Empty response from AI (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"AI prompt error (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        return None
    
    def get_column_selection_sql(self, table_info: Dict[str, Any]) -> Optional[str]:
        """Get SQL for column selection"""
        query = f"""
        You are an expert SQL code generator for clustering analysis using SQLite syntax.
        Select the most relevant columns for k-means clustering.
        Return ONLY a SELECT statement: SELECT col1, col2, col3 FROM table_name;
        Do NOT use functions or computed columns.
        
        Table information:
        {table_info['records']}
        {table_info['numeric_stats']}
        {table_info['categorical_catalog']}
        """
        
        return self._retry_prompt(self._clean_sql_response, prompt_AI(query))
    
    def get_important_columns(self, variables: List[str]) -> List[str]:
        """Get most important columns for clustering"""
        query = f"""
        You are a clustering expert for a travel agency.
        Select maximum 3 most relevant variables for customer segmentation.
        Return as a Python list: ["var1", "var2", "var3"]
        
        Variables: {variables}
        """
        
        response = self._retry_prompt(prompt_AI, query)
        if response:
            try:
                # Extract list from response
                import ast
                return ast.literal_eval(response.strip())
            except:
                # Fallback: extract quoted strings
                return re.findall(r'"([^"]*)"', response)
        
        return variables[:3]  # Fallback
    
    def _clean_sql_response(self, response: str) -> str:
        """Clean and validate SQL response"""
        if not response:
            return ""
        
        # Extract SELECT statements
        select_pattern = r'(SELECT.+?;)'
        matches = re.findall(select_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        return response.strip()

class ClusteringEngine:
    """Main clustering engine"""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
    
    def detect_optimal_clusters(self, X: np.ndarray) -> Tuple[int, str]:
        """Detect optimal number of clusters"""
        if self.config.n_clusters > 0:
            return self.config.n_clusters, "User specified"
        
        k_range = range(self.config.min_k, self.config.max_k + 1)
        
        # Method 1: Elbow method
        inertias = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        knee_locator = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
        if knee_locator.elbow:
            return knee_locator.elbow, "Elbow method"
        
        # Method 2: Silhouette score
        silhouette_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        return optimal_k, "Silhouette score"
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for clustering"""
        # Handle categorical variables
        df_processed = df.copy()
        
        # Process high cardinality categoricals
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_processed[col].nunique() > self.config.max_card:
                continue
            
            # Keep top N categories, group rest as 'Other'
            top_categories = df_processed[col].value_counts().nlargest(self.config.top_n).index
            df_processed[col] = df_processed[col].apply(
                lambda x: x if x in top_categories else 'Other'
            )
        
        # Get numeric columns
        numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
        
        # Create dummy variables for categoricals
        categorical_dummies = pd.get_dummies(
            df_processed[categorical_cols], 
            drop_first=False
        ) if len(categorical_cols) > 0 else pd.DataFrame(index=df_processed.index)
        
        # Scale numeric features
        scaler = RobustScaler()
        X_numeric = scaler.fit_transform(df_processed[numeric_cols]) if numeric_cols else np.zeros((len(df_processed), 0))
        
        # Combine features
        X = np.concatenate([X_numeric, categorical_dummies.values], axis=1)
        feature_names = numeric_cols + list(categorical_dummies.columns)
        
        return X, feature_names
    
    def apply_clustering(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int, str]:
        """Apply clustering to DataFrame"""
        X, feature_names = self.prepare_features(df)
        
        # Detect optimal clusters
        n_clusters, method = self.detect_optimal_clusters(X)
        
        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_result = df.copy()
        df_result['Cluster'] = kmeans.fit_predict(X)
        
        # Check cluster balance
        cluster_sizes = df_result['Cluster'].value_counts(normalize=True)
        min_size = cluster_sizes.min()
        
        if min_size < 0.05:  # Less than 5%
            logger.info("Applying balanced clustering due to imbalanced clusters")
            df_result['Cluster'] = self._balanced_kmeans(X, n_clusters)
        
        logger.info(f"Clustering completed using {method} with {n_clusters} clusters")
        return df_result, n_clusters, method
    
    def _balanced_kmeans(self, X: np.ndarray, n_clusters: int, max_iter: int = 100) -> np.ndarray:
        """Balanced K-means implementation"""
        n_samples = X.shape[0]
        size_min = n_samples // n_clusters
        size_max = size_min + (1 if n_samples % n_clusters > 0 else 0)
        
        # Initialize centroids
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        centroids = kmeans.fit(X).cluster_centers_
        
        labels = np.full(n_samples, -1)
        
        for iteration in range(max_iter):
            # Calculate distances
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            
            # Sort samples by minimum distance to any centroid
            idx_sorted = np.argsort(np.min(distances, axis=1))
            
            # Assign samples to clusters with size constraints
            assigned_counts = [0] * n_clusters
            labels.fill(-1)
            
            for i in idx_sorted:
                centroid_order = np.argsort(distances[i])
                for c in centroid_order:
                    if assigned_counts[c] < size_max:
                        labels[i] = c
                        assigned_counts[c] += 1
                        break
            
            # Update centroids
            new_centroids = np.array([
                X[labels == c].mean(axis=0) if np.any(labels == c) else centroids[c]
                for c in range(n_clusters)
            ])
            
            # Check convergence
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            
            centroids = new_centroids
        
        return labels

class ClusterAnalyzer:
    """Analyzes clustering results"""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
    
    def generate_cluster_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive cluster summary"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col.lower() != 'cluster']
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Global statistics
        global_numeric_means = {col: float(df[col].mean()) for col in numeric_cols}
        global_categorical_dist = {
            col: df[col].value_counts(normalize=True).to_dict() 
            for col in categorical_cols
        }
        
        summary = {}
        n_clusters = df['Cluster'].nunique()
        
        for cluster_id in range(n_clusters):
            cluster_data = df[df['Cluster'] == cluster_id]
            
            cluster_info = {
                'cluster_id': int(cluster_id),
                'size': int(len(cluster_data)),
                'percentage': float(round(len(cluster_data) / len(df) * 100, self.config.decimals))
            }
            
            # Numeric features analysis
            numeric_analysis = self._analyze_numeric_features(
                cluster_data, numeric_cols, global_numeric_means
            )
            if numeric_analysis:
                cluster_info['numeric_features'] = numeric_analysis
            
            # Categorical features analysis
            categorical_analysis = self._analyze_categorical_features(
                cluster_data, categorical_cols, global_categorical_dist
            )
            if categorical_analysis:
                cluster_info['categorical_features'] = categorical_analysis
            
            summary[cluster_id] = cluster_info
        
        return summary
    
    def _analyze_numeric_features(self, cluster_data: pd.DataFrame, 
                                 numeric_cols: List[str], 
                                 global_means: Dict[str, float]) -> Dict[str, Any]:
        """Analyze numeric features for a cluster"""
        analysis = {}
        
        for col in numeric_cols:
            cluster_mean = float(cluster_data[col].mean())
            global_mean = global_means[col]
            
            # Only include if there's a meaningful difference
            if abs(cluster_mean - global_mean) >= 0.01:
                analysis[col] = {
                    'cluster_mean': float(round(cluster_mean, self.config.decimals)),
                    'global_mean': float(round(global_mean, self.config.decimals)),
                    'difference': float(round(cluster_mean - global_mean, self.config.decimals))
                }
        
        return analysis
    
    def _analyze_categorical_features(self, cluster_data: pd.DataFrame,
                                    categorical_cols: List[str],
                                    global_distributions: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze categorical features for a cluster"""
        analysis = {}
        
        for col in categorical_cols:
            cluster_dist = cluster_data[col].value_counts(normalize=True)
            global_dist = global_distributions[col]
            
            # Find categories with significant differences
            significant_categories = {}
            all_categories = set(cluster_dist.index).union(set(global_dist.keys()))
            
            for category in all_categories:
                cluster_pct = cluster_dist.get(category, 0) * 100
                global_pct = global_dist.get(category, 0) * 100
                
                if abs(cluster_pct - global_pct) >= self.config.cat_threshold:
                    significant_categories[category] = {
                        'cluster_percentage': float(round(cluster_pct, self.config.decimals)),
                        'global_percentage': float(round(global_pct, self.config.decimals)),
                        'difference': float(round(cluster_pct - global_pct, self.config.decimals))
                    }
            
            if significant_categories:
                analysis[col] = significant_categories
        
        return analysis

class ClusteringPipeline:
    """Main clustering pipeline"""
    
    def __init__(self, config: ClusteringConfig = None):
        self.config = config or ClusteringConfig()
        self.prompt_manager = AIPromptManager(self.config.max_retries)
        self.clustering_engine = ClusteringEngine(self.config)
        self.analyzer = ClusterAnalyzer(self.config)
    
    def run_full_pipeline(self, file_path: str, username: str) -> Tuple[Dict[str, Any], str]:
        """Run the complete clustering pipeline"""
        try:
            logger.info("Starting clustering pipeline")
            
            # Step 1: Load and prepare data
            df = DataProcessor.load_file_to_dataframe(file_path)
            table_name = get_clean_table_name(file_path)
            
            # Step 2: Save to SQLite
            db_path = get_user_db_path(username)
            with SQLiteManager(db_path) as db:
                db.create_table_from_dataframe(df, table_name)
                table_info = db.get_table_info(table_name)
                
                # Step 3: Feature selection (optional AI-assisted)
                try:
                    sql_select = self.prompt_manager.get_column_selection_sql(table_info)
                    if sql_select:
                        sql_select = sql_select.replace('table_name', table_name)
                        df_selected = db.execute_query(sql_select)
                        if not df_selected.empty:
                            df = df_selected
                        else:
                            logger.warning("AI column selection returned empty result, using original data")
                except Exception as e:
                    logger.warning(f"AI column selection failed: {e}. Using original data.")
            
            # Step 4: Apply clustering
            df_clustered, n_clusters, method = self.clustering_engine.apply_clustering(df)
            
            # Step 5: Generate analysis
            cluster_summary = self.analyzer.generate_cluster_summary(df_clustered)
            
            # Step 6: Save results
            json_path = self._save_results(username, table_name, df_clustered, cluster_summary)
            
            logger.info(f"Pipeline completed successfully. Results saved to {json_path}")
            return cluster_summary, json_path
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise ClusteringError(f"Clustering pipeline failed: {e}")
        
        finally:
            # Clean up uploaded files
            self._cleanup_files(username)
    
    def _save_results(self, username: str, table_name: str, 
                     df_clustered: pd.DataFrame, cluster_summary: Dict[str, Any]) -> str:
        """Save clustering results"""
        # Save to SQLite
        db_path = get_user_db_path(username)
        with SQLiteManager(db_path) as db:
            db.create_table_from_dataframe(df_clustered, f"{table_name}_clustered")
        
        # Save summary to JSON
        clusters_folder = get_user_folder_path(username, 'clusters')
        os.makedirs(clusters_folder, exist_ok=True)
        
        json_filename = f"{table_name}_clustering_summary.json"
        json_path = os.path.join(clusters_folder, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved: SQLite table '{table_name}_clustered', JSON '{json_filename}'")
        return json_path
    
    def _cleanup_files(self, username: str):
        """Clean up uploaded files"""
        try:
            user_data_dir = get_user_folder_path(username, 'data_files')
            if os.path.exists(user_data_dir):
                shutil.rmtree(user_data_dir)
                logger.info("Uploaded files cleaned up successfully")
        except Exception as e:
            logger.warning(f"Failed to cleanup files: {e}")

# Main execution functions
def full_clustering_pipeline(file_path: str, username: str, n_clusters: int = 0) -> Tuple[Dict[str, Any], str]:
    """Main entry point for clustering pipeline"""
    config = ClusteringConfig(n_clusters=n_clusters)
    pipeline = ClusteringPipeline(config)
    return pipeline.run_full_pipeline(file_path, username)

if __name__ == "__main__":
    # Example usage
    try:
        summary, json_path = full_clustering_pipeline("data.csv", "test_user")
        print(f"Clustering completed. Results: {json_path}")
    except ClusteringError as e:
        print(f"Clustering failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")