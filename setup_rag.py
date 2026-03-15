"""setup_rag.py — One-time RAG setup

Build a ChromaDB vector index for recipe retrieval.

MVP-friendly data strategy (proposal-aligned):
- Tier-1: Food.com subset (20K–50K) if you provide CSV files locally
- Fallback: built-in small sample recipes (so the demo always runs)

How to provide Food.com data:
1) Download Food.com data from Kaggle.
2) Put files at:
   - chefcoach/data/RAW_recipes.csv (required for Food.com indexing)
   - chefcoach/data/RAW_interactions.csv (optional, for ratings enrichment)
3) Run: python setup_rag.py

Usage:
  python setup_rag.py

Environment variables:
- RECIPE_SUBSET_SIZE: number of rows to ingest from RAW_recipes.csv (default: 20000)
- INTERACTIONS_CHUNK_SIZE: chunk size when aggregating interactions ratings (default: 300000)
- CHROMA_DB_PATH: where to store the Chroma DB (default: ./chroma_db)
- EMBEDDING_PROVIDER / EMBEDDING_MODEL: see modules/rag_retriever.py
"""

from __future__ import annotations

import ast
import json
import os
import sys
from typing import List, Dict, Optional, Set, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SUBSET_SIZE = int(os.getenv("RECIPE_SUBSET_SIZE", "20000"))
INTERACTIONS_CHUNK_SIZE = int(os.getenv("INTERACTIONS_CHUNK_SIZE", "300000"))

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")
SAMPLE_RECIPES_PATH = os.path.join(DATA_DIR, "sample_recipes.json")
FOODCOM_CSV_PATH = os.path.join(DATA_DIR, "RAW_recipes.csv")
FOODCOM_INTERACTIONS_PATH = os.path.join(DATA_DIR, "RAW_interactions.csv")
TECHNIQUE_QNA_PATH = os.path.join(DATA_DIR, "technique_qna.json")


def load_recipes() -> List[Dict]:
    """Load recipes for indexing."""

    if os.path.exists(FOODCOM_CSV_PATH):
        print(f"Found Food.com CSV: {FOODCOM_CSV_PATH}")
        return _parse_foodcom_csv(
            FOODCOM_CSV_PATH,
            nrows=SUBSET_SIZE,
            interactions_path=FOODCOM_INTERACTIONS_PATH,
        )

    print("Food.com CSV not found. Using built-in sample_recipes.json (small demo set).")
    with open(SAMPLE_RECIPES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_parse_list(raw: object) -> List[str]:
    if not isinstance(raw, str):
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return []


def _safe_int_id(value: object) -> Optional[int]:
    try:
        if value != value:
            return None
        return int(value)
    except Exception:
        return None


def _build_rating_map(path: str, *, allowed_recipe_ids: Optional[Set[int]] = None) -> Dict[int, Dict[str, float]]:
    """Build recipe_id -> {rating, rating_count} from RAW_interactions.csv.

    Uses chunked loading so large interaction files do not blow up memory.
    If allowed_recipe_ids is provided, only those IDs are aggregated.
    """
    rating_stats: Dict[int, Tuple[float, int]] = {}
    filtered = bool(allowed_recipe_ids)

    try:
        chunk_iter = pd.read_csv(
            path,
            usecols=["recipe_id", "rating"],
            chunksize=max(50000, INTERACTIONS_CHUNK_SIZE),
            low_memory=False,
        )
    except Exception as e:
        print(f"Could not parse RAW_interactions.csv ({e}). Skipping ratings enrichment.")
        return {}

    for i, chunk in enumerate(chunk_iter, start=1):
        chunk["recipe_id"] = pd.to_numeric(chunk["recipe_id"], errors="coerce")
        chunk["rating"] = pd.to_numeric(chunk["rating"], errors="coerce")
        chunk = chunk.dropna(subset=["recipe_id", "rating"])
        if chunk.empty:
            continue

        chunk["recipe_id"] = chunk["recipe_id"].astype(int)
        if filtered:
            chunk = chunk[chunk["recipe_id"].isin(allowed_recipe_ids)]
            if chunk.empty:
                continue

        grouped = chunk.groupby("recipe_id")["rating"].agg(["sum", "count"])
        for rid, row in grouped.iterrows():
            prev_sum, prev_cnt = rating_stats.get(int(rid), (0.0, 0))
            rating_stats[int(rid)] = (prev_sum + float(row["sum"]), prev_cnt + int(row["count"]))

        if i % 5 == 0:
            print(f"Processed {i} interaction chunks...")

    rating_map: Dict[int, Dict[str, float]] = {}
    for rid, (rating_sum, rating_count) in rating_stats.items():
        if rating_count <= 0:
            continue
        rating_map[rid] = {
            "rating": round(rating_sum / rating_count, 3),
            "rating_count": rating_count,
        }
    return rating_map


def _parse_foodcom_csv(
    path: str, *, nrows: int, interactions_path: Optional[str] = None
) -> List[Dict]:
    """Parse Food.com RAW_recipes.csv into our internal recipe schema."""
    df = pd.read_csv(
        path,
        nrows=nrows,
        usecols=["id", "name", "minutes", "tags", "ingredients", "steps"],
        low_memory=False,
    )

    # Build subset recipe ID set first, then rating enrichment only for those IDs.
    subset_ids: Set[int] = set()
    for rid in df["id"].tolist():
        rid_int = _safe_int_id(rid)
        if rid_int is not None:
            subset_ids.add(rid_int)

    rating_map: Dict[int, Dict[str, float]] = {}
    if interactions_path and os.path.exists(interactions_path):
        print(f"Found Food.com interactions CSV: {interactions_path}")
        rating_map = _build_rating_map(interactions_path, allowed_recipe_ids=subset_ids)
        print(f"Loaded rating aggregates for {len(rating_map)} subset recipes")
    else:
        print("RAW_interactions.csv not found. Indexing without ratings enrichment.")

    recipes: List[Dict] = []
    dropped_missing_id = 0
    dropped_missing_title = 0

    for _, row in df.iterrows():
        rid_int = _safe_int_id(row.get("id"))
        if rid_int is None:
            dropped_missing_id += 1
            continue

        name = str(row.get("name", "")).strip()
        if not name or name.lower() == "nan":
            dropped_missing_title += 1
            continue

        minutes_raw = row.get("minutes", 0)
        minutes = int(minutes_raw) if minutes_raw == minutes_raw else 0

        tags_list = _safe_parse_list(row.get("tags", "[]"))[:15]
        ing_list = _safe_parse_list(row.get("ingredients", "[]"))
        steps_list = _safe_parse_list(row.get("steps", "[]"))

        # Keep rows even if ingredients/steps are sparse, but ensure list types.
        rating_info = rating_map.get(rid_int, {})
        rating = rating_info.get("rating")
        rating_count = int(rating_info.get("rating_count", 0))

        recipes.append(
            {
                "id": f"fc_{rid_int}",
                "title": name,
                "tags": tags_list,
                "minutes": minutes,
                "ingredients": ing_list,
                "steps": steps_list,
                "rating": rating,
                "rating_count": rating_count,
            }
        )

    print(f"Parsed {len(recipes)} recipes from CSV")
    if dropped_missing_id or dropped_missing_title:
        print(
            f"Dropped rows -> missing_id: {dropped_missing_id}, "
            f"missing_title: {dropped_missing_title}"
        )
    return recipes


def build_index(recipes: List[Dict]) -> bool:
    """Build (or update) ChromaDB index."""
    from modules.rag_retriever import index_recipes, reset_collection

    print("Resetting Chroma collection to avoid stale vectors...")
    if not reset_collection():
        print("⚠️ Could not reset collection cleanly; attempting index anyway.")
    print(f"Indexing {len(recipes)} recipes into ChromaDB...")
    ok = index_recipes(recipes)
    if ok:
        print("✅ ChromaDB index built successfully!")
    else:
        print("⚠️ ChromaDB indexing failed. App will fall back to keyword search.")
    return ok


def main() -> None:
    print("=" * 60)
    print("ChefCoach RAG Setup")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    recipes = load_recipes()
    if not recipes:
        print("❌ No recipes loaded.")
        sys.exit(1)

    rated_count = sum(1 for r in recipes if r.get("rating") is not None)
    avg_rating = None
    if rated_count:
        avg_rating = round(
            sum(float(r["rating"]) for r in recipes if r.get("rating") is not None) / rated_count,
            3,
        )

    build_index(recipes)

    source = "foodcom_csv" if os.path.exists(FOODCOM_CSV_PATH) else "sample_recipes_json"
    technique_snippets_count = 0
    if os.path.exists(TECHNIQUE_QNA_PATH):
        try:
            with open(TECHNIQUE_QNA_PATH, "r", encoding="utf-8") as f:
                qna_rows = json.load(f)
            if isinstance(qna_rows, list):
                technique_snippets_count = len(qna_rows)
        except Exception:
            technique_snippets_count = 0

    rag_sources_available = [source]
    if technique_snippets_count > 0:
        rag_sources_available.append("technique_qna")

    summary = {
        "source": source,
        "recipe_subset_size_config": SUBSET_SIZE,
        "total_recipes_indexed": len(recipes),
        "sample_titles": [r.get("title", "") for r in recipes[:5]],
        "recipes_with_rating": rated_count,
        "avg_rating_in_indexed_set": avg_rating,
        "technique_snippets_indexed": technique_snippets_count,
        "rag_sources_available": rag_sources_available,
    }
    with open(os.path.join(DATA_DIR, "index_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Setup complete!")
    print("Run the UI with: streamlit run app.py")


if __name__ == "__main__":
    main()
