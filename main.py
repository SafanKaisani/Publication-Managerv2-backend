# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import create_model, Field, BaseModel, ConfigDict
import pandas as pd
import numpy as np
import sqlite3
import os
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any, List
from collections import defaultdict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends
import secrets


app = FastAPI()
security = HTTPBasic()


USERNAME = os.getenv("APP_ADMIN_USERNAME", "admin")
PASSWORD = os.getenv("APP_ADMIN_PASSWORD", "supersecretpassword")

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    is_correct_username = secrets.compare_digest(credentials.username, USERNAME)
    is_correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_FOLDER = "uploads"
DB_FILE = "data.db"
TABLE_NAME = "publications" 
HISTORY_TABLE = "publications_history"

EXPECTED_COLUMNS: List[str] = [
    "Entry Date",
    "Faculty",
    "Publication Type",
    "Year",
    "Title",
    "Role",
    "Affiliation",
    "Status",
    "Reference",
    "Theme",
]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------- helpers ----------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def generate_unique_key(series_or_dict) -> str:
    """Stable key based on normalized Title + Faculty + Year.
       Accepts a pandas Series or a plain dict."""
    if isinstance(series_or_dict, pd.Series):
        title = str(series_or_dict.get("Title", "")).strip().lower()
        faculty = str(series_or_dict.get("Faculty", "")).strip().lower()
        year = str(series_or_dict.get("Year", "")).strip().lower()
    else:
        title = str(series_or_dict.get("Title", "")).strip().lower()
        faculty = str(series_or_dict.get("Faculty", "")).strip().lower()
        year = str(series_or_dict.get("Year", "")).strip().lower()
    base = f"{title}||{faculty}||{year}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def _clean_and_validate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers, drop extra columns, enforce order and dtypes."""
    # Normalize headers (strip + remove BOM)
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    # Drop columns not in expected list (keeps only expected ones)
    df = df[[c for c in df.columns if c in EXPECTED_COLUMNS]]

    # Check presence of all expected columns (case-insensitive)
    if set(c.lower() for c in df.columns) != set(c.lower() for c in EXPECTED_COLUMNS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV format. Expected columns: {EXPECTED_COLUMNS}, Found: {list(df.columns)}",
        )

    # Reorder to expected order (preserving exact casing)
    ci_map = {c.lower(): c for c in df.columns}
    ordered_cols = [ci_map[c.lower()] for c in EXPECTED_COLUMNS]
    df = df[ordered_cols]
    df.columns = EXPECTED_COLUMNS

    # Parse Entry Date -> datetime (coerce errors to NaT)
    df["Entry Date"] = pd.to_datetime(df["Entry Date"], errors="coerce")

    # Year -> nullable integer
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # Trim whitespace for object/string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("string").str.strip()

    return df


def _json_safe_scalar(x: Any) -> Any:
    """Convert pandas/numpy scalars to JSON-safe Python types."""
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if not np.isfinite(x):
            return None
        return float(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if pd.isna(x):
        return None
    if isinstance(x, (pd.Timestamp, datetime)):
        try:
            return pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
    return x


def get_publication_template() -> Dict[str, Any]:
    """
    Build a template dict for adding a publication.
    Excludes id and unique_key because those are auto-generated.
    All keys from EXPECTED_COLUMNS present and default to None.
    """
    template = {}
    for c in EXPECTED_COLUMNS:
        template[c] = None
    return template


def get_export_person_template() -> Dict[str, Any]:
    """
    Template for the export-person-pubs-pdf endpoint so the user knows what to pass.
    'name' is the exact Faculty string; start_year and end_year are optional inclusives.
    If 'name' is omitted, endpoint will return publications for ALL faculty (subject to year filters).
    """
    return {
        "name": None,
        "start_year": None,
        "end_year": None,
        "publication_types": None,  # e.g. ["Book Chapter", "Journal Article"]
        "affiliations": None,       # e.g. ["IED", "Some Dept"]
    }


# ---------------- DB init + migration ----------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.cursor()
        # Ensure history table exists
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {HISTORY_TABLE} (
                hist_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pub_id INTEGER,
                action TEXT,
                changed_at TEXT,
                previous_data TEXT
            )
        """)
        conn.commit()

        # If publications table doesn't exist at all, create fresh schema.
        # NOTE: unique_key is kept as a regular TEXT column (no UNIQUE constraint),
        # because we will rely on 'id' as the authoritative identifier.
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE_NAME,))
        if not cur.fetchone():
            cur.execute(f"""
                CREATE TABLE {TABLE_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    unique_key TEXT,
                    [Entry Date] TEXT,
                    Faculty TEXT,
                    [Publication Type] TEXT,
                    Year INTEGER,
                    Title TEXT,
                    Role TEXT,
                    Affiliation TEXT,
                    Status TEXT,
                    Reference TEXT,
                    Theme TEXT,
                    source TEXT,
                    last_modified TEXT
                )
            """)
            conn.commit()
        else:
            # Table exists: check schema and migrate if necessary
            migrate_table_if_needed(conn)
    finally:
        conn.close()


def migrate_table_if_needed(conn: sqlite3.Connection):
    """
    If the existing publications table lacks 'id' or 'unique_key' (old schema),
    migrate data into a new table with the desired schema (unique_key NOT UNIQUE).
    """
    cur = conn.cursor()
    # Read table info
    cur.execute(f"PRAGMA table_info({TABLE_NAME})")
    cols_info = cur.fetchall()  # each row: (cid, name, type, notnull, dflt_value, pk)
    existing_cols = [c[1] for c in cols_info]

    # If id and unique_key already present, nothing to do
    if "id" in existing_cols and "unique_key" in existing_cols:
        return

    # Otherwise we need to migrate into a new table WITHOUT unique constraint on unique_key
    tmp_table = f"{TABLE_NAME}_new"
    # Create new table with correct schema (unique_key is plain TEXT)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {tmp_table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unique_key TEXT,
            [Entry Date] TEXT,
            Faculty TEXT,
            [Publication Type] TEXT,
            Year INTEGER,
            Title TEXT,
            Role TEXT,
            Affiliation TEXT,
            Status TEXT,
            Reference TEXT,
            Theme TEXT,
            source TEXT,
            last_modified TEXT
        )
    """)
    conn.commit()

    # Read existing rows from old table (select whatever columns present)
    try:
        cur.execute(f"SELECT * FROM {TABLE_NAME}")
        rows = cur.fetchall()
        col_names = [d[0] for d in cur.description] if cur.description else []
    except Exception:
        rows = []
        col_names = []

    # For each old row, map fields to expected columns and insert into the new table
    for row in rows:
        rowdict = dict(zip(col_names, row))
        # Build a row_values dict for expected columns
        row_values = {}
        for c in EXPECTED_COLUMNS:
            v = rowdict.get(c)
            # If Entry Date exists, try to format it
            if c == "Entry Date" and v is not None:
                try:
                    v = pd.to_datetime(v, errors="coerce")
                    v = v.strftime("%Y-%m-%d %H:%M:%S") if not pd.isna(v) else None
                except Exception:
                    try:
                        v = str(v)
                    except Exception:
                        v = None
            # Year -> keep as-is or None
            if c == "Year" and v is not None:
                try:
                    v = int(v)
                except Exception:
                    try:
                        v = int(str(v))
                    except Exception:
                        v = None
            row_values[c] = v
        # generate unique_key from Title/Faculty/Year as a hint (not enforced unique now)
        ukey = generate_unique_key(rowdict)
        # Insert into new table
        cols = ["unique_key"] + EXPECTED_COLUMNS + ["source", "last_modified"]
        placeholders = ",".join("?" for _ in cols)
        col_names_quoted = ",".join(f'"{c}"' for c in cols)
        values = [ukey]
        for c in EXPECTED_COLUMNS:
            vv = row_values.get(c)
            if vv is None:
                values.append(None)
            else:
                values.append(str(vv))
        values.append(rowdict.get("source", "migrated"))  # mark migrated rows' source
        values.append(rowdict.get("last_modified", None))
        cur.execute(f"INSERT OR IGNORE INTO {tmp_table} ({col_names_quoted}) VALUES ({placeholders})", values)
    conn.commit()

    # Drop old table and rename new to publications
    cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    cur.execute(f"ALTER TABLE {tmp_table} RENAME TO {TABLE_NAME}")
    conn.commit()

def get_model_fields(model):
    """
    Get a dict of field names and types for a Pydantic model.
    """
    return {name: str(field.annotation) for name, field in model.model_fields.items()}


init_db()


# ---------------- DB helpers ----------------
def db_get_all(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    # return rows ordered by Year desc, Entry Date desc, Title asc (logical ordering)
    cur.execute(f'SELECT * FROM {TABLE_NAME} ORDER BY Year DESC, "Entry Date" DESC, Title ASC')
    rows = cur.fetchall()
    return [dict(r) for r in rows]


def db_get_by_unique_key(conn: sqlite3.Connection, unique_key: str) -> Optional[Dict[str, Any]]:
    """
    Return the first row matching the unique_key (unique_key is no longer enforced UNIQUE).
    This is used for CSV-merge heuristics; updates/deletes use 'id' only.
    """
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {TABLE_NAME} WHERE unique_key = ? LIMIT 1", (unique_key,))
    row = cur.fetchone()
    return dict(row) if row else None


def db_insert(conn: sqlite3.Connection, unique_key: str, row_values: Dict[str, Any], source: str) -> int:
    cur = conn.cursor()
    cols = ["unique_key"] + EXPECTED_COLUMNS + ["source", "last_modified"]
    col_names = ",".join(f'"{c}"' for c in cols)
    placeholders = ",".join("?" for _ in cols)
    values = [unique_key]
    for c in EXPECTED_COLUMNS:
        v = row_values.get(c)
        if c == "Entry Date":
            if v is None:
                v_str = None
            else:
                try:
                    v_str = pd.to_datetime(v, errors="coerce").strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    v_str = str(v)
        else:
            v_str = None if (v is None) else str(v)
        values.append(v_str)
    values.append(source)
    values.append(now_iso())
    sql = f"INSERT INTO {TABLE_NAME} ({col_names}) VALUES ({placeholders})"
    cur.execute(sql, values)
    conn.commit()
    new_id = cur.lastrowid
    cur.execute(
        f"INSERT INTO {HISTORY_TABLE} (pub_id, action, changed_at, previous_data) VALUES (?, ?, ?, ?)",
        (new_id, "insert", now_iso(), json.dumps({}, default=str))
    )
    conn.commit()
    return new_id


def db_update(conn: sqlite3.Connection, pub_id: int, unique_key: str, row_values: Dict[str, Any], source: str, prev_row: Dict[str, Any]):
    """
    Update a publication by id. IMPORTANT: Do NOT modify unique_key here --
    the 'id' is the authoritative identifier for updates.
    """
    cur = conn.cursor()
    set_parts = []
    values = []
    for c in EXPECTED_COLUMNS:
        set_parts.append(f'"{c}" = ?')
        v = row_values.get(c)
        if c == "Entry Date":
            if v is None:
                v_str = None
            else:
                try:
                    v_str = pd.to_datetime(v, errors="coerce").strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    v_str = str(v)
            values.append(v_str)
        else:
            values.append(None if v is None else str(v))
    set_parts.append("source = ?")
    values.append(source)
    set_parts.append("last_modified = ?")
    lm = now_iso()
    values.append(lm)
    # DO NOT update unique_key on manual updates; keep existing unique_key as-is.
    sql = f"UPDATE {TABLE_NAME} SET {', '.join(set_parts)} WHERE id = ?"
    values.append(pub_id)
    cur.execute(sql, values)
    conn.commit()
    cur.execute(
        f"INSERT INTO {HISTORY_TABLE} (pub_id, action, changed_at, previous_data) VALUES (?, ?, ?, ?)",
        (pub_id, "update", now_iso(), json.dumps(prev_row, default=str))
    )
    conn.commit()


def db_delete(conn: sqlite3.Connection, pub_id: int, prev_row: Dict[str, Any]):
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {TABLE_NAME} WHERE id = ?", (pub_id,))
    conn.commit()
    cur.execute(
        f"INSERT INTO {HISTORY_TABLE} (pub_id, action, changed_at, previous_data) VALUES (?, ?, ?, ?)",
        (pub_id, "delete", now_iso(), json.dumps(prev_row, default=str))
    )
    conn.commit()


class ExportPersonPubsRequest(BaseModel):
    """
    Request model for /export-person-pubs-pdf
    - name: exact Faculty string (optional). If omitted or null => search ALL faculty.
    - start_year: optional inclusive start year (int)
    - end_year: optional inclusive end year (int)
    """
    name: Optional[str] = Field(None, title="Exact Faculty name", description="Exact Faculty string to match; omit/null for all faculty")
    start_year: Optional[int] = Field(None, title="Start year (inclusive)")
    end_year: Optional[int] = Field(None, title="End year (inclusive)")
    publication_types: Optional[List[str]] = Field(None, title="Publication types (list)", description="List of exact Publication Type strings to filter (OR).")
    affiliations: Optional[List[str]] = Field(None, title="Affiliations (list)", description="List of exact Affiliation strings to filter (OR).")

class ExportPublicationsExcelRequest(BaseModel):
    year: Optional[int] = Field(None, title="Year")
    start_year: Optional[int] = Field(None, title="Start year (inclusive)")
    end_year: Optional[int] = Field(None, title="End year (inclusive)")
    faculty: Optional[str] = Field(None, title="Faculty name")
    publication_types: Optional[List[str]] = Field(None, title="Publication types (list)")
    affiliations: Optional[List[str]] = Field(None, title="Affiliations (list)")

class PersonStatisticsRequest(BaseModel):
    faculty: str

class PersonStatisticsResponse(BaseModel):
    faculty: str
    total_publications: int
    role_counts: dict
    status_counts: dict
    yearly_counts: dict

# ---------------- upload & merge ----------------
@app.post("/upload-publications",dependencies=[Depends(authenticate)])
async def upload_publications(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are allowed")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_path = os.path.join(UPLOAD_FOLDER, f"uploaded_{ts}.csv")
    with open(saved_path, "wb") as f:
        f.write(await file.read())

    # Read CSV robustly
    try:
        df = pd.read_csv(saved_path, encoding="utf-8-sig", quotechar='"')
    except UnicodeDecodeError:
        df = pd.read_csv(saved_path, encoding="ISO-8859-1", quotechar='"')

    # Clean and validate
    df = _clean_and_validate_df(df)

    # Add unique_key (used for CSV matching heuristics)
    df["_unique_key"] = df.apply(generate_unique_key, axis=1)

    csv_import_time = datetime.now(timezone.utc)
    csv_import_iso = csv_import_time.isoformat(timespec="seconds")

    conn = sqlite3.connect(DB_FILE)
    try:
        # Ensure schema is up-to-date (in case DB changed externally)
        migrate_table_if_needed(conn)

        db_rows = db_get_all(conn)
        db_keys = {r.get("unique_key"): r for r in db_rows if r.get("unique_key")}
        incoming_keys = set(df["_unique_key"].tolist())

        added = []
        updated = []
        unchanged = []
        missing_in_csv = []

        for _, row in df.iterrows():
            ukey = row["_unique_key"]

            # Re-check DB for this key (handles duplicates in incoming CSV and any concurrent inserts)
            existing = db_keys.get(ukey)
            if not existing:
                existing = db_get_by_unique_key(conn, ukey)
                if existing:
                    db_keys[ukey] = existing

            incoming_record: Dict[str, Any] = {}
            for c in EXPECTED_COLUMNS:
                v = row.get(c)
                if pd.isna(v):
                    incoming_record[c] = None
                else:
                    if c == "Entry Date":
                        try:
                            incoming_record[c] = pd.to_datetime(v, errors="coerce").strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            incoming_record[c] = None
                    else:
                        incoming_record[c] = str(v).strip()

            if not existing:
                # Insert new row
                new_id = db_insert(conn, ukey, incoming_record, source="csv")
                # Immediately fetch/attach the inserted row to db_keys to avoid duplicate insert attempts
                inserted = db_get_by_unique_key(conn, ukey)
                if inserted:
                    db_keys[ukey] = inserted
                added.append({"id": new_id, "unique_key": ukey, **incoming_record})
            else:
                # Decide whether to keep existing (if last_modified exists and is later than csv_import_time)
                keep_existing = False
                existing_lm = existing.get("last_modified")
                if existing_lm:
                    try:
                        existing_dt = datetime.fromisoformat(existing_lm)
                        if existing_dt > csv_import_time:
                            keep_existing = True
                    except Exception:
                        keep_existing = False

                if keep_existing:
                    unchanged.append({"id": existing["id"], **existing})
                else:
                    prev_snapshot = existing.copy()
                    changed_fields = {}
                    for c in EXPECTED_COLUMNS:
                        old_v = existing.get(c)
                        old_norm = None if old_v is None else str(old_v).strip()
                        new_norm = incoming_record.get(c)
                        new_norm = None if new_norm is None or str(new_norm).strip() == "" else str(new_norm).strip()
                        if old_norm != new_norm:
                            changed_fields[c] = {"old": old_v, "new": new_norm}
                    if changed_fields:
                        # Use existing's unique_key (do not change unique_key on update)
                        db_update(conn, existing["id"], existing.get("unique_key"), incoming_record, source="csv", prev_row=prev_snapshot)
                        # refresh db_keys entry after update
                        refreshed = db_get_by_unique_key(conn, ukey)
                        if refreshed:
                            db_keys[ukey] = refreshed
                        updated.append({"id": existing["id"], "unique_key": ukey, "changes": changed_fields})
                    else:
                        unchanged.append({"id": existing["id"], **existing})

        # missing_in_csv
        for k, r in db_keys.items():
            if k not in incoming_keys:
                missing_in_csv.append({"id": r["id"], "unique_key": k, "Title": r.get("Title"), "Faculty": r.get("Faculty"), "Year": r.get("Year")})

        # merged CSV (kept for record; previous "merged" behavior preserved) - ordered logically
        merged = pd.read_sql(f'SELECT * FROM {TABLE_NAME} ORDER BY Year DESC, "Entry Date" DESC, Title ASC', conn)
        if "Entry Date" in merged.columns:
            merged["Entry Date"] = pd.to_datetime(merged["Entry Date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        merged_path = os.path.join(UPLOAD_FOLDER, f"merged_{ts}.csv")
        merged.to_csv(merged_path, index=False)

        summary = {
            "import_time": csv_import_iso,
            "added_count": len(added),
            "updated_count": len(updated),
            "unchanged_count": len(unchanged),
            "missing_in_csv_count": len(missing_in_csv),
            "added": added[:50],
            "updated": updated[:200],
            "missing_in_csv": missing_in_csv[:200],
            "merged_csv": os.path.basename(merged_path),
        }
        return summary
    finally:
        conn.close()


# ---------------- read / search / modify endpoints ----------------
@app.get("/get-publications")
def get_publications():
    conn = sqlite3.connect(DB_FILE)
    try:
        rows = db_get_all(conn)
    finally:
        conn.close()

    result = []
    for d in rows:
        # Normalize Entry Date and Year types for JSON response
        if d.get("Entry Date"):
            try:
                d["Entry Date"] = pd.to_datetime(d["Entry Date"], errors="coerce").strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
        if d.get("Year") is not None:
            try:
                d["Year"] = int(d["Year"])
            except Exception:
                pass
        # Ensure JSON-safe None for nulls
        for k, v in list(d.items()):
            if pd.isna(v):
                d[k] = None
        result.append(d)
    return {"data": result}


@app.get("/search-publications")
def search_publications(title: str):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT id, Title, Faculty, Year, Status FROM {TABLE_NAME}")
    rows = cur.fetchall()
    conn.close()
    q = title.strip().lower()
    matches = []
    for r in rows:
        t = (r["Title"] or "").lower()
        if q in t:
            matches.append(dict(r))
    return {"matches": matches}


@app.put("/update-publication/{pub_id}",dependencies=[Depends(authenticate)])
def update_publication(pub_id: int, payload: Dict[str, Any]):
    """
    Update by ID. Do NOT recalculate or overwrite unique_key here.
    Only fields present in EXPECTED_COLUMNS are updated; missing fields keep previous values.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {TABLE_NAME} WHERE id = ?", (pub_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Publication not found")
    prev = dict(row)
    update_vals: Dict[str, Any] = {}
    for c in EXPECTED_COLUMNS:
        if c in payload:
            update_vals[c] = payload[c]
        else:
            update_vals[c] = prev.get(c)
    # Use existing unique_key when updating (do not change it)
    existing_ukey = prev.get("unique_key")
    db_update(conn, pub_id, existing_ukey, update_vals, source="manual", prev_row=prev)
    conn.close()
    return {"message": "Updated", "id": pub_id}


@app.delete("/delete-publication/{pub_id}",dependencies=[Depends(authenticate)])
def delete_publication(pub_id: int):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {TABLE_NAME} WHERE id = ?", (pub_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Publication not found")
    prev = dict(row)
    db_delete(conn, pub_id, prev)
    conn.close()
    return {"message": "Deleted", "id": pub_id}


@app.get("/history/{pub_id}")
def get_history(pub_id: int):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {HISTORY_TABLE} WHERE pub_id = ? ORDER BY changed_at DESC", (pub_id,))
    rows = cur.fetchall()
    conn.close()
    history = []
    for r in rows:
        rowd = dict(r)
        try:
            rowd["previous_data"] = json.loads(rowd.get("previous_data") or "{}")
        except Exception:
            pass
        history.append(rowd)
    return {"history": history}


# ---------------- export latest DB as CSV (single-button) ----------------
@app.get("/export-latest")
def export_latest():
    """
    Export the current publications table to a timestamped CSV and return it for download.
    The CSV includes all columns: id, unique_key, Entry Date, Faculty, Publication Type, Year, Title, Role,
    Affiliation, Status, Reference, Theme, source, last_modified
    """
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql(f'SELECT * FROM {TABLE_NAME} ORDER BY Year DESC, "Entry Date" DESC, Title ASC', conn)
    finally:
        conn.close()

    # Normalize Entry Date formatting (if present)
    if "Entry Date" in df.columns:
        df["Entry Date"] = pd.to_datetime(df["Entry Date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    # Build timestamped filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"publications_export_{ts}.csv"
    path = os.path.join(UPLOAD_FOLDER, filename)

    # Write CSV to disk and return it (FileResponse will trigger download in browsers)
    df.to_csv(path, index=False)

    return FileResponse(path, filename=filename, media_type="text/csv")


# ---------------- dynamic Pydantic model from DB schema ----------------
def _get_db_columns_for_create() -> List[Tuple[str, type]]:
    """
    Query the DB schema and return a list of (column_name, python_type) for
    columns that should be present in the create model.
    Excludes id, unique_key, source, last_modified.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({TABLE_NAME})")
        info = cur.fetchall()  # (cid, name, type, notnull, dflt_value, pk)
        conn.close()
    except Exception:
        info = []

    cols: List[Tuple[str, type]] = []
    if not info:
        # fallback to EXPECTED_COLUMNS
        for c in EXPECTED_COLUMNS:
            if c == "Year":
                cols.append((c, Optional[int]))
            else:
                cols.append((c, Optional[str]))
        return cols

    for col in info:
        name = col[1]
        if name in ("id", "unique_key", "source", "last_modified"):
            continue
        # map sqlite type hint or known column names to pydantic types
        if name == "Year":
            cols.append((name, Optional[int]))
        elif name == "Entry Date":
            # keep as string in input; we'll parse it later
            cols.append((name, Optional[str]))
        else:
            cols.append((name, Optional[str]))
    # If DB schema missing any expected column, ensure expected ones included
    existing_names = [n for n, _ in cols]
    for c in EXPECTED_COLUMNS:
        if c not in existing_names:
            if c == "Year":
                cols.append((c, Optional[int]))
            elif c == "Entry Date":
                cols.append((c, Optional[str]))
            else:
                cols.append((c, Optional[str]))
    return cols


def _make_safe_field_name(name: str, used: set) -> str:
    # Replace non-alnum with underscore and collapse duplicates
    safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
    if not safe:
        safe = "field"
    # ensure it doesn't start with digit
    if safe[0].isdigit():
        safe = f"f_{safe}"
    orig = safe
    i = 1
    while safe in used:
        safe = f"{orig}_{i}"
        i += 1
    used.add(safe)
    return safe


# Build a dynamic model for creation (used in FastAPI docs)
def build_publication_create_model():
    # get desired columns (original names with spaces)
    cols = _get_db_columns_for_create()
    fields = {}
    used = set()

    for orig_name, ptype in cols:
        safe_name = _make_safe_field_name(orig_name, used)
        # alias keeps original DB column name in the OpenAPI schema
        fields[safe_name] = (ptype, Field(None, alias=orig_name, title=orig_name))

    class _Base(BaseModel):
        # Pydantic v2 config
        model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # Pydantic v2 create_model: pass __base__ instead of __config__
    model = create_model("PublicationCreate", __base__=_Base, **fields)
    return model


PublicationCreate = build_publication_create_model()


@app.post("/add-publication", response_model=Dict[str, Any],dependencies=[Depends(authenticate)])
def add_publication(payload: PublicationCreate = Body(...)):  # type: ignore
    """
    Add a single publication via JSON (typed by PublicationCreate).
    - Payload may include any of EXPECTED_COLUMNS (all optional).
    - Missing fields become NULL.
    - Entry Date defaults to now if not provided.
    - Year coerced to int if possible.
    - unique_key is generated from Title/Faculty/Year (not enforced UNIQUE).
    - Returns the created publication row (including id).
    """
    # Convert payload -> dict using aliases so keys are the original DB column names (including spaces)
    raw = payload.dict(by_alias=True, exclude_unset=True)

    # Build row_values for DB insertion
    row_values: Dict[str, Any] = {}
    for c in EXPECTED_COLUMNS:
        if c in raw:
            v = raw[c]
            if c == "Year":
                try:
                    v = int(v) if v is not None and str(v).strip() != "" else None
                except Exception:
                    v = None
            if c == "Entry Date":
                # Normalize entry date if provided
                try:
                    v_parsed = pd.to_datetime(v, errors="coerce")
                    v = v_parsed.strftime("%Y-%m-%d %H:%M:%S") if not pd.isna(v_parsed) else None
                except Exception:
                    v = None
            row_values[c] = v
        else:
            # default behavior for missing fields
            if c == "Entry Date":
                # default entry date -> now (string)
                row_values[c] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                row_values[c] = None

    # generate unique_key (as a hint — id is authoritative)
    unique_key = generate_unique_key(row_values)

    conn = sqlite3.connect(DB_FILE)
    try:
        # ensure migration/schema up-to-date
        migrate_table_if_needed(conn)

        new_id = db_insert(conn, unique_key, row_values, source="manual")

        # fetch inserted row
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {TABLE_NAME} WHERE id = ?", (new_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=500, detail="Failed to retrieve inserted publication")

        pub = dict(row)
        # normalize types for JSON
        if pub.get("Entry Date"):
            try:
                pub["Entry Date"] = pd.to_datetime(pub["Entry Date"], errors="coerce").strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
        if pub.get("Year") is not None:
            try:
                pub["Year"] = int(pub["Year"])
            except Exception:
                pass
        # sanitize NaNs
        for k, v in list(pub.items()):
            if pd.isna(v):
                pub[k] = None

        return {"message": "Created", "publication": pub}
    finally:
        conn.close()


# ---------------- PDF generation helper ----------------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4


def _sanitize_filename_part(s: str) -> str:
    # simple sanitizer for file name sections
    keep = "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in s)
    return "_".join(keep.strip().split())


def generate_person_pdf(name: str, rows: List[Dict[str, Any]], start_year: Optional[int], end_year: Optional[int], out_path: str):
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    normal = styles["BodyText"]
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=9)

    doc = SimpleDocTemplate(out_path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    elements: List[Any] = []

    # Title
    period = ""
    if start_year is None and end_year is None:
        period = "All years"
    else:
        s = start_year if start_year is not None else ""
        e = end_year if end_year is not None else ""
        period = f"{s} - {e}" if s or e else "All years"

    elements.append(Paragraph(f"Publications for: {name if name is not None else 'All Faculty'}", title_style))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"Period: {period}", normal))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small))
    elements.append(Spacer(1, 12))

    if not rows:
        elements.append(Paragraph("No publications found for the requested criteria.", normal))
        doc.build(elements)
        return

    # For each publication, add a block
    for idx, r in enumerate(rows, start=1):
        # Header for the publication
        header = f"{idx}. {r.get('Title') or '(No title)'}"
        elements.append(Paragraph(header, styles["Heading3"]))
        elements.append(Spacer(1, 4))

        # Build two-column table of key/value pairs
        kv = []
        def safe(val):
            if val is None:
                return ""
            # for timestamps, ensure string
            if isinstance(val, (pd.Timestamp, datetime)):
                try:
                    return pd.to_datetime(val).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    return str(val)
            return str(val)

        fields_to_show = [
            ("Year", r.get("Year")),
            ("Publication Type", r.get("Publication Type")),
            ("Role", r.get("Role")),
            ("Faculty / Author", r.get("Faculty")),
            ("Affiliation", r.get("Affiliation")),
            ("Status", r.get("Status")),
            ("Entry Date", r.get("Entry Date")),
            ("Theme", r.get("Theme")),
        ]
        for k, v in fields_to_show:
            kv.append([Paragraph(f"<b>{k}</b>", small), Paragraph(safe(v), small)])

        # Add Reference in full width below if exists
        t = Table(kv, colWidths=[100, 420])
        t.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        elements.append(t)

        if r.get("Reference"):
            elements.append(Spacer(1, 4))
            elements.append(Paragraph("<b>Reference:</b>", small))
            elements.append(Paragraph(safe(r.get("Reference")), small))
        elements.append(Spacer(1, 10))

        # Add a faint horizontal rule
        hr = Table([[""]], colWidths=[520])
        hr.setStyle(TableStyle([("LINEBELOW", (0,0), (-1,-1), 0.25, colors.grey)]))
        elements.append(hr)
        elements.append(Spacer(1, 10))

    doc.build(elements)


# ---------------- new: export publications for a person as PDF ----------------
@app.post("/export-person-pubs-pdf", dependencies=[Depends(authenticate)])
def export_person_pubs_pdf(
    payload: Optional[ExportPersonPubsRequest] = Body(
        None,
        examples={
            "range": {
                "summary": "Name with year range",
                "value": {"name": "Dr. Jane Doe", "start_year": 2019, "end_year": 2024},
            },
            "all_years_name": {
                "summary": "Just a name (all years)",
                "value": {"name": "Dr. Jane Doe"},
            },
            "all_faculty_years": {
                "summary": "No name (all faculty), with years",
                "value": {"start_year": 2020, "end_year": 2021},
            },
            "all_faculty_all_years": {
                "summary": "No body -> returns template",
                "value": None,
            },
        },
    )
):
    """
    Export publications for a person (or for ALL if name omitted).
    Treats name="string" as not provided; year==0 as not provided; flattens comma lists; case-insensitive matching.
    """
    if payload is None:
        return {
            "template": get_export_person_template(),
            "note": "Provide 'name' (exact Faculty string) or omit/null for all faculty. Use start_year/end_year (0 means not provided) or year."
        }

    # normalize name
    name_raw = payload.name
    name = None
    if isinstance(name_raw, str):
        nr = name_raw.strip()
        if nr != "" and nr.lower() != "string":
            name = nr

    def to_nullable_year(v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            iv = int(v)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Year must be an integer or 0 to mean not provided: {v}")
        return None if iv == 0 else iv

    start_year = to_nullable_year(payload.start_year)
    end_year = to_nullable_year(payload.end_year)

    def flatten_list_field(val: Optional[List[str]]) -> List[str]:
        if not val:
            return []
        items: List[str] = []
        for item in val:
            if item is None:
                continue
            for part in str(item).split(","):
                p = part.strip()
                if not p or p.lower() == "string":
                    continue
                items.append(p)
        seen = set()
        out = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    publication_types = flatten_list_field(payload.publication_types)
    affiliations = flatten_list_field(payload.affiliations)

    # Build SQL with case-insensitive matching
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    sql = f"SELECT * FROM {TABLE_NAME}"
    params: List[Any] = []
    where_clauses: List[str] = []

    if name is not None:
        where_clauses.append("LOWER(Faculty) = ?")
        params.append(name.lower())

    if publication_types:
        placeholders = ",".join("?" for _ in publication_types)
        where_clauses.append(f'LOWER("Publication Type") IN ({placeholders})')
        params.extend([p.lower() for p in publication_types])

    if affiliations:
        placeholders = ",".join("?" for _ in affiliations)
        where_clauses.append(f'LOWER(Affiliation) IN ({placeholders})')
        params.extend([a.lower() for a in affiliations])

    # Year: apply even if name omitted
    if start_year is not None and end_year is not None:
        where_clauses.append("Year BETWEEN ? AND ?")
        params.extend([start_year, end_year])
    elif start_year is not None:
        where_clauses.append("Year >= ?")
        params.append(start_year)
    elif end_year is not None:
        where_clauses.append("Year <= ?")
        params.append(end_year)

    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    sql += ' ORDER BY Year DESC, "Entry Date" DESC, Title ASC'

    try:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        publications = [dict(r) for r in rows]
    finally:
        conn.close()

    # Normalize fields for PDF (unchanged)
    for p in publications:
        if p.get("Entry Date"):
            try:
                p["Entry Date"] = pd.to_datetime(p["Entry Date"], errors="coerce").strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
        if p.get("Year") is not None:
            try:
                p["Year"] = int(p["Year"])
            except Exception:
                pass
        for k, v in list(p.items()):
            if pd.isna(v):
                p[k] = None

    safe_name_part = _sanitize_filename_part(name if name is not None else "ALL")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name_part}_publications_{ts}.pdf"
    out_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        generate_person_pdf(name if name is not None else "All Faculty", publications, start_year, end_year, out_path)
    except Exception as e:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {e}")

    return FileResponse(out_path, filename=filename, media_type="application/pdf")




@app.post("/export-publications-excel", response_class=FileResponse, dependencies=[Depends(authenticate)])
def export_publications_excel(payload: ExportPublicationsExcelRequest = Body(...)):
    """
    Export publications as an Excel file.

    Behavior changes/fixes:
      - Treat year == 0 as "not provided" (so it won't conflict with start_year/end_year).
      - start_year/end_year: 0 treated as not provided.
      - publication_types/affiliations accept list elements or comma-separated strings.
      - placeholder literal "string" (case-insensitive) is ignored.
      - Matching is case-insensitive for Faculty/Publication Type/Affiliation.
    """
    # helper: treat 0 as "not provided"
    def to_nullable_year_field(v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            iv = int(v)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid integer for year field: {v}")
        return None if iv == 0 else iv

    # parse year (treat 0 as None)
    year = None
    if payload.year is not None:
        try:
            yv = int(payload.year)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid 'year' value")
        year = None if yv == 0 else yv

    start_year = to_nullable_year_field(payload.start_year)
    end_year = to_nullable_year_field(payload.end_year)

    # flatten helper for publication_types/affiliations and drop placeholder "string"
    def flatten_list_field(val: Optional[List[str]]) -> List[str]:
        if not val:
            return []
        items: List[str] = []
        for item in val:
            if item is None:
                continue
            for part in str(item).split(","):
                p = part.strip()
                # drop placeholder literal "string" (case-insensitive)
                if not p or p.lower() == "string":
                    continue
                items.append(p)
        # dedupe preserving order
        seen = set()
        out = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    publication_types = flatten_list_field(payload.publication_types)
    affiliations = flatten_list_field(payload.affiliations)

    # guard: don't allow providing 'year' together with start/end range
    # (year already treated as None if it was 0)
    if year is not None and (start_year is not None or end_year is not None):
        raise HTTPException(
            status_code=400,
            detail="Please provide either 'year' OR 'start_year'/'end_year', not both."
        )

    # faculty: treat empty string or literal "string" (case-insensitive placeholder) as not provided
    faculty_raw = payload.faculty if payload.faculty is not None else ""
    faculty = None
    if isinstance(faculty_raw, str):
        fr = faculty_raw.strip()
        if fr != "" and fr.lower() != "string":
            faculty = fr

    # Build SQL (use case-insensitive matching via LOWER(...))
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    sql = f"SELECT * FROM {TABLE_NAME}"
    params: List[Any] = []
    where_clauses: List[str] = []

    if faculty:
        where_clauses.append("LOWER(Faculty) = ?")
        params.append(faculty.lower())

    if publication_types:
        placeholders = ",".join("?" for _ in publication_types)
        where_clauses.append(f'LOWER("Publication Type") IN ({placeholders})')
        params.extend([p.lower() for p in publication_types])

    if affiliations:
        placeholders = ",".join("?" for _ in affiliations)
        where_clauses.append(f'LOWER(Affiliation) IN ({placeholders})')
        params.extend([a.lower() for a in affiliations])

    if year is not None:
        where_clauses.append("Year = ?")
        params.append(year)
    elif start_year is not None and end_year is not None:
        where_clauses.append("Year BETWEEN ? AND ?")
        params.extend([start_year, end_year])
    elif start_year is not None:
        where_clauses.append("Year >= ?")
        params.append(start_year)
    elif end_year is not None:
        where_clauses.append("Year <= ?")
        params.append(end_year)

    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    sql += ' ORDER BY Year DESC, "Entry Date" DESC, Title ASC'

    cur.execute(sql, tuple(params))
    rows = cur.fetchall()
    conn.close()

    df = pd.DataFrame([dict(r) for r in rows])

    if df.empty:
        raise HTTPException(status_code=404, detail="No publications found for the given filters")

    # ✅ Drop internal-use columns
    df = df.drop(columns=[c for c in ["id", "unique_key", "Entry Date", "last_modified", "source"] if c in df.columns], errors="ignore")

    # Build filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filter_part = []
    if faculty:
        filter_part.append(faculty.replace(" ", "_"))
    if year:
        filter_part.append(str(year))
    elif start_year or end_year:
        filter_part.append(f"{start_year or ''}-{end_year or ''}")

    if publication_types:
        pts = [p.replace(" ", "_") for p in publication_types][:5]
        filter_part.append("PT-" + "+".join(pts))
    if affiliations:
        affs = [a.replace(" ", "_") for a in affiliations][:5]
        filter_part.append("AFF-" + "+".join(affs))

    filter_str = "_".join(filter_part) if filter_part else "ALL"
    filename = f"publications_{filter_str}_{ts}.xlsx"
    out_path = os.path.join(UPLOAD_FOLDER, filename)

    # Save to Excel
    df.to_excel(out_path, index=False)

    return FileResponse(
        out_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )




@app.post("/statistics/person", response_model=PersonStatisticsResponse)
def get_person_statistics(payload: PersonStatisticsRequest = Body(...)):
    """
    Get statistics for a given faculty member:
    - Total publications
    - Role distribution (Main Author, Co-Author, etc.)
    - Status distribution (Published, Unpublished, etc.)
    - Yearly breakdown
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Fetch all rows for the given faculty
    cur.execute(f"SELECT * FROM {TABLE_NAME} WHERE Faculty = ?", (payload.faculty,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No publications found for {payload.faculty}")

    df = pd.DataFrame([dict(r) for r in rows])

    # Prepare statistics
    total_publications = len(df)

    # Count roles
    role_counts = df["Role"].value_counts().to_dict() if "Role" in df.columns else {}

    # Count status (Published, In-progress, etc.)
    status_counts = df["Status"].value_counts().to_dict() if "Status" in df.columns else {}

    # Count yearly publications
    yearly_counts = df["Year"].value_counts().sort_index().to_dict() if "Year" in df.columns else {}

    return PersonStatisticsResponse(
        faculty=payload.faculty,
        total_publications=total_publications,
        role_counts=role_counts,
        status_counts=status_counts,
        yearly_counts=yearly_counts
    )

@app.get("/statistics/publications")
def statistics_publications(
    start_year: int = Query(None, description="Filter publications from this year onwards"),
    end_year: int = Query(None, description="Filter publications up to this year"),
    publication_types: Optional[List[str]] = Query(None, description="Publication types (repeatable or comma-separated)"),
    affiliations: Optional[List[str]] = Query(None, description="Affiliations (repeatable or comma-separated)")
):
    """
    Return publication statistics by year with optional multi-value filters:
      - publication_types: OR semantics within list
      - affiliations: OR semantics within list
      - Year filters still apply
    Also returns the actual filters applied in `filters_applied` for frontend use.
    """
    # --- helpers ---
    def flatten_query_list(raw: Optional[List[str]]) -> List[str]:
        """Accept repeated query params or comma-separated items; dedupe and ignore placeholder 'string'."""
        if not raw:
            return []
        items: List[str] = []
        for item in raw:
            if item is None:
                continue
            for part in str(item).split(","):
                p = part.strip()
                if not p:
                    continue
                if p.lower() == "string":  # placeholder from docs — ignore it
                    continue
                items.append(p)
        # dedupe preserving order but case-insensitive uniqueness
        seen_lowers = set()
        out = []
        for x in items:
            key = x.lower()
            if key not in seen_lowers:
                seen_lowers.add(key)
                out.append(x)
        return out

    ptypes = flatten_query_list(publication_types)
    affs = flatten_query_list(affiliations)

    # Build SQL with case-insensitive matching where applicable
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    sql = f"SELECT Year, Title, Faculty, \"Publication Type\", Affiliation FROM {TABLE_NAME}"
    params: List[Any] = []
    where_clauses: List[str] = []

    # Year filters (apply even when other filters absent)
    if start_year is not None:
        where_clauses.append("Year >= ?")
        params.append(start_year)
    if end_year is not None:
        where_clauses.append("Year <= ?")
        params.append(end_year)

    # Publication types (case-insensitive)
    if ptypes:
        placeholders = ",".join("?" for _ in ptypes)
        where_clauses.append(f'LOWER("Publication Type") IN ({placeholders})')
        params.extend([p.lower() for p in ptypes])

    # Affiliations (case-insensitive)
    if affs:
        placeholders = ",".join("?" for _ in affs)
        where_clauses.append(f'LOWER(Affiliation) IN ({placeholders})')
        params.extend([a.lower() for a in affs])

    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    sql += ' ORDER BY Year DESC, "Entry Date" DESC, Title ASC'

    cur.execute(sql, tuple(params))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="No publications found for the given filters")

    # Aggregate while handling None years safely
    year_stats = {}
    for row in rows:
        year = row["Year"]
        title = (row["Title"] or "").strip()
        faculty = (row["Faculty"] or "").strip()

        # normalize year key: try to use int if possible else None
        try:
            ykey = int(year) if year is not None and str(year).strip() != "" and str(year).strip().isdigit() else None
        except Exception:
            ykey = None

        if ykey not in year_stats:
            year_stats[ykey] = {"publications": 0, "contributors": set(), "authors_per_pub": {}}
        year_stats[ykey]["publications"] += 1
        if faculty:
            year_stats[ykey]["contributors"].add(faculty)
        # authors per pub: use title as key
        if title not in year_stats[ykey]["authors_per_pub"]:
            year_stats[ykey]["authors_per_pub"][title] = set()
        if faculty:
            year_stats[ykey]["authors_per_pub"][title].add(faculty)

    # Build response sorted by year (None last)
    sorted_years = sorted([y for y in year_stats.keys() if y is not None], reverse=True)
    if None in year_stats:
        sorted_years.append(None)

    result = []
    for year in sorted_years:
        stats = year_stats[year]
        pub_count = stats["publications"]
        contributors_count = len(stats["contributors"])
        avg_authors = (
            sum(len(authors) for authors in stats["authors_per_pub"].values()) /
            max(1, len(stats["authors_per_pub"]))
        )
        result.append({
            "year": year,
            "publications": pub_count,
            "unique_contributors": contributors_count,
            "average_authors_per_publication": round(avg_authors, 2)
        })

    # Build filters_applied: categories + actual lists for frontend
    categories = 0
    if start_year is not None or end_year is not None:
        categories += 1
    if ptypes:
        categories += 1
    if affs:
        categories += 1

    filters_detail = {
        "categories": categories,
        "publication_types": ptypes,   # list of actual publication type filters applied
        "affiliations": affs,          # list of actual affiliation filters applied
        "year_filter_applied": bool(start_year is not None or end_year is not None)
    }

    return {
        "statistics": result,
        "filters_applied": filters_detail
    }




