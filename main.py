# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import sqlite3
import os
from typing import List, Dict, Any, Optional
import hashlib
import json
from datetime import datetime, timezone

app = FastAPI()

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

        # If publications table doesn't exist at all, create fresh schema
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE_NAME,))
        if not cur.fetchone():
            cur.execute(f"""
                CREATE TABLE {TABLE_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    unique_key TEXT UNIQUE,
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
    migrate data into a new table with the desired schema.
    """
    cur = conn.cursor()
    # Read table info
    cur.execute(f"PRAGMA table_info({TABLE_NAME})")
    cols_info = cur.fetchall()  # each row: (cid, name, type, notnull, dflt_value, pk)
    existing_cols = [c[1] for c in cols_info]

    # If id and unique_key already present, nothing to do
    if "id" in existing_cols and "unique_key" in existing_cols:
        return

    # Otherwise we need to migrate
    tmp_table = f"{TABLE_NAME}_new"
    # Create new table with correct schema
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {tmp_table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unique_key TEXT UNIQUE,
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

    # For each old row, map fields to expected columns, generate unique_key and insert into new table
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
        # generate unique_key from Title/Faculty/Year (fall back to hashing whole row if empty)
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
        # Use the pre-built quoted column list in the f-string to avoid backslash issues
        cur.execute(f"INSERT OR IGNORE INTO {tmp_table} ({col_names_quoted}) VALUES ({placeholders})", values)
    conn.commit()

    # Drop old table and rename new to publications
    cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    cur.execute(f"ALTER TABLE {tmp_table} RENAME TO {TABLE_NAME}")
    conn.commit()


init_db()


# ---------------- DB helpers ----------------
def db_get_all(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    # SELECT * ORDER BY id (id should now exist after migration/init)
    cur.execute(f"SELECT * FROM {TABLE_NAME} ORDER BY id")
    rows = cur.fetchall()
    return [dict(r) for r in rows]


def db_get_by_unique_key(conn: sqlite3.Connection, unique_key: str) -> Optional[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {TABLE_NAME} WHERE unique_key = ?", (unique_key,))
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
    set_parts.append("unique_key = ?")
    values.append(unique_key)
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


# ---------------- upload & merge ----------------
@app.post("/upload-publications")
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

    # Add unique_key
    df["_unique_key"] = df.apply(generate_unique_key, axis=1)

    csv_import_time = datetime.now(timezone.utc)
    csv_import_iso = csv_import_time.isoformat(timespec="seconds")

    conn = sqlite3.connect(DB_FILE)
    try:
        # Ensure schema is up-to-date (in case DB changed externally)
        migrate_table_if_needed(conn)

        db_rows = db_get_all(conn)
        db_keys = {r["unique_key"]: r for r in db_rows if r.get("unique_key")}
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
                        db_update(conn, existing["id"], ukey, incoming_record, source="csv", prev_row=prev_snapshot)
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

        # merged CSV (kept for record; previous "merged" behavior preserved)
        merged = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY id", conn)
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
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {TABLE_NAME} ORDER BY id")
    rows = cur.fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
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


@app.put("/update-publication/{pub_id}")
def update_publication(pub_id: int, payload: Dict[str, Any]):
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
    series = pd.Series({k: update_vals.get(k) for k in EXPECTED_COLUMNS})
    new_ukey = generate_unique_key(series)
    db_update(conn, pub_id, new_ukey, update_vals, source="manual", prev_row=prev)
    conn.close()
    return {"message": "Updated", "id": pub_id}


@app.delete("/delete-publication/{pub_id}")
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
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY id", conn)
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
