import requests
import pandas as pd
from io import StringIO
import html
import re
import streamlit as st

def make_unique_columns(cols):
    """
    Buat nama kolom unik jika ada duplikat:
    'A', 'A' -> 'A', 'A.1', ...
    """
    counts = {}
    new = []
    for c in cols:
        c = "" if c is None else str(c)
        base = c.strip() if c.strip().lower() != "nan" and c.strip() != "" else "Kolom"
        if base not in counts:
            counts[base] = 0
            new_name = base
        else:
            counts[base] += 1
            new_name = f"{base}.{counts[base]}"
        new.append(new_name)
    return new

def detect_data_start(df):
    """
    Cari baris pertama yang mengandung angka (indikator baris data).
    """
    for i in range(len(df)):
        row = df.iloc[i]
        digit_mask = row.apply(lambda v: bool(re.search(r"\d", str(v))) if pd.notna(v) else False)
        if digit_mask.sum() >= len(row) // 2:
            return i
    return 0

def parse_bps_table_structure_adaptive(url: str) -> pd.DataFrame:
    """
    Ambil HTML tabel dari API BPS dan kembalikan DataFrame.
    """
    r = requests.get(url)
    r.raise_for_status()
    j = r.json()

    try:
        table_html = html.unescape(j["data"]["table"])
    except Exception as e:
        raise ValueError("Tidak menemukan field data.table pada respon API.") from e

    # Tandai koma desimal sementara
    table_html = re.sub(r"(?<=\d),(?=\d)", "§", table_html)

    dfs = pd.read_html(StringIO(table_html), header=None)
    if not dfs:
        raise ValueError("Tidak ditemukan tabel dalam HTML.")
    df_raw = dfs[0]

    df = df_raw.astype(str).apply(lambda col: col.str.replace("§", ",", regex=False))

    data_start = detect_data_start(df)
    header_df = df.iloc[:data_start] if data_start > 0 else pd.DataFrame()
    data_df = df.iloc[data_start:].copy() if data_start < len(df) else pd.DataFrame()

    if not header_df.empty:
        header_values = header_df.fillna("").astype(str).values
        new_cols = []
        for idx, col in enumerate(zip(*header_values)):
            parts = [str(x).strip() for x in col if str(x).strip() not in ("", "nan")]
            colname = " ".join(parts).strip()
            if colname == "":
                colname = f"Kolom{idx}"
            new_cols.append(colname)
        new_cols = make_unique_columns(new_cols)
        data_df.columns = new_cols
    else:
        default_cols = [f"Kolom{i}" for i in range(df.shape[1])]
        data_df.columns = make_unique_columns(default_cols)

    data_df.columns = make_unique_columns(data_df.columns)
    for i, col in enumerate(data_df.columns):
        series = data_df.iloc[:, i]
        if isinstance(series, pd.Series):
            cleaned = series.astype(str).str.replace("\u00a0", " ", regex=False).str.strip()
            data_df.iloc[:, i] = cleaned

    data_df = data_df.dropna(how="all").reset_index(drop=True)
    return data_df

def parse_table_static(entity: dict) -> str:
    """
    Wrapper untuk RAG chain: build URL static table BPS dari id_table.
    Return JSON string hasil DataFrame.
    """
    id_table = entity.get("id_table")
    if not id_table:
        return entity.get("page_content", "")

    api_key = st.secrets.get("BPS_API_KEY", "")
    url = f"https://webapi.bps.go.id/v1/api/view/domain/3273/model/statictable/lang/ind/id/{id_table}/key/{api_key}/"
    df = parse_bps_table_structure_adaptive(url)
    return df.to_json(orient="records", force_ascii=False, indent=2)