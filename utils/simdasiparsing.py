import requests
import pandas as pd
import re

def clean_html(text: str) -> str:

    # Membersihkan teks dari tag HTML sederhana, misalnya <i>, <br>, dll.
    if text is None:
        return None
    return re.sub(r"<[^>]+>", "", str(text)).strip()

def parse_simdasi(url: str, include_kode_wilayah: bool = False) -> pd.DataFrame:
    """
    Parsing data dari API SIMDASI.
    - Kolom pertama diberi nama sesuai lingkup_id (misalnya 'Provinsi', 'Kabupaten', 'Kecamatan').
    - Nama variabel dibersihkan dari tag HTML.
    - Nilai kosong tetap dipertahankan sebagai None.
    - Bisa pilih apakah kode wilayah ikut ditampilkan.
    """
    resp = requests.get(url).json()
    if resp.get("status") != "OK":
        raise ValueError(f"Response error: {resp}")

    meta = resp["data"][1]

    # Nama kolom untuk wilayah sesuai lingkup_id
    lingkup_col = meta.get("lingkup_id")

    # Peta kode variabel → nama variabel (dibersihkan HTML-nya)
    kolom_map = {
        kode: clean_html(info["nama_variabel"]) 
        for kode, info in meta["kolom"].items()
    }

    rows = []
    for row in meta["data"]:
        # Setiap baris punya label lingkup, misal nama kecamatan
        record = {lingkup_col: row["label"]}

        # Tambahkan kode wilayah bila diminta
        if include_kode_wilayah:
            record["KodeWilayah"] = row.get("kode_wilayah")

        # Tambahkan semua variabel indikator
        for kode, value in row["variables"].items():
            nama_var = kolom_map.get(kode, kode)
            record[nama_var] = value.get("value_raw", None)

        rows.append(record)

    df = pd.DataFrame(rows)
    return df

def parse_table_simdasi(entity: dict) -> str:
    """
    Wrapper untuk SIMDASI (tablesource=3).
    """
    id_table = entity.get("id_table")
    if not id_table:
        return entity.get("page_content", "")
    
    year_latest = max(years)
    
    api_key = st.secrets.get("BPS_API_KEY", "")
    url = f"https://webapi.bps.go.id/v1/api/interoperabilitas/datasource/simdasi/id/25/tahun/{year_latest}/id_tabel/{id_table}/wilayah/3273000/key/{api_key}"
    df = parse_simdasi(url, include_kode_wilayah=False)
    return df.to_string(index=False)
