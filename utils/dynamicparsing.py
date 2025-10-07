import requests
import pandas as pd
import re

def clean_label(s):
    """Hapus tag HTML kecil-kecilan dan strip."""
    if s is None: 
        return s
    return re.sub(r"<[^>]+>", "", str(s)).strip()

def parse_bps_api(url: str, clean_html: bool = True, promote_subvar_if_tidak_ada: bool = False):
    """
    Parse respon API BPS menjadi dataframe 'data mentah' (one row per datacontent).
    - clean_html: jika True, akan membersihkan tag HTML pada label.
    - promote_subvar_if_tidak_ada: jika True dan turvar hanya 'Tidak ada', maka
      Subvariabel akan diisi dengan nilai Variabel (promote).
    """
    resp = requests.get(url).json()

    # mapping value->label (kunci disimpan sebagai string)
    vervar_map   = {str(v["val"]): v["label"] for v in resp.get("vervar", [])}
    var_map      = {str(v["val"]): v["label"] for v in resp.get("var", [])}
    turvar_map   = {str(t["val"]): t["label"] for t in resp.get("turvar", [])}
    tahun_map    = {str(t["val"]): t["label"] for t in resp.get("tahun", [])}
    turtahun_map = {str(t["val"]): t["label"] for t in resp.get("turtahun", [])}

    # decimal precision (ambil dari metadata var bila ada)
    try:
        decimal_prec = int(resp.get("var", [{}])[0].get("decimal", 2))
    except Exception:
        decimal_prec = 2

    def format_number(v):
        if isinstance(v, (int, float)):
            s = format(v, f",.{decimal_prec}f")   # menghasilkan "1,234.56"
            # swap to "1.234,56"
            return s.replace(",", "X").replace(".", ",").replace("X", ".")
        return v

    # prepare candidate keys (urutan: coba yang paling panjang dulu -> kurangi cabang)
    vervar_keys   = sorted(vervar_map.keys(), key=len, reverse=True)
    var_keys      = sorted(var_map.keys(), key=len, reverse=True)
    turvar_keys   = sorted(turvar_map.keys(), key=len, reverse=True)
    tahun_keys    = sorted(tahun_map.keys(), key=len, reverse=True)
    turtahun_keys = sorted(turtahun_map.keys(), key=len, reverse=True)

    # urutan token sesuai spec
    keys_list = [vervar_keys, var_keys, turvar_keys, tahun_keys, turtahun_keys]

    # DFS + memo untuk cari partition yang tepat sehingga concat(tokens) == key
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def dfs(i, rem):
        """
        return tuple of matched keys from position i...end if success, otherwise None
        i: index in keys_list
        rem: remaining string to match
        """
        if i == len(keys_list):
            return () if rem == "" else None
        keys = keys_list[i]
        # jika tidak ada kandidat untuk posisi ini, skip (treat as optional)
        if not keys:
            return dfs(i+1, rem)
        # coba semua kandidat (yang prefix dari rem)
        for k in keys:
            if rem.startswith(k):
                rest = dfs(i+1, rem[len(k):])
                if rest is not None:
                    return (k,) + rest
        return None

    data_rows = []
    for raw_key, value in resp.get("datacontent", {}).items():
        s = str(raw_key)
        matched = dfs(0, s)

        if matched is None:
            # fallback: jika DFS gagal, simpan raw_key agar bisa debug kemudian
            # tetap coba greedy prefix (panjang terbesar first) untuk salvage
            def greedy_match(s):
                out = []
                rem = s
                for keys in keys_list:
                    found = None
                    for k in keys:
                        if rem.startswith(k):
                            found = k
                            rem = rem[len(k):]
                            break
                    out.append(found)
                return tuple(out)
            matched = greedy_match(s)

        # matched is tuple of 5 (some items may be None)
        # map to labels (fallback to the matched string if label missing)
        try:
            v_v = matched[0]
            v_var = matched[1]
            v_tur = matched[2]
            v_thn = matched[3]
            v_tth = matched[4]
        except Exception:
            v_v = v_var = v_tur = v_thn = v_tth = None

        vervar_label   = vervar_map.get(v_v, v_v)
        var_label      = var_map.get(v_var, v_var)
        turvar_label   = turvar_map.get(v_tur, v_tur)
        tahun_label    = tahun_map.get(v_thn, v_thn)
        turtahun_label = turtahun_map.get(v_tth, v_tth)

        if clean_html:
            vervar_label   = clean_label(vervar_label)
            var_label      = clean_label(var_label)
            turvar_label   = clean_label(turvar_label)
            tahun_label    = clean_label(tahun_label)
            turtahun_label = clean_label(turtahun_label)

        # optionally, kalau turvar hanya 'Tidak ada' dan user mau promote
        if promote_subvar_if_tidak_ada and (len(turvar_map) == 1 and list(turvar_map.values())[0].lower().strip() in ("tidak ada", "0", "none")):
            subvar_final = var_label
        else:
            subvar_final = turvar_label

        row = {
            resp.get("labelvervar", "Vervar"): vervar_label,
            "Variabel": var_label,
            "Subvariabel": subvar_final,
            "Tahun": tahun_label,
            "TurTahun": turtahun_label,
            "Nilai": format_number(value),
            "RawKey": raw_key if (None in matched or matched is None) else None
        }
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    # drop RawKey column bila tidak ada issue
    if 'RawKey' in df.columns and df['RawKey'].isna().all():
        df = df.drop(columns=['RawKey'])
    return df

def parse_table_dynamic(entity: dict) -> str:
    """
    Wrapper untuk dynamic table (tablesource=2).
    Ambil id_table & tahun terbaru dari entity, bangun URL API.
    """
    id_table = entity.get("id_table")
    years = entity.get("years") or []
    if not id_table or not years:
        return entity.get("page_content", "")
    
    year_latest = max(years)
    th_code = year_latest - 1900
    
    api_key = st.secrets.get("BPS_API_KEY", "")
    url = f"https://webapi.bps.go.id/v1/api/list/model/data/lang/ind/domain/3273/var/{id_table}/th/{th_code}/key/{api_key}/"
    df = parse_bps_api(url, clean_html=True)
    return df.to_string(index=False)
