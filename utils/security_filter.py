import re

def is_malicious(query: str) -> bool:
    """
    Simple heuristic filter untuk spam, SQL injection, prompt injection, abuse.
    """
    bad_patterns = [
        r"(drop\s+table)",
        r"(union\s+select)",
        r"(insert\s+into)",
        r"(delete\s+from)",
        r"(shutdown)",
        r"(system\s*\()",
        r"(http[s]?://.*malware)",
        r"(fuck|shit|blegug|anjing)",  
    ]
    for pat in bad_patterns:
        if re.search(pat, query, re.IGNORECASE):
            return True
    return False


def is_relevant_intent(intent: str) -> bool:
    """
    Pengecekan intent dari 5 koleksi yang ada
    """
    return intent in ["publications", "press", "news", "general", "tables"]
