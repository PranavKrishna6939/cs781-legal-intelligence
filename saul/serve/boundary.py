"""Conservative candidate eligibility diagnostic; no learned or reference features."""
import math
import re

HEADER = re.compile(
    r'\b(?:IN\s+THE\s+SUPREME\s+COURT\s+OF\s+INDIA|'
    r'(?:CIVIL|CRIMINAL)\s+(?:APPELLATE|ORIGINAL)\s+JURISDICTION|'
    r'RECORD\s+OF\s+PROCEEDINGS)\b')
CLOSURE = re.compile(
    r'\b(?:appeals?|petitions?)\b.{0,100}\b(?:dismissed|allowed|disposed)\b|'
    r'\bno\s+order\s+as\s+to\s+costs\b|'
    r'\bfor\s+the\s+reasons\s+aforementioned\b', re.IGNORECASE)
POLICY = 'closure_then_new_case_header_v1'


def boundary_choice(texts, scores, prompt, native_index):
    """Keep native choice unless it closes/restarts a case and an alternative exists.

    Header/closure context in the prompt exempts the entire pool. A candidate is
    flagged only for a canonical uppercase header after >=24 generated words,
    with a disposal/closure phrase in its preceding 100 words. Mean MBR utilities
    and their pseudo-reference pool are unchanged. This is an unproven heuristic,
    not a claim that every flagged transition is invalid.
    """
    if not isinstance(prompt, str) or len(prompt) > 100000:
        raise ValueError('Invalid boundary prompt')
    if not 1 <= len(texts) <= 128 or len(texts) != len(scores):
        raise ValueError('Need matching candidates and MBR scores')
    if not 0 <= native_index < len(texts) or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError('Invalid native index or MBR scores')
    tail = ' '.join(prompt.split()[-128:])
    exempt = bool(HEADER.search(tail) or CLOSURE.search(' '.join(tail.split()[-64:])))
    flagged = []
    for text in texts:
        flag = False
        if not exempt:
            for match in HEADER.finditer(text):
                before = text[:match.start()].split()
                if len(before) >= 24 and CLOSURE.search(' '.join(before[-100:])):
                    flag = True
                    break
        flagged.append(flag)
    eligible = [i for i, flag in enumerate(flagged) if not flag]
    chosen = native_index
    if flagged[native_index] and eligible:
        chosen = max(eligible, key=lambda i: scores[i])
    return dict(policy=POLICY, selected_index=chosen, flagged=flagged,
                changed=chosen != native_index, prompt_exempt=exempt,
                fallback_all_flagged=not eligible)
