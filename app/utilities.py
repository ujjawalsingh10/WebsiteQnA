import re

def normalize_whitespace(text: str) -> str:
    # Collapse 3+ newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace on each line
    text = "\n".join(line.strip() for line in text.splitlines())

    return text.strip()

def remove_navigation(text: str):
    junk_lines = {
        "Home",
        "Articles"
    }
    lines = text.splitlines()
    cleaned = [l for l in lines if l not in junk_lines]
    return "\n".join(cleaned)

def remove_tag_lines(text: str):
    lines = text.splitlines()
    cleaned = [
        line for line in lines
        if not line.startswith("#")
    ]
    return "\n".join(cleaned)