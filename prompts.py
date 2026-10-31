import uuid

SYSTEM_TEMPLATE = (
    "[session:{salt}]\n"
    "You are a finance analyst assistant. Answer only using the document below. "
    "If the answer is not in the document, say so.\n\n"
    "DOCUMENT:\n{document}"
)


def new_salt() -> str:
    return uuid.uuid4().hex[:12]


def build_messages(document: str, question: str, salt: str) -> list:
    # Static content first (salt + instructions + document), dynamic question last.
    return [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(salt=salt, document=document)},
        {"role": "user", "content": question},
    ]