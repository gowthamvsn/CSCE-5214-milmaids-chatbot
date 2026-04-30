import json
import math
import os
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv
import openai

ROOT_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT_DIR / 'knowledge'
DOCS_PATH = KNOWLEDGE_DIR / 'documents.json'
EMBEDDINGS_PATH = KNOWLEDGE_DIR / 'embeddings.json'
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 2
MIN_SIMILARITY = 0.35
EMBEDDING_MODEL = 'text-embedding-3-small'
COMPLETION_MODEL = 'gpt-3.5-turbo'

BASE_PRICE = 150
RATE_PER_SQFT = 0.23
MIN_SQFT = 500
MAX_SQFT = 6000

load_dotenv(ROOT_DIR / '.env')
openai.api_key = os.getenv('OPENAI_API_KEY') or openai.api_key

if not openai.api_key:
    raise RuntimeError('OPENAI_API_KEY is not set. Please add it to .env or your environment.')


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.strip().split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def load_knowledge_documents() -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []
    for path in sorted(KNOWLEDGE_DIR.glob('*.txt')):
        text = path.read_text(encoding='utf-8').strip()
        if not text:
            continue
        for idx, chunk in enumerate(chunk_text(text), start=1):
            documents.append({
                'id': f'{path.stem}-{idx}',
                'source': path.name,
                'text': chunk,
            })
    return documents


def save_index(documents: List[Dict[str, str]], embeddings: List[List[float]]) -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    with DOCS_PATH.open('w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2)
    with EMBEDDINGS_PATH.open('w', encoding='utf-8') as f:
        json.dump(embeddings, f)


def load_index(refresh: bool = False) -> Tuple[List[Dict[str, str]], List[List[float]]]:
    if refresh or not DOCS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        return create_index()

    with DOCS_PATH.open('r', encoding='utf-8') as f:
        documents = json.load(f)
    with EMBEDDINGS_PATH.open('r', encoding='utf-8') as f:
        embeddings = json.load(f)
    return documents, embeddings


def embed_texts(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), 20):
        batch = texts[i:i + 20]
        response = openai.Embedding.create(input=batch, model=model)
        embeddings.extend(item['embedding'] for item in response['data'])
    return embeddings


def create_index() -> Tuple[List[Dict[str, str]], List[List[float]]]:
    print('Building knowledge embeddings...')
    documents = load_knowledge_documents()
    if not documents:
        raise RuntimeError('No documents found in the knowledge directory.')
    embeddings = embed_texts([doc['text'] for doc in documents])
    save_index(documents, embeddings)
    return documents, embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_top_chunks(
    prompt: str,
    documents: List[Dict[str, str]],
    embeddings: List[List[float]],
    top_k: int = TOP_K,
) -> List[Tuple[float, Dict[str, str]]]:
    query_embedding = embed_texts([prompt])[0]
    scored = [
        (cosine_similarity(query_embedding, emb), doc)
        for doc, emb in zip(documents, embeddings)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def should_escalate_to_live_quote(prompt: str) -> bool:
    lowered = prompt.lower()
    keywords = [
        'excessively cluttered',
        'very filthy',
        'excessive pet hair',
        'heavy buildup',
        'cluttered',
        'filthy',
        'excessive pet hair',
    ]
    return any(keyword in lowered for keyword in keywords)


def extract_quote_details(prompt: str) -> tuple:
    """Extract service type, sqft, and add-ons from a quote request.
    Returns: (service_type, sqft, add_ons) or (None, None, None) if not a clear quote request.
    """
    import re
    lowered = prompt.lower()
    
    service_type = None
    if 'standard clean' in lowered or 'standard cleaning' in lowered:
        service_type = 'standard'
    elif 'deep clean' in lowered or 'deep cleaning' in lowered:
        service_type = 'deep'
    elif 'move-in' in lowered or 'move-out' in lowered or 'move in' in lowered or 'move out' in lowered:
        service_type = 'move-in/out'
    
    sqft_match = re.search(r'(\d+)\s*(?:sq(?:uare)?\s*(?:ft|foot|feet)|sqft)', lowered)
    sqft = None
    if sqft_match:
        sqft = int(sqft_match.group(1))
    
    add_ons = []
    addon_keywords = {
        'oven cleaning': 50,
        'oven clean': 50,
        'refrigerator cleaning': 40,
        'fridge cleaning': 40,
        'dishwasher cleaning': 20,
        'window cleaning': 10,
        'blinds': 10,
        'baseboards': None,
        'laundry': 25,
        'wall washing': None,
    }
    for keyword, price in addon_keywords.items():
        if keyword in lowered:
            add_ons.append((keyword, price))
    
    return service_type, sqft, add_ons


def calculate_quote(service_type: str, sqft: int, add_ons: list) -> float:
    """Calculate price: BASE_PRICE + (RATE_PER_SQFT * sqft) + add-ons.
    Returns None if sqft is invalid.
    """
    if sqft < MIN_SQFT or sqft > MAX_SQFT:
        return None
    
    total = BASE_PRICE + (RATE_PER_SQFT * sqft)
    for addon_name, addon_price in add_ons:
        if addon_price is not None:
            total += addon_price
    return total


def should_use_generic_pricing_response(prompt: str) -> bool:
    lowered = prompt.lower()
    keywords = [
        'pricing formula',
        'how is pricing calculated',
        'how pricing is calculated',
        'pricing calculated',
        'price formula',
        'proprietary pricing',
        'pricing model',
        'price model',
        'how much does',
        'what is the cost',
        'how much is',
        'what is the price',
        'cost of',
        'price of',
        'quote',
    ]
    return any(keyword in lowered for keyword in keywords)


def build_prompt(prompt: str, top_chunks: List[Tuple[float, Dict[str, str]]]) -> str:
    sources = []
    for rank, (score, doc) in enumerate(top_chunks, start=1):
        sources.append(
            f"Source {rank}: {doc['source']}\n" + doc['text'] + '\n'
        )

    combined_sources = '\n---\n'.join(sources)
    return (
        "Use the content from the sources below to answer the question. "
        "Be concise and answer in 1-2 sentences. "
        "Do not add information that is not present in the sources. "
        "If the question cannot be answered from these sources, say you are unsure and direct the user to contact Mil Maids.\n\n"
        "If the user asks about pricing formulas, explain only that pricing depends on service type, home size, and add-ons. "
        "Do not reveal proprietary pricing formulas.\n\n"
        "Relevant sources:\n"
        f"{combined_sources}\n"
        f"Question: {prompt}\n"
        "Answer (keep it concise):"
    )


def generate_answer(prompt: str, top_chunks: List[Tuple[float, Dict[str, str]]]) -> str:
    if should_escalate_to_live_quote(prompt):
        return (
            "Thanks for letting us know. Because the home may require additional cleaning time due to clutter, heavy buildup, or excessive pet hair, "
            "a live Mil Maids team member will need to prepare your quote. Please contact us directly for custom pricing.\n"
            "Email: info@milmaids.com\n"
            "Fort Hood/Killeen, TX: (254) 419-0325\n"
            "JBSA/San Antonio, TX: (726) 200-3251\n"
            "Fort Bragg/Fayetteville, NC: (910) 900-6118"
        )
    
    service_type, sqft, add_ons = extract_quote_details(prompt)
    if service_type and sqft:
        price = calculate_quote(service_type, sqft, add_ons)
        if price is not None:
            addon_text = ""
            if add_ons:
                addon_names = [name for name, _ in add_ons if _ is not None]
                if addon_names:
                    addon_text = f" plus your requested add-on services ({', '.join(addon_names)})"
            return (
                f"Based on the information provided, your estimated cleaning price is approximately ${price:.0f}. "
                f"This estimate includes the {service_type} cleaning service{addon_text}. "
                f"Final pricing may vary depending on the home layout and condition."
            )
        else:
            return (
                "That estimate seems unusual based on the information provided. "
                "Let me confirm the details before generating a quote. "
                "Please contact us for a custom quote.\n"
                "Email: info@milmaids.com"
            )
    
    if should_use_generic_pricing_response(prompt):
        return (
            "Pricing depends on factors such as the type of cleaning service selected, the square footage of the home, "
            "and any requested add-on services. For custom pricing, please contact Mil Maids.\n"
            "Email: info@milmaids.com\nWebsite: https://milmaids.com"
        )

    if not top_chunks or top_chunks[0][0] < MIN_SIMILARITY:
        return (
            "I'm not sure based on the current Mil Maids knowledge. Please contact Mil Maids at info@milmaids.com or visit https://milmaids.com for help."
        )
    
    top_chunks = [chunk for chunk in top_chunks if chunk[0] >= MIN_SIMILARITY]

    messages = [
        {
            'role': 'system',
            'content': (
                'You are a helpful assistant for Mil Maids. Answer only using the provided knowledge content. '
                'If the answer is not directly supported by the sources, say you are unsure and encourage the user to contact Mil Maids. '
                'Do not hallucinate new details or invent services.'
            ),
        },
        {
            'role': 'user',
            'content': build_prompt(prompt, top_chunks),
        },
    ]

    response = openai.ChatCompletion.create(
        model=COMPLETION_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=450,
    )
    return response.choices[0].message.content.strip()


def query_answer(
    prompt: str,
    documents: List[Dict[str, str]],
    embeddings: List[List[float]],
    top_k: int = TOP_K,
) -> Tuple[str, List[Tuple[float, Dict[str, str]]]]:
    top_chunks = get_top_chunks(prompt, documents, embeddings, top_k=top_k)
    answer = generate_answer(prompt, top_chunks)
    return answer, top_chunks


def main() -> None:
    refresh = input('Refresh embeddings and documents index? (y/n) [n]: ').strip().lower() == 'y'
    documents, embeddings = load_index(refresh=refresh)
    while True:
        query = input('Ask a question about Mil Maids (type "exit" to quit): ').strip()
        if query.lower() in {'exit', 'quit'}:
            break
        if not query:
            continue
        answer, top_chunks = query_answer(query, documents, embeddings)
        print('\n=== Answer ===')
        print(answer)
        print('\n=== Top Sources ===')
        for score, doc in top_chunks:
            print(f"{doc['source']} (score={score:.3f})")
        print('\n')


if __name__ == '__main__':
    main()
