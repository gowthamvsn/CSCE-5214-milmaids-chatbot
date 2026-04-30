<h1 align="center">Mil Maids Knowledge Chatbot</h1>
<p align="center">
  A RAG-powered chatbot that answers questions about Mil Maids residential cleaning services using a local knowledge base and OpenAI.
</p>

---

## What It Does

This project builds a question-answering chatbot for **Mil Maids**, a residential cleaning company serving military communities. It uses **Retrieval-Augmented Generation (RAG)**: instead of relying on the model's general knowledge, it retrieves relevant passages from a local knowledge base and uses them to generate accurate, grounded answers.

The bot can be run as a **command-line interface** (`main.py`) or as a **Streamlit web UI** (`streamlit_app.py`).

---

## How It Answers Questions

When a question is asked, the bot follows these steps:

1. **Embed the question** — converts the query to a vector using `text-embedding-3-small`
2. **Search the index** — computes cosine similarity between the query embedding and all stored document chunk embeddings
3. **Retrieve top chunks** — selects the 2 most relevant passages from the knowledge base
4. **Check similarity threshold** — if the best match scores below 0.35, it skips the model and responds with a fallback directing the user to contact Mil Maids
5. **Generate an answer** — passes the retrieved chunks to `gpt-3.5-turbo` with instructions to answer only from the provided sources

For pricing questions, a dedicated code path extracts service type and square footage from the question, calculates an estimate, and returns it without calling the model.

**Index caching:** On first run, embeddings are generated and saved to `knowledge/documents.json` and `knowledge/embeddings.json`. Subsequent runs load from disk, so the OpenAI Embeddings API is only called when the index is refreshed.

---

## What You Can Ask

### Service questions
- *"What is included in a standard cleaning?"*
- *"What's the difference between a standard clean and a deep clean?"*
- *"Do I need to be home during the cleaning?"*
- *"Do you clean inside appliances?"*
- *"Can I add oven cleaning to my booking?"*

### Pricing and quotes
- *"How much does a standard clean cost for a 1,500 sq ft home?"*
- *"Can you quote me for a 2000 sqft deep clean with oven cleaning?."*
- *"What are your add-on services?"*
- Supported add-ons: oven cleaning, refrigerator cleaning, dishwasher cleaning, window cleaning, blinds, laundry, wall washing, baseboards, porch sweep, fireplace ash sweep, cabinet cleaning, linen change, and more
- How much does a standard cleaning cost?
- What is the price of a deep clean?


### Service areas
- *"Do you serve San Antonio?"*
- *"Which cities near Fort Hood do you cover?"*
- *"Do you clean homes in Fayetteville, NC?"*

### Booking and general
- *"How do I book a cleaning?"*
- *"Do you bring your own supplies?"*
- *"What if my home is very cluttered?"*

### Escalation to a live agent
If the home is described as **excessively cluttered, very filthy, or has excessive pet hair**, the bot will not generate an automated quote. It instead provides Mil Maids contact information and directs the customer to speak with a team member directly.

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/robindekoster/chatgpt-custom-knowledge-chatbot.git
cd chatgpt-custom-knowledge-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-api-key-here
```

### 4. Add your knowledge documents
Place `.txt` files in the `knowledge/` directory. Each file is chunked into 300-word passages with a 50-word overlap before being embedded.

> **Note:** Only `.txt` files are supported. The included knowledge base covers Mil Maids services, pricing, FAQ, and service areas.

### 5. Run the CLI
```bash
python main.py
```
You will be prompted to choose a bot:
- **`simple_vector_index`** — the full RAG chatbot (recommended)
- **`chat_completion`** — a basic demo with hardcoded knowledge (not connected to the knowledge base)

### 6. Run the Streamlit UI
```bash
streamlit run streamlit_app.py
```
Opens a web interface where you can type questions and click **"Refresh knowledge index"** to regenerate embeddings after updating the knowledge files.

---

## Project Structure

```
├── main.py                  # CLI entry point; lets you choose which bot to run
├── streamlit_app.py         # Streamlit web UI
├── bots/
│   ├── simple_vector_index.py   # RAG bot: chunking, embeddings, retrieval, generation
│   └── chat_completion.py       # Basic demo bot (hardcoded knowledge)
├── knowledge/
│   ├── *.txt                    # Source knowledge files (editable)
│   ├── documents.json           # Cached chunked documents (auto-generated)
│   └── embeddings.json          # Cached embeddings (auto-generated)
├── .env                     # Your OpenAI API key (not committed)
└── requirements.txt
```

---

## Technologies Used

- [OpenAI API](https://openai.com/) — `text-embedding-3-small` for embeddings, `gpt-3.5-turbo` for generation
- [Streamlit](https://streamlit.io/) — web UI
- [python-dotenv](https://pypi.org/project/python-dotenv/) — API key management

---

## License

MIT License. See `LICENSE` for details.
