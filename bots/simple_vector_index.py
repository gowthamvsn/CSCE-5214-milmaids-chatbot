from dotenv import load_dotenv
import os
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext, Document


def load_knowledge() -> list[Document]:
    # Load data from directory
    documents = SimpleDirectoryReader('knowledge').load_data()
    return documents


def create_index() -> GPTVectorStoreIndex:
    print('Creating new index')
    documents = load_knowledge()
    from llama_index import LLMPredictor
    from langchain.chat_models import ChatOpenAI
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        chunk_size_limit=3000
    )
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    return index

def save_index(index: GPTVectorStoreIndex):
    # Save index to file
    index.save_to_disk('knowledge/index.json')


def load_index() -> GPTVectorStoreIndex:
    # Load index from file
    try:
        index = GPTVectorStoreIndex.load_from_disk('knowledge/index.json')
    except FileNotFoundError:
        index = create_index()
    return index


def query_index(index: GPTVectorStoreIndex):
    while True:
        prompt = input("Type prompt...")
        response = index.query(prompt)
        print(response)


def main():
    # Ask user if they want to refresh the index
    refresh_index = input("Do you want to refresh the index? (y/n) [n]: ")
    refresh_index = refresh_index.lower() == 'y'

    # If refreshing the index, create new index and save to file
    if refresh_index:
        index = create_index()
    # Otherwise, load index from file
    else:
        index = load_index()

    # Query index
    query_index(index)


if __name__ == '__main__':
    main()
