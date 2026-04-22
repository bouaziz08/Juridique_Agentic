from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from time import time


# Constants
CHROMA_PATH = "./chroma"  # database path
PROMPT_TEMPLATE = """{context}

Question: {question}

Answer:"""


class RAGQueryProcessor:
    def __init__(self, chroma_path: str, model_name: str, precision: str = "fp16", max_tokens: int = 50):
        self.chroma_path = chroma_path
        self.model_name = model_name
        self.precision = precision
        self.max_tokens = max_tokens
        self.embedding_function = get_embedding_function()
        self.db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_function)

    def _search_db(self, query_text: str):
        """
        Search the Chroma DB for relevant documents based on the query.
        """
        return self.db.similarity_search_with_score(query_text, k=3)

    def _build_prompt(self, results, query_text: str):
        """
        Build the prompt by combining the context and the query text.
        """
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        return prompt_template.format(context=context_text, question=query_text)

    def _query_model(self, prompt: str):
        """
        Query the model (Ollama) with the constructed prompt.
        """
        model = OllamaLLM(model=self.model_name, precision=self.precision, max_tokens=self.max_tokens)
        return model.invoke(prompt)

    def process_query(self, query_text: str):
        """
        Full process to handle the query, search, generate response, and format the output.
        """
        start_time = time()

        # Search the DB for relevant documents
        results = self._search_db(query_text)

        # Build the prompt for the model
        prompt = self._build_prompt(results, query_text)

        # Query the model to get the response
        response_text = self._query_model(prompt)

        # Format the output
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = {
            "response": response_text,
            "sources": sources,
            "time_taken": f"{time() - start_time:.2f} seconds"
        }

        return formatted_response


# Instantiate the processor
rag_processor = RAGQueryProcessor(chroma_path=CHROMA_PATH, model_name="mistral")
