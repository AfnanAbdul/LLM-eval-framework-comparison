from flask import Flask, render_template, request, jsonify
import os
from openai import OpenAI
import chromadb

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from trulens.core import TruSession, Select
from trulens_eval.feedback import GroundTruthAgreement
from trulens.core.feedback import Feedback
from trulens.core.instruments import instrument
from trulens.providers.openai import OpenAI  
from trulens_eval import TruCustomApp
import numpy as np

app = Flask(__name__)


import json

with open("config.json") as f:
    config = json.load(f)

api_key = config["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

oai_client = OpenAI()
tru = TruSession()


provider = OpenAI(model_engine="gpt-4")

# Define a groundedness feedback function
f_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons, name="Groundedness"
    )
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
)
# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input()
    .on_output()
)

# Context relevance between question and each context chunk.
f_context_relevance = (
    Feedback(
        provider.context_relevance_with_cot_reasons, name="Context Relevance"
    )
    .on_input()
    .on(Select.RecordCalls.retrieve.rets[:])
    .aggregate(np.mean)  # choose a different aggregation method if you wish
)

class RAG_from_scratch:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    @instrument
    def retrieve(self, query: str) -> list:
        results = self.vector_store.query(query_texts=query, n_results=2)
        return results["documents"][0]

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        completion = (
            oai_client.chat.completions.create(
                model="gpt-4-turbo-preview	",
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": f"We have provided context information below. \n"
                        f"---------------------\n"
                        f"{context_str}"
                        f"\n---------------------\n"
                        f"Given this information, please answer the question: {query}",
                    }
                ],
            )
            .choices[0]
            .message.content
        )
        return completion

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve(query)
        completion = self.generate_completion(query, context_str)
        return completion
    
def index():
    return render_template("index.html")

@app.route("/process_query", methods=["POST"])
def process_query():
    try:
        data = request.json
        university_info = data.get("university_info")
        query_text = data.get("query")

        if not university_info or not query_text:
            return jsonify({"error": "Both university information and query are required."}), 400

        embedding_function = OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-ada-002")
        chroma_client = chromadb.Client()
        vector_store_local = chroma_client.get_or_create_collection(name="CustomerProductData", embedding_function=embedding_function)
        vector_store_local.add("uni_info", documents=university_info)

        rag = RAG_from_scratch(vector_store_local)
        tru_rag = TruCustomApp(rag, app_id="CS_RAG_v1", feedbacks=[f_groundedness, f_qa_relevance, f_context_relevance])
        with tru_rag as recording:
            result = rag.query(query_text)

        tru.get_leaderboard(app_ids=["CS_RAG_v1"])
        tru.run_dashboard()

        return jsonify({"result": result})
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500
def start_dashboard():
    os.system("streamlit run dashboard.py") 
    
if __name__ == "__main__":
    threading.Thread(target=start_dashboard).start()
    app.run(debug=False, host="0.0.0.0", port=4000)