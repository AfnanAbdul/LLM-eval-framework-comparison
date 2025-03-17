from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, faithfulness
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_anthropic import ChatAnthropic
import pandas as pd
import copy
import numpy as np
from dotenv import load_dotenv
import os
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load environment files and store API keys
load_dotenv()
anthropic_key = os.getenv("ANTHROPIC_KEY")

# Read in dialogue data
file_path = "../../data/Hallucination/dialogue_data.json"
dialogue_data = pd.read_json(file_path, lines=True)

# Read in general data
file_path = "../../data/Hallucination/general_data.json"
general_data = pd.read_json(file_path, lines=True)

# Read in QA data
file_path = "../../data/Hallucination/qa_data.json"
qa_data = pd.read_json(file_path, lines=True)

# Generate a random choice (True for right_answer, False for hallucinated_answer)
choice = np.random.rand(len(qa_data)) < 0.5

# Assign the chosen answer
qa_data["selected_answer"] = np.where(choice, qa_data["right_answer"], qa_data["hallucinated_answer"])

# Add a column indicating the source of the answer
qa_data["hallucinated_flag"] = np.where(choice, 0, 1)

# Set up Claude as the evaluator LLM
claude_llm = ChatAnthropic(api_key=anthropic_key, model="claude-3-7-sonnet-20250219")
evaluator_llm = LangchainLLMWrapper(claude_llm)

# Make a copy to use for hugging face dataset
qa_data_ragas = copy.deepcopy(qa_data[["question", "knowledge", "selected_answer"]])

# Rename columns according to methods
qa_data_ragas = qa_data_ragas.rename(columns=
    {"question": "user_input", 
     "knowledge": "retrieved_contexts", 
     "selected_answer":"response"
     })

# Turn context field values into lists
qa_data_ragas['retrieved_contexts'] = qa_data_ragas['retrieved_contexts'].map(lambda x: [x])

# Turn data frame into list
qa_data_ragas_dict = qa_data_ragas.to_dict(orient='list')

# Read in as dataset
dataset = Dataset.from_dict(qa_data_ragas_dict)

# Run evaluations, convert to data frame, and save results
score = evaluate(dataset,metrics=[faithfulness], llm=evaluator_llm)
qa_results = score.to_pandas()
qa_results.to_csv("../../results/Hallucination/Ragas/qa_results.csv", index=False)