import random
import time
import json
import argparse
import anthropic
from dotenv import load_dotenv
import os

# Load environment files and store API keys
load_dotenv()
anthropic_key = os.getenv("ANTHROPIC_KEY")

# Replace OpenAI with Anthropic's Claude
client = anthropic.Anthropic(api_key=anthropic_key)

def get_qa_response(model, question, answer, instruction):
    message = instruction + "\n\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:"
    
    system_content = "You are a hallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge. The answer you provided MUST be \"Yes\" or \"No\""
    
    while True:
        try:
            response = client.messages.create(
                model=model,
                system=system_content,
                messages=[
                    {"role": "user", "content": message}
                ],
                temperature=0.0,
                max_tokens=100
            )
            # Get the response text
            response_text = response.content[0].text
            break
        except anthropic.RateLimitError:
            print('anthropic.RateLimitError\nRetrying...')
            time.sleep(60)
        except anthropic.APIError:
            print('anthropic.APIError\nRetrying...')
            time.sleep(20)
        except anthropic.APIConnectionError:
            print('anthropic.APIConnectionError\nRetrying...')
            time.sleep(20)
        except Exception as e:
            print(f'Unexpected error: {e}\nRetrying...')
            time.sleep(20)
    
    return response_text


def get_dialogue_response(model, dialog, response, instruction):
    message = instruction + "\n\n#Dialogue History#: " + dialog + "\n#Response#: " + response + "\n#Your Judgement#:"
    
    system_content = "You are a response judge. You MUST determine if the provided response contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""

    while True:
        try:
            response = client.messages.create(
                model=model,
                system=system_content,
                messages=[
                    {"role": "user", "content": message}
                ],
                temperature=0.0,
                max_tokens=100
            )
            # Get the response text
            response_text = response.content[0].text
            break
        except anthropic.RateLimitError:
            print('anthropic.RateLimitError\nRetrying...')
            time.sleep(60)
        except anthropic.APIError:
            print('anthropic.APIError\nRetrying...')
            time.sleep(20)
        except anthropic.APIConnectionError:
            print('anthropic.APIConnectionError\nRetrying...')
            time.sleep(20)
        except Exception as e:
            print(f'Unexpected error: {e}\nRetrying...')
            time.sleep(20)

    return response_text


# No need for token counting functions as Claude handles context windows differently
# Note: If needed, you can still implement token counting for Claude using their tooling


def get_summarization_response(model, document, summary, instruction):
    message = instruction + "\n\n#Document#: " + document + "\n#Summary#: " + summary + "\n#Your Judgement#:"
    
    system_content = "You are a summary judge. You MUST determine if the provided summary contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""

    while True:
        try:
            response = client.messages.create(
                model=model,
                system=system_content,
                messages=[
                    {"role": "user", "content": message}
                ],
                temperature=0.0,
                max_tokens=100
            )
            # Get the response text
            response_text = response.content[0].text
            break
        except anthropic.RateLimitError:
            print('anthropic.RateLimitError\nRetrying...')
            time.sleep(60)
        except anthropic.APIError:
            print('anthropic.APIError\nRetrying...')
            time.sleep(20)
        except anthropic.APIConnectionError:
            print('anthropic.APIConnectionError\nRetrying...')
            time.sleep(20)
        except Exception as e:
            print(f'Unexpected error: {e}\nRetrying...')
            time.sleep(20)

    return response_text


def evaluation_qa_dataset(model, file, instruction, output_path):
    
    start_time_total = time.time()
    
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        for i in range(len(data)):
            start_time_sample = time.time()
            
            knowledge = data[i]["knowledge"]
            question = data[i]["question"]
            hallucinated_answer = data[i]["hallucinated_answer"]
            right_answer = data[i]["right_answer"]

            if random.random() > 0.5:
                answer = hallucinated_answer
                ground_truth = "Yes"
            else:
                answer = right_answer
                ground_truth = "No"

            ans = get_qa_response(model, question, answer, instruction)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                
                sample_time = time.time() - start_time_sample
                print('sample {} fails...... (took {:.2f}s)'.format(i, sample_time))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
                incorrect += 1

            assert(gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            sample_time = time.time() - start_time_sample
            print('sample {} success...... (took {:.2f}s)'.format(i, sample_time))
            dump_jsonl(gen, output_path, append=True)

        total_time = time.time() - start_time_total
        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct/len(data)))
        print('Total evaluation time: {:.2f}s ({:.2f}min)'.format(total_time, total_time/60))


def evaluation_dialogue_dataset(model, file, instruction, output_path):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        for i in range(len(data)):
            knowledge = data[i]["knowledge"]
            dialog = data[i]["dialogue_history"]
            hallucinated_response = data[i]["hallucinated_response"]
            right_response = data[i]["right_response"]

            if random.random() > 0.5:
                response = hallucinated_response
                ground_truth = "Yes"
            else:
                response = right_response
                ground_truth = "No"

            ans = get_dialogue_response(model, dialog, response, instruction)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
            assert (gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            print('sample {} success......'.format(i))
            dump_jsonl(gen, output_path, append=True)

        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct / len(data)))


def evaluation_summarization_dataset(model, file, instruction, output_path):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        for i in range(len(data)):

            document = data[i]["document"]
            hallucinated_summary = data[i]["hallucinated_summary"]
            right_summary = data[i]["right_summary"]

            if random.random() > 0.5:
                summary = hallucinated_summary
                ground_truth = "Yes"
            else:
                summary = right_summary
                ground_truth = "No"

            ans = get_summarization_response(model, document, summary, instruction)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
            assert (gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            print('sample {} success......'.format(i))
            dump_jsonl(gen, output_path, append=True)

        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct / len(data)))


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hallucination Generation")

    parser.add_argument("--task", default="qa", help="qa, dialogue, or summarization")
    parser.add_argument("--model", default="claude-3-5-haiku-20241022", help="model name")
    args = parser.parse_args()

    instruction_file = "../../data/Hallucination/{}_evaluation_instruction.txt".format(args.task)
    f = open(instruction_file, 'r', encoding="utf-8")
    instruction = f.read()

    model = args.model
    output_path = "../../results/Hallucination/HaluEval/{}/{}_{}_results_sample.json".format(args.task, args.task, args.model)

    data = "../../data/Hallucination/{}_data_sample.json".format(args.task)

    if args.task == "qa":
        evaluation_qa_dataset(model, data, instruction, output_path)
    elif args.task == "dialogue":
        evaluation_dialogue_dataset(model, data, instruction, output_path)
    elif args.task == "summarization":
        evaluation_summarization_dataset(model, data, instruction, output_path)
    else:
        raise ValueError("The task must be qa, dialogue, or summarization!")
    
# python HaluEval_evaluation.py --task qa --model claude-3-5-haiku-20241022