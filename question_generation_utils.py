import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

def entropy_based_logits_processor(token_ids, logits):
        probs = torch.nn.functional.softmax(logits, dim=0)
        log_probs = torch.nn.functional.log_softmax(logits, dim=0)
        entropy = -torch.sum(probs * log_probs)
        # Set logits to -inf if entropy + 1.1 < -log_probs
        logits[entropy + 1.1 < -log_probs] = float('-inf')
        return logits
    
class QuestionGenerationUtils:
    def __init__(self, device, model_name):
        if device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Invalid device: {device}. Must be one of 'cpu', 'cuda', 'mps'.")
        
        if not model_name:
            raise ValueError("model_name cannot be empty.")
        
        self.device = device
        self.llm = LLM(model=model_name, dtype=torch.bfloat16, max_model_len=4096, gpu_memory_utilization=0.4, enable_prefix_caching=True)
       
    
    def generate_all_artifacts_with_explanations(self, context, num_samples):
        params = SamplingParams(temperature=1, max_tokens=1024, n=min(5*num_samples, 50), top_p=0.9, min_p=0.2, logprobs=1, logits_processors=[entropy_based_logits_processor])
    
        messages = [
            {"role": "system", "content": "You are an educational expert."},
            {"role": "user", "content": f"Generate a question, an answer and 3 distractors based on the context.\n\n\n\nContext:\n{context}"}
        ]
        response = self.llm.chat(messages, params)
        
        response_list = []
        for output in response[0].outputs:
            try:
                question_text = output.text.split("Question:")[1].split("Answer explanation:")[0].strip()
                answer_explanation_text = output.text.split("Answer explanation:")[1].split("Answer:")[0].strip()
                answer_text = output.text.split("Answer:")[1].split("Distractors:")[0].strip()
                whole_distractors_text = output.text.split("Distractors:")[1].strip()
                
                splitted_whole_distractors_text = whole_distractors_text.split("\n\n")
                distractors_text = []
                for text in splitted_whole_distractors_text:
                    distractor_category = text.split("Distractor category:")[1].split("Distractor explanation:")[0].strip()
                    distractor_explanation = text.split("Distractor explanation:")[1].split("Distractor:")[0].strip()
                    distractor = text.split("Distractor:")[1].strip()
                    distractors_text.append({
                        'distractor_category_text': distractor_category,
                        'distractor_explanation_text': distractor_explanation,
                        'distractor_text': distractor,
                    })
                    
                response = {
                    'question_text': question_text,
                    'answer_explanation_text': answer_explanation_text,
                    'answer_text': answer_text,
                    'distractors': distractors_text,
                    "logprob": output.cumulative_logprob,
                }
                
                response_list.append(response)
            except:
                pass

        return response_list
