import io
import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from question_filter_utils import QuestionFilterUtils
from question_generation_utils import QuestionGenerationUtils



app = Flask(__name__)
CORS(app)
device = "cuda"

def load_models():
    generator = QuestionGenerationUtils('cuda', 'upb-nlp/llama3.1_8b_qall_with_explanations')
    filters = QuestionFilterUtils('cuda')
    return generator, filters

@app.route('/generate', methods=['POST'] )
def generate_quiz():
    args = request.json
    if "token" not in args:
        return "Token not provided", 403
    if args["token"] != os.getenv("API_TOKEN"):
        return "Invalid token", 403
    if "context" not in args:
        return "Context not provided", 400
    context = args["context"]
    if len(context.split()) > 2048:
        return "Maximum context length = 1000 words", 400
    num_questions = args.get("num_questions", 10)
    questions = generator.generate_all_artifacts_with_explanations(context, num_questions)
    questions = filters.filter_questions(questions, context)[:num_questions]
    questions = filters.clean_response_dict(questions)
    return jsonify({"questions": questions}), 200

if __name__ == '__main__':
    print("Starting server...")
    generator, filters = load_models()
    app.run(host='0.0.0.0', port=5002)