from flask import Flask, render_template, request, jsonify
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from transformers import pipeline

app = Flask(__name__, template_folder="templates")  # Specify template folder


# Initialize LLM
llm = HuggingFacePipeline(pipeline=pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=200
))

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    if "data-file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["data-file"]
    file_path = "data/" + file.filename
    file.save(file_path)

    # Load data
    loader = CSVLoader(file_path=file_path)
    data = loader.load()

    # Create vector database
    vectorstore = Chroma.from_documents(data, embeddings)

    # Initialize QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # Get user question
    question = request.form.get("question", "")

    # Get answer
    answer = qa_chain.invoke(question)

    return jsonify({"answer": answer["result"]})  # Ensure correct key


if __name__ == "__main__":
    app.run(debug=True)
