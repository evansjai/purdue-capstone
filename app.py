from flask import Flask, render_template, request, jsonify
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import CSVLoader
from transformers import pipeline
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import faiss
import plotly.express as px

app = Flask(__name__, template_folder="templates")

llm = HuggingFacePipeline(pipeline=pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=200
))

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

    df = pd.read_csv(file_path)
    categorical_columns = ['Date','Product','Region','Sales','Customer_Age','Customer_Gender','Customer_Satisfaction']# add the columns here.
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    pca = PCA(n_components=16)
    tabular_embeddings = pca.fit_transform(df_encoded)

    index = faiss.IndexFlatL2(tabular_embeddings.shape[1])
    index.add(np.array(tabular_embeddings).astype("float32"))

    question = request.form.get("question", "")
    query_vector = pca.transform(df_encoded.iloc[[0]])
    D, I = index.search(np.array(query_vector).astype("float32"), k=5)
    results = df.iloc[I.flatten()].to_dict(orient="records")

    formatted_results = "\n".join([f"Product: {item['Product']}, Region: {item['Region']}, Sales: {item['Sales']}" for item in results])
    prompt = f"Based on the following sales data:\n{formatted_results}\nAnswer the question: {question}"
    response = llm.invoke(prompt)

    if "plot" in question.lower() or "chart" in question.lower():
        if "sales over time" in question.lower() or "sales by date" in question.lower():
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                fig = px.line(df, x="Date", y="Sales", title="Sales Over Time")
                plotly_chart = fig.to_html(full_html=False)

                return jsonify({"answer": response, "plotly_chart": plotly_chart})
            except Exception as e:
                return jsonify({"answer": f"{response}. Could not create sales over time plot. Error: {e}"})

        elif "sales by region" in question.lower():
            try:
                sales_by_region = df.groupby("Region")["Sales"].sum().reset_index()
                fig = px.bar(sales_by_region, x="Region", y="Sales", title="Total Sales by Region")
                plotly_chart = fig.to_html(full_html=False)
                return jsonify({"answer": response, "plotly_chart": plotly_chart})
            except Exception as e:
                return jsonify({"answer": f"{response}. Could not create sales by region plot. Error: {e}"})

        elif "customer age" in question.lower() and "sales" in question.lower():
            try:
                fig = px.scatter(df, x="Customer_Age", y="Sales", title="Sales vs. Customer Age")
                plotly_chart = fig.to_html(full_html=False)
                return jsonify({"answer": response, "plotly_chart": plotly_chart})
            except Exception as e:
                return jsonify({"answer": f"{response}. Could not create sales vs customer age plot. Error: {e}"})
        else:
            return jsonify({"answer": f"{response}. Could not determine the plot type."})

    else:
        return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)