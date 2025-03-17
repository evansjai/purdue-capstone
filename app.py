import pandas as pd
#import io
from flask import Flask, render_template, request, jsonify
from langchain_community.llms import HuggingFacePipeline
#from langchain.embeddings import OpenAIEmbeddings
#from langchain_community.vectorstores import Chroma
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import DirectoryLoader, CSVLoader
#from langchain.chains import RetrievalQA
from transformers import pipeline
#import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

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

    # Data Exploration and Preprocessing (from notebook)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Customer_Gender'] = df['Customer_Gender'].astype('category')
    df['Quarter'] = df['Date'].dt.quarter
    import calendar
    df['Date_Month'] = df.apply(lambda x: f"{x['Date'].year}-{calendar.month_abbr[x['Date'].month]}", axis=1)

    # Calculate Sales by Gender, Age, and Month
    sales_by_gender = df.groupby('Customer_Gender', observed=True)['Sales'].sum()
    bins = [18, 25, 35, 45, 55, 65, 100]
    labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    df['Age_Range'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels, right=False)
    sales_by_age = df.groupby('Age_Range', observed=True)['Sales'].sum()
    sales_by_month = df.groupby('Month')['Sales'].sum()

    # Calculate Sales by Gender and Age
    gender_age_sales = df.groupby(['Customer_Gender', 'Age_Range'])['Sales'].sum().reset_index()

    # Update the prompt
    question = request.form.get("question", "")
    if not question:
        question = "Summarize the key insights from the data."

    prompt = f"""
    Sales Data Summary:
    - Total Sales: {df['Sales'].sum()}
    - Average Sales: {df['Sales'].mean()}
    - Unique Products: {list(set(df['Product'].unique()))}
    - Unique Regions: {df['Region'].unique()}
    - Average Customer Age: {df['Customer_Age'].mean()}
    - Average Customer Satisfaction: {df['Customer_Satisfaction'].mean()}
    - Sales by Product: {df.groupby('Product')['Sales'].sum().to_string()}
    - Sales by Region: {df.groupby('Region')['Sales'].sum().to_string()}
    - Sales by Gender: {sales_by_gender.to_string()}
    - Sales by Age Range: {sales_by_age.to_string()}
    - Sales by Month: {sales_by_month.to_string()}
    - Sales by Gender and Age Range: {gender_age_sales.to_string()}
    
    Answer the question from the Sales Data Summary:
    {question}
    
    """

    response = llm.invoke(prompt)
    response = format_sales_data_to_html(response)

    # Visualization Logic (extended from notebook)
    if "plot" in question.lower() or "chart" in question.lower() or "visualize" in question.lower():
        if "satisfaction" in question.lower():
            df1 = df.groupby(by='Region', as_index=False)['Customer_Satisfaction'].mean()
            plt.figure(figsize=(6, 4))
            plt.bar(x=df1['Region'], height=df1['Customer_Satisfaction'])
            plt.title("Customer Satisfaction by Region")
            plt.xlabel("Region")
            plt.ylabel("Customer Satisfaction")
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return jsonify({"answer": response, "plotly_chart": f'<img src="data:image/png;base64,{plot_data}" />'})
        elif "quarter" in question.lower():
            sales_by_quarter_2024 = df[df['Date'].dt.year == 2024].groupby('Quarter')['Sales'].sum()
            plt.figure(figsize=(6, 3))
            plt.plot(sales_by_quarter_2024.index, sales_by_quarter_2024.values, marker='o', linestyle='-')
            plt.xlabel('Quarter')
            plt.ylabel('Total Sales')
            plt.title('Sales by Quarter 2024')
            plt.xticks(sales_by_quarter_2024.index)
            plt.grid(True)
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return jsonify({"answer": response, "plotly_chart": f'<img src="data:image/png;base64,{plot_data}" />'})
        elif "month" in question.lower():
            total_sales = df[df['Date'].dt.year == 2024].groupby(by='Date_Month', as_index=False)['Sales'].mean()
            plt.figure(figsize=(8, 4))
            sns.lineplot(x=total_sales['Date_Month'], y=total_sales['Sales'])
            plt.title("Total Sales by Date")
            plt.xlabel("Date_Month")
            plt.xticks(rotation=90)
            plt.ylabel("Total Sales")
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return jsonify({"answer": response, "plotly_chart": f'<img src="data:image/png;base64,{plot_data}" />'})
        elif "correlation" in question.lower():
            corrs = df[['Sales', 'Customer_Age', 'Customer_Satisfaction']].corr(method='pearson')
            plt.figure(figsize=(6, 4))
            sns.heatmap(corrs, annot=True, cmap='bwr')
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return jsonify({"answer": response, "plotly_chart": f'<img src="data:image/png;base64,{plot_data}" />'})
        elif "region" in question.lower():
            region_sales = df.groupby('Region')['Sales'].sum().reset_index()
            plt.figure(figsize=(6, 4))
            sns.barplot(x='Region', y='Sales', data=region_sales)
            plt.title('Sales by Region')
            plt.xlabel('Region')
            plt.ylabel('Sales')
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return jsonify({"answer": response, "plotly_chart": f'<img src="data:image/png;base64,{plot_data}" />'})
        elif "gender" in question.lower() and "age" in question.lower():
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Age_Range', y='Sales', hue='Customer_Gender', data=gender_age_sales)
            plt.title('Sales by Gender and Age Range')
            plt.xlabel('Age Range')
            plt.ylabel('Sales')
            plt.xticks(rotation=45, ha='right')
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return jsonify({"answer": response, "plotly_chart": f'<img src="data:image/png;base64,{plot_data}" />'})
        elif "gender" in question.lower() or "product" in question.lower() or "widget" in question.lower():
            product_gender_sales = df.groupby(['Product', 'Customer_Gender'])['Sales'].sum().reset_index()
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Product', y='Sales', hue='Customer_Gender', data=product_gender_sales)
            plt.title('Sales by Product and Gender')
            plt.xlabel('Product')
            plt.ylabel('Sales')
            plt.xticks(rotation=45, ha='right')
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return jsonify({"answer": response, "plotly_chart": f'<img src="data:image/png;base64,{plot_data}" />'})
        elif "heatmap" in question.lower():
            # Group and aggregate data
            gender_age_product_sales = df.groupby(['Customer_Gender', 'Age_Range', 'Product'])[
                'Sales'].sum().reset_index()

            # Get unique products
            products = gender_age_product_sales['Product'].unique()

            # Create multiple heatmaps, one for each product
            heatmap_images = []
            for product in products:
                product_data = gender_age_product_sales[gender_age_product_sales['Product'] == product]
                heatmap_data = product_data.pivot_table(index='Customer_Gender', columns='Age_Range', values='Sales',
                                                        fill_value=0)

                plt.figure(figsize=(10, 6))
                sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu")
                plt.title(f'Sales by Gender and Age Range for {product}')
                plt.xlabel('Age Range')
                plt.ylabel('Gender')

                buffer = BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                heatmap_images.append(f'<img src="data:image/png;base64,{plot_data}" />')
                plt.close()  # close plot to prevent overlaying plots.
            return jsonify({"answer": response, "plotly_chart": f'<img src="data:image/png;base64,{"".join(heatmap_images)}" />'})
        else:
            return jsonify({"answer": response, "plotly_chart": "Could not determine plot type."})
    else:
        return jsonify({"answer": response})

    if __name__ == "__main__":
        app.run(debug=True)

def format_sales_data_to_html(sales_data_string):
    """
    Formats a string of sales data into clean HTML.

    Args:
        sales_data_string (str): The sales data string to format.

    Returns:
        str: An HTML string representing the formatted sales data.
    """

    lines = sales_data_string.split('- ')  # Split the string into lines
    html_lines = []

    html_lines.append("    <h3>Sales Data Summary</h3>")
    html_lines.append("    <ul>")

    for line in lines[1:]:  # Skip the first empty line
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            html_lines.append(f"        <li><strong>{key}:</strong> {value}</li>")
        else:
            html_lines.append(f"        <li>{line}</li>")
    html_lines.append("    </ul>")

    return "\n".join(html_lines)

def plot_gender_age_product_heatmap(df):
    """Plots sales by gender, age range, and product using multiple heatmaps."""

    # Group and aggregate data
    gender_age_product_sales = df.groupby(['Customer_Gender', 'Age_Range', 'Product'])['Sales'].sum().reset_index()

    # Get unique products
    products = gender_age_product_sales['Product'].unique()

    # Create multiple heatmaps, one for each product
    heatmap_images = []
    for product in products:
        product_data = gender_age_product_sales[gender_age_product_sales['Product'] == product]
        heatmap_data = product_data.pivot_table(index='Customer_Gender', columns='Age_Range', values='Sales', fill_value=0)

        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu")
        plt.title(f'Sales by Gender and Age Range for {product}')
        plt.xlabel('Age Range')
        plt.ylabel('Gender')

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        heatmap_images.append(f'<img src="data:image/png;base64,{plot_data}" />')
        plt.close() # close plot to prevent overlaying plots.

    return "".join(heatmap_images)

