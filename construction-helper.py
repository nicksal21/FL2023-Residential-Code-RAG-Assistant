from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import getpass
import os
from os import listdir
from os.path import isfile, join
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from flask import Flask, request, render_template, redirect, url_for, session

def load_environment_variables():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key

def create_vector_store():
    md_directory = Path("./florida_residential_code_mds")
    # Source - https://stackoverflow.com/a/3207973
    # Posted by pycruft, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-06-11, License - CC BY-SA 4.0
    # Load file paths from the markdown directory and sort
    file_paths = [f for f in listdir(md_directory) if isfile(join(md_directory, f))]
    file_paths.sort()

    # Open all markdown files and read their content into a list
    files = {}
    for file_path in file_paths:
        with open(join(md_directory, file_path), "r", encoding="utf-8") as file:
            content = file.read()
            files[file_path] = content

    # Define headers to split on for the MarkdownHeaderTextSplitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # Split the content of each file based on the defined headers and store the splits in a list
    docs = []
    for file_path, content in files.items():
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(content)
        for doc in md_header_splits:
            doc.metadata["source"] = file_path

        docs.extend(md_header_splits)

    # Create an instance of the OpenAIEmbeddings class with the specified model
    embed = OpenAIEmbeddings(model="text-embedding-3-large")

    # Create and populate the in-memory vector store with the documents and their embeddings
    vector_store = InMemoryVectorStore(embedding=embed)
    vector_store.add_documents(docs)
    
    return vector_store

def ask_question(vector_store, client, question, k=5):
    # Get the top 5 most similar documents from the vector store based on the query
    results = vector_store.similarity_search(question, k=k) 
    
    # Combine the metadata and content of the retrieved documents into a single context string
    context = "\n\n---\n\n".join(
        f"Source: {result.metadata.get('source')}\n"
        f"Headers: {result.metadata}\n\n"
        f"Content: {result.page_content}"
        for result in results
    )


    response = client.responses.create(
        model="gpt-5.5",
        instructions=(
            "You answer questions using only the provided Florida Residential Code context. "
            "If the context does not contain the answer, say you do not know. "
            "Cite the source filename and relevant section/header when possible."
        ),
        input=f"""
    Question:
    {question}
        
    Context:
    {context}
        """
    )

    return response.output_text

def main():
    # Load environment variables, including the OpenAI API key
    load_environment_variables()

    # Create an instance of the OpenAI client
    client = OpenAI()
    
    # Create the vector store from the markdown files
    vector_store = create_vector_store()

    k_value = 5  # Number of similar documents to retrieve
    
    @app.route("/")
    def home():
        answer = session.pop("answer", None)  # Retrieve the answer from the session, if available
        return render_template("index.html", answer=answer)
    
    @app.route("/submit", methods=["POST"])
    def handle_data():
        user_input = request.form["user_input"]
        answer = ask_question(vector_store, client, user_input, k=k_value)
        session["answer"] = answer  # Store the answer in the session
        return redirect(url_for("home"))

if __name__ == "__main__":
    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_KEY")
    main()
    app.run(debug=True)