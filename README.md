# 2023 Florida Residential Building Codes Assistant

I developed this project to learn about LangChain and how it can be used for RAG pipelines.
The goal of this applet is to provide a RAG-based assistant for answering questions about building codes.

---
## Architecture
Here is the following stack I used:
- Python
 - LangChain
 - OpenAI
 - Flask
 - Docling
- HTML/CSS
---
## How to run:
To just test the program, you will need:
- An OpenAI API Key
- A Flask Secret Key
These are to be stored in a .env, but *ONLY FOR TESTING* (obviously not for production environments)!
---
## How to use:
1. Ask a question (related to building codes)
2. Receive a response! :D
3. The model will punt or at least provide the closest answer it can if the info cannot be found in the docs.
---
## How it works:
1. The `doc_parser.py` was used to convert the PDFs to MD files __(not present for copyright reasons)__.
2. The main app loads the environment variables.
3. The program reads the doc files.
4. The files are split into overlapping chunks based on headers.
5. Metadata is produced from each chunk.
6. The data is converted into embeddings via OpenAI Embeddings Model.
7. A vector store of all the data is created.
8. The user types a question and submits it via a POST action.
9. A similarity search of the top K (default 5) chunks is done based on the query.
10. After reading the docs (Retrieval) GPT 5.5 crafts (Augmented) an answer (Generation) based on context, system prompts, and the query.
11. The answer is returned and displayed!
---
## Things to do:
I have several ideas:
- Expand the codes to the main body of codes (which are referenced frequently)
- Add chat history (and more context)
- Expand system instructions for more helpful responses
