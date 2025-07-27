#import libraries
from flask import Flask
import json
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, WebBaseLoader, YoutubeLoader, DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from exa_py import Exa
from groq import Groq
import os


# initialise flask
app = Flask(__name__)

# api keys and constants
EXA_API_KEY = os.getenv('EXA_API')
PINECONE_API_KEY = os.getenv('PINECONE_API')
GROQ_API_KEY = os.getenv('GROQ_API')
exa = Exa(EXA_API_KEY)
VAPI_WEBHOOK_SECRET = os.getenv("VAPI_WEBHOOK_SECRET")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.getenv('PINECONE_API'))
groq_client = Groq(api_key=os.getenv('GROQ_API'))
index_name = "aven-agent"
dimension = 384
namespace = "aven documents"
index = None
vectorstore = None

# initializing Pinecone
def pineconeinit():
    global index, vectorstore
    if index_name not in pc.list_indexes().names():
      pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec= ServerlessSpec(cloud='aws', region='us-east-1'),
            )
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=os.getenv('PINECONE_API'))


with app.app_context(): 
    initialize_pinecone()

#ragfunction
def ragquery(query : str) -> str:
    query_embeddings = embeddings.embed_query(text = query)
    top_matches = index.query(vector=query_embeddings, top_k=1, include_metadata=True, namespace=namespace)
    contexts = [item['metadata']['text'] for item in top_matches['matches']]



    # Free Llama 3.1 API via Groq
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
    system_prompt = f"""Your name is AvenVoix. Everytime user asks you something, you will greet the user and say thanks. And then you will respon with the information you have. Keep it casual. Ask if you can help them with anything else. Don't say anything about the context to the user. If you have related information, share that."
    """
    llm_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )
    response = llm_response.choices[0].message.content
    print(response)
    return response


# flask routes
@app.route("/api/python")
def hello_world_test():
    """
    A simple test endpoint to confirm the Flask app is running.
    """
    return "<p>Hello, World! RAG backend is ready for action.</p>"

@app.route("/api/knowledgebase", methods = ["POST"])
def knowledgebase():
    # scraping data
    results = exa.search_and_contents(
    "Aven Support Articles",
    text = True,
    )
    text_content = ""
    for line in results.results:
      text_content += line.text + "\n"



    # processing scraped data 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 50,
        chunk_overlap  = 10,
        separators = ["\n-","\n\n"]
    )
    documents = text_splitter.create_documents([text_content])



    # embedding using HuggingFace Embeddings client
    embed = []
    for i,n in enumerate(documents):
      embed.append(embeddings.embed_query(documents[i].page_content))


    # inserting data into pinecone
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        i_end = min(i+batch_size, len(documents))
        batch = documents[i:i_end]
        metadatas = [{"text": doc.page_content} for doc in batch]
        texts = [doc.page_content for doc in batch]
        docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=os.getenv('PINECONE_API'))
        docsearch.add_texts(texts, metadatas=metadatas, namespace=namespace)

    return jsonify({"status": "success", "message": "Knowledge base update process initiated."})


@app.route('/api/query-aven', methods=['POST'])
def query_aven_api():
    try:
        data = request.json # Get JSON data from the request body
        query = data.get('query') # Extract the 'query' field

        if not query:
            return jsonify({"error": "No query provided in request body."}), 400

        answer = _perform_rag_query(query)
        return jsonify({"query": query, "answer": answer}), 200

    except RuntimeError as re: # Catch specific errors from the helper function
        return jsonify({"status": "error", "message": str(re)}), 500
    except Exception as e:
        print(f"Error in /api/query-aven: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred."}), 500



@app.route('/api/vapi-webhook', methods=['POST'])
def vapi_webhook():
    vapi_signature = request.headers.get('X-Vapi-Signature')
    if not VAPI_WEBHOOK_SECRET:
        print("WARNING: VAPI_WEBHOOK_SECRET not set in environment variables.")
        return jsonify({"status": "error", "message": "Server configuration error"}), 500

    if vapi_signature != VAPI_WEBHOOK_SECRET:
        print(f"Unauthorized Vapi webhook request: Signature mismatch. Received: {vapi_signature}, Expected: {VAPI_WEBHOOK_SECRET}")
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        data = request.json
        event_type = data.get('type')
        print(f"Received Vapi event type: {event_type}")

        if event_type == 'function_call':
            function_call_data = data.get('functionCall', {})
            function_name = function_call_data.get('name')
            parameters = function_call_data.get('parameters')
            tool_call_id = function_call_data.get('id') # Used to tell Vapi which tool call this result belongs to.

            if function_name == 'query_aven_knowledge_base':
                user_query_from_vapi = parameters.get('query')

                if not user_query_from_vapi:
                    print("Error: Vapi function_call for 'query_aven_knowledge_base' missing 'query' parameter.")
                    return jsonify({
                        "status": "error",
                        "toolCallResults": [{
                            "toolCallId": tool_call_id,
                            "output": {"error": "Missing 'query' parameter for knowledge base tool."}
                        }]
                    }), 400

                print(f"Vapi triggered RAG query with: '{user_query_from_vapi}'")
                # CORRECTED: Function name from _perform_rag_query to ragquery
                rag_answer = ragquery(user_query_from_vapi)

                # ADDED: This return statement is crucial for Vapi to receive the tool output
                return jsonify({
                    "status": "success",
                    "toolCallResults": [{
                        "toolCallId": tool_call_id,
                        "output": {"answer": rag_answer}
                    }]
                }), 200

            else:
                print(f"Unknown function call received: {function_name}")
                return jsonify({"status": "error", "message": "Unknown function call"}), 400

        # Handle other Vapi event types if necessary (e.g., 'call.end', 'speech.update')
        # For a simple setup, you might just return a success for unhandled types
        return jsonify({"status": "success"}), 200

    except Exception as e:
        print(f"Error in Vapi webhook: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred in Vapi webhook."}), 500



