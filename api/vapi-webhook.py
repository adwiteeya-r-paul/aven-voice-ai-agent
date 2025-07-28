# Import necessary libraries
from flask import Flask, request, jsonify
import json
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, WebBaseLoader, YoutubeLoader, DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from pinecone import ServerlessSpec
from pinecone import Pinecone
from exa_py import Exa
from groq import Groq
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

EXA_API_KEY = os.getenv('EXA_API')
PINECONE_API_KEY = os.getenv('PINECONE_API')
GROQ_API_KEY = os.getenv('GROQ_API')
VAPI_WEBHOOK_SECRET = os.getenv("VAPI_WEBHOOK_SECRET")

# Initialize clients (ensure these are correctly populated from environment variables)
# IMPORTANT: Ensure sentence-transformers is in your requirements.txt
try:
    exa = Exa(EXA_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("All API clients initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize one or more API clients: {e}", exc_info=True)
    # If clients fail to initialize, the app might not work, but we proceed to allow Flask to start.

index_name = "aven-agent"
dimension = 384
namespace = "aven documents"
index = None
vectorstore = None

# Initializing Pinecone
def pineconeinit():
    global index, vectorstore
    logger.info("Attempting Pinecone initialization...")
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY is not set. Pinecone initialization skipped.")
        return

    try:
        if index_name not in pc.list_indexes().names():
            logger.info(f"Creating Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec= ServerlessSpec(cloud='aws', region='us-east-1'),
            )
        index = pc.Index(index_name)
        # Ensure the vectorstore is correctly initialized with the index
        vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
        logger.info(f"Pinecone index '{index_name}' and vectorstore initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}", exc_info=True)

# Run Pinecone initialization within the Flask app context
with app.app_context():
    pineconeinit()

# RAG function
def ragquery(query: str) -> str:
    logger.info(f"Starting RAG query for: '{query}'")
    if index is None or vectorstore is None:
        logger.error("Pinecone index or vectorstore not initialized. Cannot perform RAG query.")
        return "I'm sorry, my knowledge base isn't fully set up right now. Please try again later."

    try:
        query_embeddings = embeddings.embed_query(text=query)
        logger.info("Query embedded successfully.")

        top_matches = index.query(vector=query_embeddings, top_k=5, include_metadata=True, namespace=namespace)
        
        contexts = []
        if top_matches and top_matches['matches']:
            contexts = [item['metadata']['text'] for item in top_matches['matches'] if 'text' in item['metadata']]
            logger.info(f"Retrieved {len(contexts)} contexts from Pinecone.")
            for i, context in enumerate(contexts):
                logger.debug(f"Context {i+1}: {context[:100]}...")
        else:
            logger.warning("No relevant contexts found in Pinecone. Attempting web search with Exa.")
            # Fallback to Exa if no Pinecone matches
            if EXA_API_KEY: # Only try Exa if API key is set
                exa_results = exa.search_and_contents(query, text=True, num_results=3)
                if exa_results and exa_results.results:
                    contexts = [res.text for res in exa_results.results if res.text]
                    logger.info(f"Retrieved {len(contexts)} contexts from Exa.")
                else:
                    logger.warning("No results found from Exa search either.")
            else:
                logger.warning("EXA_API_KEY is not set. Skipping Exa search.")

        if not contexts:
            logger.info("No context available, returning a generic response.")
            return "I'm sorry, I couldn't find specific information related to that in my knowledge base or through a quick search. Is there anything else I can help you with?"

        augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
        logger.info("Augmented query prepared for Groq.")

        system_prompt = f"""Your name is AvenVoix. Everytime user asks you something, you will greet the user and say thanks. And then you will respon with the information you have. Keep it casual. Ask if you can help them with anything else. Don't say anything about the context to the user. If you have related information, share that."
        """
        logger.info("Sending request to Groq...")

        llm_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )
        response_content = llm_response.choices[0].message.content
        logger.info(f"Groq response received: {response_content[:200]}...")
        return response_content

    except Exception as e:
        logger.error(f"Error during RAG query: {e}", exc_info=True)
        return "I'm experiencing a technical issue and can't answer that right now. Please try again later."


@app.route("/api/python")
def hello_world_test():
    logger.info("Hello World test endpoint hit.")
    return "<p>Hello, World! RAG backend is ready for action.</p>"

@app.route("/api/knowledgebase", methods=["POST"])
def knowledgebase():
    logger.info("Knowledge base update endpoint hit.")
    if not EXA_API_KEY:
        logger.error("EXA_API_KEY is not set. Cannot update knowledge base.")
        return jsonify({"status": "error", "message": "EXA_API_KEY not set. Cannot update knowledge base."}), 500
    if index is None or vectorstore is None:
        logger.error("Pinecone index or vectorstore not initialized. Cannot update knowledge base.")
        return jsonify({"status": "error", "message": "Pinecone not ready. Cannot update knowledge base."}), 500

    try:
        results = exa.search_and_contents(
            "Aven Support Articles",
            text=True,
            num_results=10
        )
        text_content = ""
        if results and results.results:
            for line in results.results:
                if line.text:
                    text_content += line.text + "\n"
            logger.info(f"Scraped {len(results.results)} articles for knowledge base.")
        else:
            logger.warning("Exa search returned no results for 'Aven Support Articles'.")
            return jsonify({"status": "warning", "message": "No articles found to update knowledge base."}), 200

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        documents = text_splitter.create_documents([text_content])
        logger.info(f"Split scraped data into {len(documents)} documents.")
        
        if not documents:
            logger.warning("No documents created after splitting text content.")
            return jsonify({"status": "warning", "message": "No documents created from scraped data."}), 200

        batch_size = 100
        for i in range(0, len(documents), batch_size):
            i_end = min(i + batch_size, len(documents))
            batch = documents[i:i_end]
            metadatas = [{"text": doc.page_content} for doc in batch]
            texts = [doc.page_content for doc in batch]
            
            vectorstore.add_texts(texts, metadatas=metadatas, namespace=namespace)
            logger.info(f"Inserted batch {i//batch_size + 1} into Pinecone.")

        logger.info("Knowledge base update process completed.")
        return jsonify({"status": "success", "message": "Knowledge base update process initiated."}), 200

    except Exception as e:
        logger.error(f"Error in knowledgebase update: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "An internal error occurred during knowledge base update."}), 500


@app.route('/api/query-aven', methods=['POST'])
def query_aven():
    logger.info("Query Aven endpoint hit.")
    try:
        data = request.json
        query = data.get('query')

        if not query:
            logger.warning("No query provided in request body for /api/query-aven.")
            return jsonify({"error": "No query provided in request body."}), 400

        answer = ragquery(query)
        logger.info(f"Returning answer for query '{query[:50]}...': {answer[:100]}...")
        return jsonify({"query": query, "answer": answer}), 200

    except Exception as e:
        logger.error(f"Error in /api/query-aven: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "An internal error occurred."}), 500


@app.route('/api/vapi-webhook', methods=['POST'])
def vapi_webhook():
    logger.info("Vapi webhook endpoint hit.")
    vapi_signature = request.headers.get('X-Vapi-Signature')
    
    if not VAPI_WEBHOOK_SECRET:
        logger.error("VAPI_WEBHOOK_SECRET is not set in environment variables. Webhook cannot be verified.")
        return jsonify({"status": "error", "message": "Server configuration error: VAPI_WEBHOOK_SECRET missing."}), 500

    if vapi_signature != VAPI_WEBHOOK_SECRET:
        logger.warning(f"Unauthorized Vapi webhook request: Signature mismatch. Received: {vapi_signature}, Expected: {VAPI_WEBHOOK_SECRET}")
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        data = request.json
        event_type = data.get('type')
        logger.info(f"Received Vapi event type: {event_type}")

        if event_type == 'function_call':
            function_call_data = data.get('functionCall', {})
            function_name = function_call_data.get('name')
            parameters = function_call_data.get('parameters')
            tool_call_id = function_call_data.get('id')

            logger.info(f"Function call: {function_name} with parameters: {parameters}")

            if function_name == 'query_aven_knowledge_base':
                user_query_from_vapi = parameters.get('query')

                if not user_query_from_vapi:
                    logger.error("Vapi function_call for 'query_aven_knowledge_base' missing 'query' parameter.")
                    return jsonify({
                        "result": {
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": "I'm sorry, I didn't receive a clear query for the knowledge base. Can you please rephrase?",
                                    "type": "speech"
                                }
                            ]
                        }
                    }), 400

                logger.info(f"Vapi triggered RAG query with: '{user_query_from_vapi}'")
                rag_answer = ragquery(user_query_from_vapi)
                logger.info(f"RAG answer for Vapi: {rag_answer[:200]}...")

                return jsonify({
                    "result": {
                        "messages": [
                            {
                                "role": "assistant",
                                "content": rag_answer,
                                "type": "speech"
                            }
                        ]
                    }
                }), 200

            else:
                logger.warning(f"Unknown function call received: {function_name}")
                return jsonify({
                    "result": {
                        "messages": [
                            {
                                "role": "assistant",
                                "content": "I'm sorry, I don't know how to handle that request.",
                                "type": "speech"
                            }
                        ]
                    }
                }), 400

        elif event_type == 'end':
            logger.info("Conversation ended by Vapi.")
            return jsonify({"status": "success", "message": "Conversation ended."}), 200
        elif event_type == 'hang':
            logger.info("Call hung up by Vapi.")
            return jsonify({"status": "success", "message": "Call hung up."}), 200
        elif event_type == 'speech_start' or event_type == 'speech_end' or event_type == 'transcript' or event_type == 'conversation_update':
            logger.debug(f"Acknowledging Vapi event type: {event_type}")
            return jsonify({"status": "success", "message": "Acknowledged"}), 200
        else:
            logger.info(f"Unhandled Vapi event type: {event_type}")
            return jsonify({"status": "success", "message": "Unhandled event type acknowledged"}), 200

    except Exception as e:
        logger.error(f"Critical error in Vapi webhook: {e}", exc_info=True)
        return jsonify({
            "result": {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I'm sorry, I encountered an unexpected error. Please try again.",
                        "type": "speech"
                    }
                ]
            }
        }), 500

