import os
import gzip
import json
import tempfile

# Set API keys
os.environ['OPENAI_API_KEY'] = OPENAI_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

# -----------------------
# BACKEND SETUP
# -----------------------

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document as LC_Document
from langgraph.graph import END, StateGraph

# Create conversation memory so that chat history is preserved.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------
# Load and Process Data
# -----------------------
wikipedia_filepath = 'simplewiki-2020-11-01.jsonl.gz'
docs = []
with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        data = json.loads(line.strip())
        # Restrict to first 3 paragraphs (for speed)
        docs.append({
            'metadata': {
                'title': data.get('title'),
                'article_id': data.get('id')
            },
            'data': ' '.join(data.get('paragraphs')[0:3])
        })

# Subset to documents that contain the keyword 'india'
docs = [doc for doc in docs if 'india' in doc['data'].lower().split()]

# Create Document objects for LangChain
docs = [Document(page_content=doc['data'], metadata=doc['metadata']) for doc in docs]

# Chunk documents
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
chunked_docs = splitter.split_documents(docs)
print("Chunked docs sample:", chunked_docs[:3])

# -----------------------
# Create the Vector DB
# -----------------------
openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')

chroma_db = Chroma.from_documents(
    documents=chunked_docs,
    collection_name='rag_wikipedia_db',
    embedding=openai_embed_model,
    # Use cosine distance for similarity search
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="./wikipedia_db"
)

similarity_threshold_retriever = chroma_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3}
)

# -----------------------
# LLM Grading Setup
# -----------------------
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

SYS_PROMPT = (
    "You are an expert grader assessing relevance of a retrieved document to a user question.\n"
    "Follow these instructions for grading:\n"
    "  - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.\n"
    "  - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant."
)
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", "Retrieved document:\n{document}\nUser question:\n{question}\n")
    ]
)
doc_grader = grade_prompt | structured_llm_grader

# -----------------------
# QA and Question Rewriting Setup
# -----------------------
# QA prompt now includes conversation history.
qa_prompt = (
    "You are an assistant for question-answering tasks.\n"
    "Chat History:\n{chat_history}\n"
    "Use the following pieces of retrieved context to answer the question.\n"
    "If no context is present or if you don't know the answer, just say that you don't know the answer.\n"
    "Do not invent details not provided in the context.\n"
    "Give a detailed and to the point answer.\n"
    "Question:\n{question}\n"
    "Context:\n{context}\n"
    "Answer:\n"
)
prompt_template = ChatPromptTemplate.from_template(qa_prompt)
chatgpt = ChatOpenAI(model_name='gpt-4o', temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_rag_chain = (
    {
        "context": itemgetter('context') | RunnableLambda(format_docs),
        "question": itemgetter('question'),
        "chat_history": itemgetter('chat_history')
    }
    | prompt_template
    | chatgpt
    | StrOutputParser()
)

# LLM for question rewriting
llm_rewrite = ChatOpenAI(model="gpt-4o", temperature=0)
SYS_PROMPT_REWRITE = (
    "Act as a question re-writer. Optimize the following question for web search "
    "by reasoning about its semantic intent.\n"
)
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT_REWRITE),
        ("human", "Here is the initial question:\n{question}\nFormulate an improved question.\n")
    ]
)
question_rewriter = re_write_prompt | llm_rewrite | StrOutputParser()

# -----------------------
# Web Search Tool Setup
# -----------------------
tv_search = TavilySearchResults(max_results=3, search_depth='advanced', max_tokens=10000)

# -----------------------
# Graph / Pipeline Definition
# -----------------------
class GraphState(TypedDict):
    question: str
    generation: str
    web_search_needed: str
    documents: List[str]

def retrieve(state):
    print("---RETRIEVAL FROM VECTOR DB---")
    question = state["question"]
    documents = similarity_threshold_retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search_needed = "No"
    if documents:
        for d in documents:
            score = doc_grader.invoke({"question": question, "document": d.page_content})
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search_needed = "Yes"
                continue
    else:
        print("---NO DOCUMENTS RETRIEVED---")
        web_search_needed = "Yes"
    return {"documents": filtered_docs, "question": question, "web_search_needed": web_search_needed}

def rewrite_query(state):
    print("---REWRITE QUERY---")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    docs = tv_search.invoke(question)
    web_results = "\n\n".join([d["content"] for d in docs])
    web_results = LC_Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}

def generate_answer(state):
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    chat_history = memory.load_memory_variables({}).get("chat_history", "")
    generation = qa_rag_chain.invoke({
        "context": documents,
        "question": question,
        "chat_history": chat_history
    })
    memory.save_context({"input": question}, {"output": generation})
    return {"documents": documents, "question": question, "generation": generation}

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    if state["web_search_needed"] == "Yes":
        print("---DECISION: DOCUMENTS NOT RELEVANT, REWRITE QUERY---")
        return "rewrite_query"
    else:
        print("---DECISION: GENERATE RESPONSE---")
        return "generate_answer"

agentic_rag = StateGraph(GraphState)
agentic_rag.add_node("retrieve", retrieve)
agentic_rag.add_node("grade_documents", grade_documents)
agentic_rag.add_node("rewrite_query", rewrite_query)
agentic_rag.add_node("web_search", web_search)
agentic_rag.add_node("generate_answer", generate_answer)
agentic_rag.set_entry_point("retrieve")
agentic_rag.add_edge("retrieve", "grade_documents")
agentic_rag.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"},
)
agentic_rag.add_edge("rewrite_query", "web_search")
agentic_rag.add_edge("web_search", "generate_answer")
agentic_rag.add_edge("generate_answer", END)
agentic_rag = agentic_rag.compile()

# -----------------------
# FRONTEND WITH STREAMLIT
# -----------------------

import streamlit as st

st.set_page_config(page_title="Agentic RAG Chat", layout="wide")
st.title("Agentic RAG Chat")
st.markdown("Interact with the Agentic RAG system below. You can upload files (including PDFs) to enrich your context.")

# File uploader widget (supports txt, jsonl, gz, pdf)
uploaded_file = st.file_uploader("Upload a file", type=["txt", "jsonl", "gz", "pdf"])
if uploaded_file is not None:
    upload_dir = "temp_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    temp_file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File '{uploaded_file.name}' uploaded and saved to '{temp_file_path}'")
    
    # Process PDF files
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == "pdf":
        # Use LangChain's PDF loader
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        pdf_docs = loader.load()
        # Split the PDF documents
        pdf_docs_chunked = splitter.split_documents(pdf_docs)
        # Add these documents to the vector DB
        chroma_db.add_documents(pdf_docs_chunked)
        st.success("PDF processed and added to the vector DB.")
    else:
        # You can add custom processing logic for other file types here.
        st.info("Uploaded file type is not specifically handled. You can add custom processing as needed.")

# Input for user question
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not question:
        st.error("Please enter a question.")
    else:
        st.info("Processing your question...")
        # Prepare initial state for the graph
        initial_state = {
            "question": question,
            "documents": [],  # Optionally, you might add documents from the uploaded file here.
            "web_search_needed": "No",
            "generation": ""
        }
        try:
            result_state = agentic_rag.run(initial_state)
            answer = result_state.get("generation", "No answer generated.")
            st.markdown("### Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
