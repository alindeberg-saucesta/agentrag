import os
os.environ['OPENAI_API_KEY'] = OPENAI_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

from langchain_openai import OpenAIEmbeddings
# details here: https://openai.com/blog/new-embedding-models-and-api-updates
openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')

import gzip
import json
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load Wikipedia data
wikipedia_filepath = 'simplewiki-2020-11-01.jsonl.gz'
docs = []
with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        data = json.loads(line.strip())
        # Restrict data to first 3 paragraphs to run later modules faster
        docs.append({
            'metadata': {
                'title': data.get('title'),
                'article_id': data.get('id')
            },
            'data': ' '.join(data.get('paragraphs')[0:3])
        })

# Subset our data to use a subset of Wikipedia documents (only documents containing 'india')
docs = [doc for doc in docs if 'india' in doc['data'].lower().split()]

# Create Document objects
docs = [Document(page_content=doc['data'], metadata=doc['metadata']) for doc in docs]

# Chunk docs
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
chunked_docs = splitter.split_documents(docs)
print(chunked_docs[:3])

from langchain_chroma import Chroma
# Create vector DB of docs and embeddings
chroma_db = Chroma.from_documents(
    documents=chunked_docs,
    collection_name='rag_wikipedia_db',
    embedding=openai_embed_model,
    # Need to set the distance function to cosine else it uses Euclidean by default
    # See: https://docs.trychroma.com/guides#changing-the-distance-function
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="./wikipedia_db"
)

similarity_threshold_retriever = chroma_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3}
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Data model for LLM output format
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# LLM for grading
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt template for grading
SYS_PROMPT = (
    "You are an expert grader assessing relevance of a retrieved document to a user question.\n"
    "Follow these instructions for grading:\n"
    "  - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.\n"
    "  - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not."
)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", 
         "Retrieved document:\n{document}\nUser question:\n{question}\n")
    ]
)
# Build grader chain
doc_grader = grade_prompt | structured_llm_grader

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# Create RAG prompt for response generation
prompt = (
    "You are an assistant for question-answering tasks.\n"
    "Use the following pieces of retrieved context to answer the question.\n"
    "If no context is present or if you don't know the answer, just say that you don't know the answer.\n"
    "Do not make up the answer unless it is there in the provided context.\n"
    "Give a detailed and to the point answer with regard to the question.\n"
    "Question:\n{question}\nContext:\n{context}\nAnswer:\n"
)
prompt_template = ChatPromptTemplate.from_template(prompt)
# Initialize connection with GPT-4o
chatgpt = ChatOpenAI(model_name='gpt-4o', temperature=0)

# Used for separating context docs with new lines
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create QA RAG chain
qa_rag_chain = (
    {
        "context": itemgetter('context') | RunnableLambda(format_docs),
        "question": itemgetter('question')
    }
    | prompt_template
    | chatgpt
    | StrOutputParser()
)

# LLM for question rewriting (reuse or rename if needed)
llm_rewrite = ChatOpenAI(model="gpt-4o", temperature=0)
SYS_PROMPT_REWRITE = (
    "Act as a question re-writer and perform the following task:\n"
    " - Convert the following input question to a better version that is optimized for web search.\n"
    " - When re-writing, look at the input question and try to reason about the underlying semantic intent / meaning.\n"
)
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT_REWRITE),
        ("human", "Here is the initial question:\n{question}\nFormulate an improved question.\n")
    ]
)
# Create rephraser chain
question_rewriter = re_write_prompt | llm_rewrite | StrOutputParser()

from langchain_community.tools.tavily_search import TavilySearchResults
tv_search = TavilySearchResults(max_results=3, search_depth='advanced', max_tokens=10000)

from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM response generation
        web_search_needed: flag of whether to add web search - yes or no
        documents: list of context documents
    """
    question: str
    generation: str
    web_search_needed: str
    documents: List[str]

def retrieve(state):
    """
    Retrieve documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents - that contains retrieved context documents
    """
    print("---RETRIEVAL FROM VECTOR DB---")
    question = state["question"]
    # Retrieval
    documents = similarity_threshold_retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    by using an LLM Grader.
    If any document is not relevant to question or documents are empty - Web Search needs to be done.
    If all documents are relevant to question - Web Search is not needed.
    Helps filter out irrelevant documents.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
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
    """
    Rewrite the query to produce a better question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates question key with a re-phrased or re-written question
    """
    print("---REWRITE QUERY---")
    question = state["question"]
    documents = state["documents"]
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

from langchain.schema import Document as LC_Document
def web_search(state):
    """
    Web search based on the re-written question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with appended web results
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    # Web search
    docs = tv_search.invoke(question)
    web_results = "\n\n".join([d["content"] for d in docs])
    web_results = LC_Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}

def generate_answer(state):
    """
    Generate answer from context documents using LLM.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    generation = qa_rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    Args:
        state (dict): The current graph state
    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    web_search_needed = state["web_search_needed"]
    if web_search_needed == "Yes":
        # Some documents are not relevant: rewrite query
        print("---DECISION: SOME or ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, REWRITE QUERY---")
        return "rewrite_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE RESPONSE---")
        return "generate_answer"

from langgraph.graph import END, StateGraph

agentic_rag = StateGraph(GraphState)
# Define the nodes
agentic_rag.add_node("retrieve", retrieve)   # retrieve
agentic_rag.add_node("grade_documents", grade_documents)   # grade documents
agentic_rag.add_node("rewrite_query", rewrite_query)   # transform query
agentic_rag.add_node("web_search", web_search)   # web search
agentic_rag.add_node("generate_answer", generate_answer)   # generate answer
# Build graph
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
# Compile
agentic_rag = agentic_rag.compile()
