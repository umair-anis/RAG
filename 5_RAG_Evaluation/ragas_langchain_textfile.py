import os
import csv
import sys
import argparse
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_classic.chains import RetrievalQA
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

load_dotenv()
# Configuration variables
MAX_TOKENS = 15000
MODEL_NAME = "gpt-5.4-mini"
TEMPERATURE = 0.4
TOP_K = 5
EMBED_BATCH_SIZE = 500

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILE = os.path.join(
    BASE_DIR, "..", "941156621-1811946734_sutter-medical-center-sacramento_standardcharges.txt"
)
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index_sutter")

# Set up OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

csv.field_size_limit(2**31 - 1)  # sys.maxsize overflows a C long on Windows

# Columns turned into labelled text, in reading order
TEXT_FIELDS = [
    ("billing_class", "Billing class"),
    ("setting", "Setting"),
    ("payer_name", "Payer"),
    ("plan_name", "Plan"),
    ("modifiers", "Modifiers"),
    ("standard_charge|gross", "Gross charge"),
    ("standard_charge|discounted_cash", "Discounted cash charge"),
    ("standard_charge|negotiated_dollar", "Negotiated dollar"),
    ("standard_charge|negotiated_percentage", "Negotiated percentage"),
    ("standard_charge|negotiated_algorithm", "Negotiated algorithm"),
    ("estimated_amount", "Estimated amount"),
    ("median_amount", "Median amount"),
    ("10th_percentile", "10th percentile"),
    ("90th_percentile", "90th percentile"),
    ("count", "Count"),
    ("standard_charge|methodology", "Methodology"),
    ("standard_charge|min", "Min charge"),
    ("standard_charge|max", "Max charge"),
    ("drug_unit_of_measurement", "Drug unit"),
    ("drug_type_of_measurement", "Drug unit type"),
    ("additional_generic_notes", "Notes"),
]
METADATA_FIELDS = ["payer_name", "plan_name", "code|1", "setting", "billing_class"]


def row_to_document(row, row_number):
    parts = [f"Service: {row['description'].strip()}"]

    codes = []
    for i in range(1, 5):
        code = (row.get(f"code|{i}") or "").strip()
        if code:
            code_type = (row.get(f"code|{i}|type") or "").strip()
            codes.append(f"{code_type} {code}".strip())
    if codes:
        parts.append("Codes: " + ", ".join(codes))

    for column, label in TEXT_FIELDS:
        value = (row.get(column) or "").strip()
        if value:
            parts.append(f"{label}: {value}")

    metadata = {"row": row_number}
    for column in METADATA_FIELDS:
        metadata[column] = (row.get(column) or "").strip()
    return Document(page_content=". ".join(parts) + ".", metadata=metadata)


def load_charges_file(path):
    """Read the CSV-formatted charges file; one Document per non-empty row."""
    documents = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        # Some hospital files have 2 metadata rows before the real header
        first_line = f.readline()
        if not first_line.lower().lstrip('"').startswith("description"):
            f.readline()
        else:
            f.seek(0)
        reader = csv.DictReader(f)
        if not reader.fieldnames or "description" not in reader.fieldnames:
            raise ValueError("Could not find a 'description' column header in the file.")

        for i, row in enumerate(reader, 1):
            if not (row.get("description") or "").strip():
                continue
            documents.append(row_to_document(row, i))

    print(f"\nNumber of documents loaded: {len(documents)}")
    if documents:
        print("Sample of loaded content:")
        print(documents[0].page_content[:300] + "...")
        print(f"Metadata: {documents[0].metadata}")
    return documents


def print_sample_embeddings(texts, embeddings):
    if texts:
        sample_text = texts[0].page_content
        sample_embedding = embeddings.embed_query(sample_text)
        print("\nSample Text:")
        print(sample_text[:200] + "..." if len(sample_text) > 200 else sample_text)
        print("\nSample Embedding (first 10 dimensions):")
        print(np.array(sample_embedding[:10]))
        print(f"\nEmbedding shape: {np.array(sample_embedding).shape}")
    else:
        print("No texts available for embedding sample.")


def build_vectorstore(documents, embeddings):
    vectorstore = None
    total = len(documents)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = documents[start:start + EMBED_BATCH_SIZE]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        print(f"Embedded {min(start + EMBED_BATCH_SIZE, total)}/{total} documents")
    return vectorstore


def get_vectorstore(path, embeddings, rebuild=False):
    if not rebuild and os.path.isdir(INDEX_DIR):
        print(f"Loading existing FAISS index from {INDEX_DIR}")
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    documents = load_charges_file(path)
    if not documents:
        return None
    print_sample_embeddings(documents, embeddings)
    print("Creating embeddings and vector store...")
    vectorstore = build_vectorstore(documents, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"FAISS index saved to {INDEX_DIR}")
    return vectorstore


# Set up OpenAI language model
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS
)

template = """You are answering questions about hospital standard charges (price transparency data).

Context:
{context}

Question: {question}

Answer concisely based only on the charges in the context. Quote the service, payer, plan and dollar amount where relevant. If the context doesn't contain the information, say "I don't have enough information to answer that question."
"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)


def rag_pipeline(query, qa_chain, vectorstore):
    relevant_docs = vectorstore.similarity_search_with_score(query, k=TOP_K)

    print(f"\nTop {TOP_K} most relevant chunks:")
    context = ""
    for i, (doc, score) in enumerate(relevant_docs, 1):
        print(f"{i}. Relevance Score: {score:.4f}")
        print(f"   Content: {doc.page_content[:200]}...")
        print()
        context += doc.page_content + "\n\n"

    # Print the full prompt
    full_prompt = PROMPT.format(context=context, question=query)
    print("\nFull Prompt sent to the model:")
    print(full_prompt)
    print("\n" + "=" * 50 + "\n")

    response = qa_chain.invoke({"query": query})
    return response['result'], relevant_docs


def evaluate_rag_performance(query, answer, relevant_docs, ground_truth=""):
    sample = SingleTurnSample(
        user_input=query,
        response=answer,
        retrieved_contexts=[doc.page_content for doc, _ in relevant_docs],
        reference=ground_truth or None,
    )
    dataset = EvaluationDataset(samples=[sample])

    # Precision and recall need a reference answer
    metrics = [faithfulness, answer_relevancy]
    if ground_truth:
        metrics += [context_precision, context_recall]

    result = evaluate(dataset=dataset, metrics=metrics)
    scores = result.to_pandas().iloc[0]

    print("\nRAGAS Evaluation Results:")
    for metric in metrics:
        print(f"{metric.name}: {scores[metric.name]:.4f}")

    # Provide suggestions based on scores
    if scores["faithfulness"] < 0.7:
        print("Suggestion: Improve answer generation to stick more closely to the retrieved context.")
    if scores["answer_relevancy"] < 0.7:
        print("Suggestion: Refine the question-answering model to generate more relevant answers.")
    if ground_truth:
        if scores["context_precision"] < 0.7:
            print("Suggestion: Improve context retrieval by fine-tuning embeddings or adjusting top-k.")
        if scores["context_recall"] < 0.7:
            print("Suggestion: Improve retrieval to cover more relevant charges (metadata filtering or hybrid BM25).")
    else:
        print("Note: context_precision and context_recall skipped (no ground truth provided).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the FAISS index from the file")
    args = parser.parse_args()
    rebuild = args.rebuild

    print("Welcome to the Hospital Charges RAG Pipeline with RAGAS Evaluation.")

    while True:
        file_path = input(f"Enter the path of the charges file (press Enter for default, or 'quit' to exit): ").strip()
        if file_path.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            break
        if not file_path:
            file_path = DEFAULT_FILE

        try:
            print("Processing file content...")
            embeddings = OpenAIEmbeddings()
            vectorstore = get_vectorstore(file_path, embeddings, rebuild)
            rebuild = False

            if vectorstore is None:
                print("No content found in the file. Please try a different file.")
                continue

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
                chain_type_kwargs={"prompt": PROMPT}
            )

            print("\nRAG Pipeline initialized. You can now enter your queries.")
            print("Enter 'new' to load a different file, 'evaluate' to run RAGAS evaluation, or 'quit' to exit the program.")

            while True:
                user_query = input("\nEnter your query: ")
                if user_query.lower() == 'quit':
                    print("Exiting the program. Goodbye!")
                    sys.exit()
                elif user_query.lower() == 'new':
                    rebuild = True
                    break
                elif user_query.lower() == 'evaluate':
                    eval_query = input("Enter a query for evaluation: ")
                    ground_truth = input("Enter the expected answer (optional, press Enter to skip): ").strip()
                    result, relevant_docs = rag_pipeline(eval_query, qa, vectorstore)
                    print(f"RAG Response: {result}")
                    evaluate_rag_performance(eval_query, result, relevant_docs, ground_truth)
                else:
                    result, _ = rag_pipeline(user_query, qa, vectorstore)
                    print(f"RAG Response: {result}")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try a different file or check your API key.")
