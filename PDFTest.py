import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import time

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Extract text from a TXT file
def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Prepare document chunks from the dataset
def prepare_document_chunks_from_text(text, chunk_size=100):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to create embeddings
def get_embeddings(texts):
    return embedding_model.encode(texts, convert_to_tensor=True)

# Load LLaMA 8B model for better performance
model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Ensure pad_token_id is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Function to retrieve documents
def retrieve_documents(query, index, document_chunks, k=5):
    query_embedding = get_embeddings([query]).cpu().numpy()
    distances, indices = index.search(query_embedding, k)
    return indices[0], [document_chunks[idx] for idx in indices[0]]

# Function to generate answers
def generate_answer(query, retrieved_docs):
    # Limit the context length to avoid issues with long sequences
    context = "\n\n".join(retrieved_docs[:3])  # Separating chunks more clearly
    input_text = f"""You are an AI assistant that helps with software and hardware issues. 
You will be given document(s) to give you knowledge of the task. Answer the question as precisely as possible.
Use your own words. Do not hallucinate or give fake information. Do not give unnecessary information. Do not repeat yourself.
Be concise.
Here are the document(s) to help you understand: {context}
\n\nQuestion: {query}\n\nAnswer:"""
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=200,  # Adjust the number of tokens as necessary
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    response = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

# Function to find and highlight the exact location of the answer
def find_answer_location(answer, retrieved_docs):
    answer_embedding = get_embeddings([answer]).cpu().numpy()
    max_similarity = 0
    best_match = None
    best_match_idx = -1

    for idx, doc in enumerate(retrieved_docs):
        doc_embedding = get_embeddings([doc]).cpu().numpy()
        similarity = cosine_similarity(answer_embedding, doc_embedding).flatten()[0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = doc
            best_match_idx = idx

    return best_match_idx, best_match

# Main function
def main():
    start_time = time.time()
    
    # Choose between PDF or TXT
    file_path = r'Manual.txt'
    
    if file_path.lower().endswith('.pdf'):
        print(f"Extracting text from PDF: {file_path}")
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        print(f"Extracting text from TXT: {file_path}")
        text = extract_text_from_txt(file_path)
    else:
        print("Unsupported file format. Please provide a PDF or TXT file.")
        return
    
    document_chunks = prepare_document_chunks_from_text(text)
    
    print("Generating embeddings for document chunks...")
    document_embeddings = get_embeddings(document_chunks).cpu().numpy()
    
    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(document_embeddings)

    queries = [
        "What should I do if the instrument fails to power on?",
        "What steps should be taken if the measurement results are inconsistent?",
        "How can I perform a software update on the TZTEK VM series instrument?",
        "What should be done if the instrument's software crashes during operation?",
        "How do I clean and maintain the lenses of the TZTEK VM series instrument?",
        "What steps should be taken if the instrument is not responding to commands?",
        "How do I reset the TZTEK VM series instrument to its factory settings?"
    ]

    for query in queries:
        print(f"\nProcessing query: {query}")
        indices, retrieved_docs = retrieve_documents(query, index, document_chunks)
        print("Generating answer...")
        answer = generate_answer(query, retrieved_docs)
        #location_idx, location_doc = find_answer_location(answer, retrieved_docs)

        print(f"Query: {query}")
        print("Answer:", answer)
        #print("Location Index:", indices[location_idx])
        #print("Document with Answer:", location_doc)
        print("\n" + "="*80 + "\n")
    
    end_time = time.time()
    print(f"Total Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
