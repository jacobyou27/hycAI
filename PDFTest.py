import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import faiss
from PyPDF2 import PdfReader
import time
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data
nltk.download('punkt')

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

# Prepare document chunks using sliding window approach
def prepare_document_chunks_from_text(text, chunk_size=100, overlap=20):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-overlap:]  # Preserve overlap sentences
            current_chunk.append(sentence)
            current_length = sum(len(sent.split()) for sent in current_chunk)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to create embeddings in batches
def get_embeddings(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedding_model.encode(batch, convert_to_tensor=True, device='cuda')
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

# Load LLaMA 3 8B Instruct model for better performance
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Ensure pad_token_id is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Function to retrieve documents using semantic search
def retrieve_documents(query, document_embeddings, document_chunks, k=5):
    query_embedding = embedding_model.encode([query], convert_to_tensor=True, device='cuda')
    cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    top_results = torch.topk(cos_scores, k)

    indices = top_results[1].cpu().numpy()
    return indices, [document_chunks[idx] for idx in indices]

# Function to generate answers
def generate_answer(query, retrieved_docs, history):
    # Limit the context length to avoid issues with long sequences
    context = "\n\n".join(retrieved_docs[:3])
    history_text = "\n".join(history)
    input_text = f"""You are an AI assistant that helps with software and hardware issues.
You will be given document(s) to give you knowledge of the task. Answer the question as precisely as possible.
Use your own words. Do not hallucinate or give fake information. Do not give unnecessary information. Do not repeat yourself.
Be concise.
Here are the document(s) to help you understand: {context}

Conversation history:
{history_text}

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

# Main function
def main():
    start_time = time.time()
    
    # Choose between PDF or TXT
    file_path = r'sampleGuide.txt'
    
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
    document_embeddings = get_embeddings(document_chunks).to('cuda')

    conversation_history = []

    while True:
        # Prompt user for a question
        user_question = input("\nEnter your question (or 'exit' to quit): ").strip()
        
        if user_question.lower() == 'exit':
            break
        
        print(f"\nProcessing query: {user_question}")
        indices, retrieved_docs = retrieve_documents(user_question, document_embeddings, document_chunks)
        print("Generating answer...")
        answer = generate_answer(user_question, retrieved_docs, conversation_history)
        
        conversation_history.append(f"User: {user_question}")
        conversation_history.append(f"AI: {answer}")

        print(f"Query: {user_question}")
        print("Answer:", answer)
        print("Document Chunk:", retrieved_docs[0])
        print("\n" + "="*80 + "\n")
    
    end_time = time.time()
    print(f"Total Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
