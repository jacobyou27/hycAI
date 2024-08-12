import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QLabel)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QIcon
import speech_recognition as sr
from gtts import gTTS
import os
from playsound import playsound

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
        batch_embeddings = embedding_model.encode(batch, convert_to_tensor=True)
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
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).to(document_embeddings.device)
    cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    top_results = torch.topk(cos_scores, k)

    indices = top_results[1].cpu().numpy()
    return indices, [document_chunks[idx] for idx in indices]

# Function to generate answers
def generate_answer(query, retrieved_docs, history):
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
        max_new_tokens=200,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    response = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

class ICTAssistant(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_document()
        self.document_embeddings = get_embeddings(self.document_chunks).to(model.device)
        self.conversation_history = []

    def initUI(self):
        self.setWindowTitle('ICT Assistant')

        # Layouts
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Left side: txt file display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        left_layout.addWidget(self.text_display)

        # Right side: chat history and input
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message here...")
        self.user_input.returnPressed.connect(self.handle_input)

        send_button = QPushButton("Send")
        send_button.clicked.connect(self.handle_input)

        # Label to show the chatbot is thinking
        self.thinking_label = QLabel("")
        self.thinking_label.setAlignment(Qt.AlignCenter)

        right_layout.addWidget(self.chat_history)
        right_layout.addWidget(self.user_input)
        right_layout.addWidget(send_button)
        right_layout.addWidget(self.thinking_label)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

    def load_document(self):
        file_path = r'manual.txt'
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.txt'):
            text = extract_text_from_txt(file_path)
        else:
            self.text_display.setPlainText("Unsupported file format. Please provide a PDF or TXT file.")
            return
        self.text_display.setPlainText(text)
        self.document_chunks = prepare_document_chunks_from_text(text)

    def handle_input(self):
        user_text = self.user_input.text().strip()
        if user_text:
            self.display_message(user_text, user=True)
            self.user_input.clear()
            self.thinking_label.setText("Thinking...")
            QApplication.processEvents()
            
            indices, retrieved_docs = retrieve_documents(user_text, self.document_embeddings, self.document_chunks)
            
            if retrieved_docs:
                ai_response = generate_answer(user_text, retrieved_docs, self.conversation_history)
                # Scroll to the relevant part of the document
                self.scroll_to_text(retrieved_docs[0])
            else:
                ai_response = generate_answer(user_text, [], self.conversation_history)
                ai_response += " (Note: This response is generated and not based on the document.)"

            self.display_message(ai_response, user=False)
            self.conversation_history.append(f"User: {user_text}")
            self.conversation_history.append(f"AI: {ai_response}")
            self.thinking_label.setText("")

    def display_message(self, message, user=True):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.End)

        if user:
            alignment = Qt.AlignLeft
            prefix = "User: "
            format = QTextCharFormat()
        else:
            alignment = Qt.AlignRight
            prefix = "AI: "
            format = QTextCharFormat()
            format.setFontItalic(True)

        self.chat_history.setAlignment(alignment)
        self.chat_history.setCurrentCharFormat(format)
        self.chat_history.append(f"{prefix}{message}")
        self.chat_history.append("")
        self.chat_history.setAlignment(Qt.AlignLeft)  # Reset alignment for next message

    def scroll_to_text(self, text):
        cursor = self.text_display.textCursor()
        doc_text = self.text_display.toPlainText()
        index = doc_text.find(text)
        if index != -1:
            cursor.setPosition(index)
            self.text_display.setTextCursor(cursor)
            self.text_display.ensureCursorVisible()
            cursor.movePosition(QTextCursor.StartOfLine)
            self.text_display.setTextCursor(cursor)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ICTAssistant()
    gui.show()
    sys.exit(app.exec_())
