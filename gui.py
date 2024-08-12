import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QLabel, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QFont, QIcon
import speech_recognition as sr
from gtts import gTTS
import os
from playsound import playsound
import random
import nltk
from nltk.tokenize import sent_tokenize


class ICTAssistant(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.recognizer = sr.Recognizer()
        self.listening = False

    def initUI(self):
        self.setWindowTitle('ICT Assistant')

        #layouts
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        
        #left side: txt file display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        with open("manual.txt", "r", encoding="utf8") as file: 
            self.text_display.setPlainText(file.read())
        left_layout.addWidget(self.text_display)

        # right side: chat history and input
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(QFont("Arial", 12))
        self.chat_history.setContextMenuPolicy(Qt.CustomContextMenu)
        self.chat_history.customContextMenuRequested.connect(self.read_aloud)

        #sending messages
        self.input_line = QLineEdit()
        self.input_line.setFont(QFont("Arial", 12))
        self.input_line.returnPressed.connect(self.process_message) 

        self.send_button = QPushButton('Send')
        self.send_button.clicked.connect(self.process_message) 

        #microphone button
        self.mic_button = QPushButton()
        self.mic_button.setIcon(QIcon('microphone.png'))
        self.mic_button.setStyleSheet("background-color: red")
        self.mic_button.clicked.connect(self.toggle_listening)


        right_layout.addWidget(QLabel("Chat History"))
        right_layout.addWidget(self.chat_history)
        right_layout.addWidget(QLabel("Your Input"))
        right_layout.addWidget(self.input_line)
        right_layout.addWidget(self.send_button)
        right_layout.addWidget(self.mic_button)

        #layouts
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

    #listening on microphone
    def toggle_listening(self):
        if not self.listening:
            self.listening = True
            self.mic_button.setText("Stop Listening")
            self.listen()
        else:
            self.listening = False
            self.mic_button.setText("Start Listening")

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)

        try:
            print("Recognizing...")
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized text: {text}")
            self.input_line.setText(text)
            self.process_message()
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
        except sr.RequestError:
            print("Could not request results from the speech recognition service.")

        self.toggle_listening()

    #messages are processed here
    def process_message(self):
        message = self.input_line.text()

        if message:
            self.chat_history.append(f"You: {message}")
            self.chat_history.setAlignment(Qt.AlignLeft)
            self.input_line.clear()

            #bot response created here

            self.chat_history.setAlignment(Qt.AlignRight)
            if message == "scroll":
                self.scroll_to_word("Due to the large field-of-view lens used in VMQ")
                response = "Scrolled"
            elif message == "quit" or message == "exit":
                sys.exit(app.exec_())
            else:
                response = "You said " + message

            self.chat_history.append(f"Bot: {response}")
            self.chat_history.setAlignment(Qt.AlignLeft)
            self.chat_history.append("")



    #right click on message to play message
    def read_aloud(self, position):
        cursor = self.chat_history.cursorForPosition(position)
        cursor.select(QTextCursor.BlockUnderCursor)
        text = cursor.selectedText()
        tts = gTTS(text=text, lang='en')
        tts.save("temp.mp3")
        playsound("temp.mp3")
        os.remove("temp.mp3")

    def scroll_to_word(self, str):
        cursor = self.text_display.textCursor()
        text = self.text_display.toPlainText()

        # find string
        index = text.find(str)
        if index != -1:
            cursor.setPosition(index)
            self.text_display.setTextCursor(cursor)

            cursor.movePosition(QTextCursor.StartOfLine)
            self.text_display.setTextCursor(cursor)
            
            block_number = cursor.blockNumber()
            scrollbar = self.text_display.verticalScrollBar()
            cursor.movePosition(QTextCursor.StartOfBlock)
            block_top = self.text_display.cursorRect(cursor).top()
            scrollbar.setValue(scrollbar.value() + block_top - self.text_display.viewport().rect().top())
            self.text_display.ensureCursorVisible()


    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ICTAssistant()
    gui.show()
    sys.exit(app.exec_())


# Download NLTK data
nltk.download('punkt')

# Function to extract text from a TXT file
def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to prepare document chunks
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

# Function to get a random document chunk
def get_random_chunk(document_chunks):
    if document_chunks:
        return random.choice(document_chunks)
    else:
        return "No document chunks available."

# Main function to load the manual, chunk it, and get a random chunk
def main():
    # Load the text from manual.txt
    file_path = 'manual.txt'
    text = extract_text_from_txt(file_path)
    
    # Prepare the document chunks
    document_chunks = prepare_document_chunks_from_text(text, chunk_size=100, overlap=20)
    
    # Get a random chunk
    random_chunk = get_random_chunk(document_chunks)
    print("Document Chunk:\n", random_chunk)

if __name__ == "__main__":
    main()
