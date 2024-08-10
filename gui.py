import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QLabel, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QFont, QIcon
import speech_recognition as sr
from gtts import gTTS
import os
from playsound import playsound


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
        with open("Manual.txt", "r") as file: 
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
            if message == "test scroll":
                self.scroll_to_word("This manual")
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

    #scrolls to where the string shows up first
    def scroll_to_word(self, str):
        cursor = self.text_display.textCursor()
        text = self.text_display.toPlainText()
        index = text.find(str)
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
