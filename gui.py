import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QLabel, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QFont

class ChatBotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ICT Assistant')

        # Layouts
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        
        # Left side: txt file display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        with open("Manual.txt", "r") as file: 
            self.text_display.setPlainText(file.read())
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
        
        right_layout.addWidget(self.chat_history)
        right_layout.addWidget(self.user_input)
        right_layout.addWidget(send_button)
        
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        
        self.setLayout(main_layout)

    #where user input is handled a response is created
    def handle_input(self):
        user_text = self.user_input.text().strip() 
        if user_text:
            self.display_message(user_text, user=True)
            if user_text == "scroll":
                    self.scroll_to_word("This manual")
                    ai_response = "scrolled"
            else:
                ai_response = f"You said {user_text}"
            self.display_message(ai_response, user=False)
            self.user_input.clear()
            

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
        self.chat_history.setAlignment(Qt.AlignLeft)  #alignment for next message

    def scroll_to_word(self, str):
        cursor = self.text_display.textCursor()
        text = self.text_display.toPlainText()
        # Find the first occurrence of the string
        index = text.find(str)
        if index != -1:
            cursor.setPosition(index)
            self.text_display.setTextCursor(cursor)
            self.text_display.ensureCursorVisible()
            # Move the cursor to the beginning of the line
            cursor.movePosition(QTextCursor.StartOfLine)
            self.text_display.setTextCursor(cursor)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    chat_bot_gui = ChatBotGUI()
    chat_bot_gui.show()
    sys.exit(app.exec_())