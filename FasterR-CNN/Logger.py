import sys
import re

# DualLogger class to write to both the console and the file, to save console log from training for later
class Logger:
    def __init__(self, file_path):
        self.console = sys.__stdout__  # console output
        # log file, utf-8 to correctly log fancy_grid in txt file
        self.file = open(file_path, 'w', encoding="utf-8")

    # method uses python regular expression to get rid on ansi formatting
    # used to keep fancy tables with colours while displaying correctly in txt file
    def remove_ansi(self, text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)


    def write(self, message):
        self.console.write(message)  # Write to console
        self.file.write(self.remove_ansi(message))  # Write to log file

    def flush(self):
        # Ensures output is flushed to both console and file
        self.console.flush()
        self.file.flush()