import sys

# DualLogger class to write to both the console and the file, to save console log from training for later
class Logger:
    def __init__(self, file_path):
        self.console = sys.__stdout__  # console output
        # log file, utf-8 to correctly log fancy_grid in txt file
        self.file = open(file_path, 'w', encoding="utf-8")

    def write(self, message):
        self.console.write(message)  # Write to console
        self.file.write(message)  # Write to log file

    def flush(self):
        # Ensures output is flushed to both console and file
        self.console.flush()
        self.file.flush()