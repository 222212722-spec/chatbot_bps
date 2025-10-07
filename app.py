import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ui.chat_ui import render_chat_ui

def main():
    render_chat_ui()

if __name__ == "__main__":
    main()