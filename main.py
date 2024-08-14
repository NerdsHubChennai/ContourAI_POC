import streamlit as st
from src.ui import show_credentials_page, show_chat_page


def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'credentials'
    if 'qa' not in st.session_state:
        st.session_state.qa = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []


def main():
    st.set_page_config(page_title="Repo Q&A", layout="wide")

    initialize_session_state()

    with st.sidebar:
        show_credentials_page()

    if st.session_state.page == 'chat':
        show_chat_page()


if __name__ == "__main__":
    main()
