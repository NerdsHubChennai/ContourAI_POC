import streamlit as st
from src.github_repo_qa import GitHubRepoQA


def show_credentials_page():
    st.title("Configurations")

    owner = st.text_input("Repository Owner", "microsoft")
    repo = st.text_input("Repository Name", "autogen")
    token = st.text_input("GitHub Token", type="password")

    if st.button("Configure QA"):
        try:
            with st.spinner("Initializing QA system..."):
                st.session_state.qa = GitHubRepoQA(owner, repo, token)
            st.session_state.page = 'chat'
            st.rerun()
        except Exception as e:
            st.error(f"Failed to initialize QA system: {str(e)}")


def show_chat_page():
    st.title("Github Chatbot")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if question := st.chat_input("Ask a question about the repository"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            answer = st.session_state.qa.answer_question(question)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # End Chat button
    if st.sidebar.button("End Chat"):
        st.session_state.page = 'credentials'
        st.session_state.qa = None
        st.session_state.messages = []
        st.rerun()
