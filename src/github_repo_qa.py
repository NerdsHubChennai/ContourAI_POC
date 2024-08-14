from langchain_huggingface import HuggingFaceEmbeddings
import os
import json
import requests
from typing import Dict, Any, List
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms.google_palm import GooglePalm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter


class GitHubRepoQA:
    def __init__(self, owner: str, repo: str, token: str):
        self.owner = owner
        self.repo = repo
        self.token = token
        self.llm = GooglePalm(google_api_key='AIzaSyDuCRB7QjFL-jckgBbAkBTXoRocPkbZO-M', temperature=0.2)
        self.qa_chain = self.setup_qa_chain()
        self.data = self.fetch_github_data()
        self.retriever = self.setup_retriever()

    def fetch_github_data(self) -> Dict[str, Any]:
        """Fetch all data from GitHub API, handling pagination."""
        headers = {'Authorization': f'token {self.token}'}
        base_url = f'https://api.github.com/repos/{self.owner}/{self.repo}'

        # Fetch repository information
        repo_info = requests.get(base_url, headers=headers).json()

        # Fetch all issues
        issues = self.fetch_all_pages(f'{base_url}/issues?state=all', headers)

        # Fetch all pull requests
        pulls = self.fetch_all_pages(f'{base_url}/pulls', headers)

        # Fetch all commits
        commits = self.fetch_all_pages(f'{base_url}/commits', headers)

        return {
            'repo_info': repo_info,
            'issues': issues,
            'pulls': pulls,
            'commits': commits
        }

    def fetch_all_pages(self, url: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
        """Fetch all pages of results from a GitHub API endpoint."""
        all_results = []
        page = 1
        per_page = 30  # Maximum per_page value is 100

        while True:
            response = requests.get(f'{url}&page={page}&per_page={per_page}', headers=headers)
            response_data = response.json()

            if not response_data:
                break  # Exit if there are no more results

            all_results.extend(response_data)
            page += 1
            if page == 3:
                break
        return all_results

    def setup_qa_chain(self):
        """Set up the QA chain using LangChain"""
        prompt_template = """
        You are an AI assistant that answers questions about a GitHub repository.
        Use the following information about the repository to answer the user's question:

        Repository Name: {repo_name}
        Description: {description}
        Stars: {stars}
        Forks: {forks}
        Open Issues: {open_issues}
        Total Issues: {total_issues}
        Pull Requests: {pull_requests}
        Commits: {commits}

        Recent Issues:
        {recent_issues}

        Human: {question}
        AI: """

        prompt = PromptTemplate(
            input_variables=["repo_name", "description", "stars", "forks", "open_issues", "total_issues",
                             "pull_requests", "commits", "recent_issues", "question"],
            template=prompt_template
        )

        return LLMChain(llm=self.llm, prompt=prompt)

    def setup_retriever(self):
        """Set up a simple retriever to check if the question is relevant."""
        json_data = self.data
        documents = [
            Document(page_content=json.dumps(item), metadata={'source': key})
            for key, value in json_data.items()
            for item in (value if isinstance(value, list) else [value])
        ]

        # Split documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        # Create a retriever from these documents
        embeddings = HuggingFaceEmbeddings()
        vectorstore = Chroma.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever()
        return retriever

    def is_in_scope(self, question: str) -> bool:
        """Check if the question is within the scope of the repository context using the retriever."""
        relevant_docs = self.retriever.get_relevant_documents(question)
        return len(relevant_docs) > 0

    def answer_question(self, question: str) -> str:
        """Answer a question about the repository using the LLM, ensuring it's in scope."""
        if not self.is_in_scope(question):
            return "The question seems to be out of scope for this GitHub repository. Please ask something related to the repository, such as issues, pull requests, commits, or other repository details."

        context = self.prepare_context()
        context['question'] = question

        response = self.qa_chain.run(context)
        return response.strip()

    def prepare_context(self):
        """Prepare the context for the QA chain"""
        repo_info = self.data['repo_info']
        issues = self.data['issues']
        pulls = self.data['pulls']
        commits = self.data['commits']

        recent_issues = "\n".join([f"- {issue['title']}" for issue in issues[:5]])

        return {
            "repo_name": repo_info['name'],
            "description": repo_info['description'],
            "stars": repo_info['stargazers_count'],
            "forks": repo_info['forks_count'],
            "open_issues": repo_info['open_issues_count'],
            "total_issues": len(issues),
            "pull_requests": len(pulls),
            "commits": len(commits),
            "recent_issues": recent_issues
        }
