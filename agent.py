# agent.py

import os
import io
import contextlib
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from flask import Flask, render_template, request, make_response
from langchain.callbacks.base import BaseCallbackHandler
import pdfkit
import sqlite3
import markdown2


app = Flask(__name__)
# Load API keys from the .env file
load_dotenv()

class MyCustomCallbackHandler(BaseCallbackHandler):
    """A custom callback handler to capture the agent's thought process."""
    def __init__(self):
        super().__init__()
        self.log = "" # A string to store the entire log

    def on_agent_action(self, action, **kwargs) -> None:
        """Append agent's action to the log."""
        self.log += f"\nThought: {kwargs.get('log')}\nAction: {action.tool}\nAction Input: {action.tool_input}\n"

    def on_tool_end(self, output, **kwargs) -> None:
        """Append tool's observation to the log."""
        self.log += f"Observation: {output}\n"

    def get_log(self) -> str:
        """Return the captured log."""
        return self.log
# --- LLM INITIALIZATION FOR HUGGING FACE ---

llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0)


# Define the set of tools the agent can use
search_tool = TavilySearch(max_results=5)
tools = [search_tool]

# This is the prompt template that instructs the agent
# In app.py

prompt = PromptTemplate.from_template(
    """
    You are a hyper-competent academic research assistant. Your mission is to provide a concise summary and a list of papers from elite academic sources.

    **INSTRUCTIONS:**
    1.  Perform a broad web search to write a brief, one-paragraph summary of the user's topic.
    2.  Next, to find academic papers, you MUST perform a dedicated search using an advanced query with `site:` operators. Search **only** the following elite academic websites: Google Scholar, arXiv (Cornell), IEEE Xplore, ScienceDirect, PubMed, ACM Digital Library, and JSTOR.
        * Example of a good Action Input: `"neural networks" site:scholar.google.com OR site:arxiv.org OR site:ieeexplore.ieee.org OR site:sciencedirect.com OR site:pubmed.ncbi.nlm.nih.gov OR site:dl.acm.org OR site:jstor.org`
    3.  From your search results, identify and list up to 5 of the most relevant papers.
    4.  Present the final answer with the summary first, followed by the list of research papers with their titles and clickable Markdown links.

    You have access to the following tools:
    {tools}

    Use the following format:
    Question: {input}
    Thought: Your reasoning here...
    Action: The action to take, should be one of [{tool_names}]
    Action Input: The input to the action
    Observation: The result of the action
    ... (This sequence can repeat)
    Thought: I have finished my research and will now provide the final answer.
    Final Answer: [Your final answer with the summary and the list of paper links]

    Begin!
    Question: {input}
    Thought:{agent_scratchpad}
    """
)
# 1. Create the Agent
agent = create_react_agent(llm, tools, prompt)

# 2. Create the Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

# --- FLASK ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
        # Load the entire conversation history from the database
    conn = sqlite3.connect('conversation_history.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    chat_history = cursor.fetchall()
    conn.close()


    final_answer = None
    thinking_process = None
    query = None

    if request.method == 'POST':
        query = request.form['query']
        
        # This is the magic part: we capture the stdout (print statements)
        # from the agent's execution to display the "thinking" process
        # Create an instance of our custom handler
        handler = MyCustomCallbackHandler()
        
        # Invoke the agent, passing the handler in the config
        # The handler will automatically capture the logs
        result = agent_executor.invoke(
            {"input": query},
            config={"callbacks": [handler]}
        )

        # Get the logs and the final answer
        thinking_process = handler.get_log()
        final_answer = result["output"]

        conn = sqlite3.connect('conversation_history.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO history (role, content) VALUES (?, ?)", ('user', query))
        cursor.execute("INSERT INTO history (role, content) VALUES (?, ?)", ('assistant', final_answer))
        conn.commit()
        conn.close()

         # Reload history to include the new messages for immediate display
        conn = sqlite3.connect('conversation_history.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history ORDER BY id DESC")
        chat_history = cursor.fetchall()
        conn.close()

    return render_template(
        'index.html', 
        query=query, 
        thinking_process=thinking_process, 
        final_answer=final_answer,
        chat_history=chat_history
    )

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    # Get the HTML content from the hidden form field
    html_content = request.form['html_content']
    pdf_html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Research Report</title>
        <style>
            body {{
                font-family: Georgia, serif; /* Use a classic, readable serif font */
                font-size: 12pt;             /* Increase base font size */
                line-height: 1.5;            /* Improve line spacing for readability */
                color: #333;
            }}
            h1, h2, h3 {{
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; /* Use a clean sans-serif for headings */
                color: #2c3e50;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 10px;
            }}
            img {{
                max-width: 90%; /* Ensure images don't overflow the page */
                height: auto;
                display: block;
                margin: 20px auto; /* Center images */
            }}
            a {{
                color: #2980b9;
                text-decoration: none;
            }}
            ul, ol {{
                padding-left: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Research Report</h1>
        {html_content}
    </body>
    </html>
    """

    # Convert the HTML string to a PDF
    # The 'enable-local-file-access' option is important for images
    options = {'enable-local-file-access': None}
    pdf = pdfkit.from_string(html_content, False, options=options)

    # Create a response to send the PDF file to the browser
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=research_report.pdf'

    return response


@app.route('/download_history/<int:message_id>')
def download_history(message_id):
    conn = sqlite3.connect('conversation_history.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history WHERE id = ?", (message_id,))
    message = cursor.fetchone()
    conn.close()

    if message and message['role'] == 'assistant':
        # Convert the stored Markdown content to HTML
        html_from_markdown = markdown2.markdown(message['content'])

        # Create the full, styled HTML document for the PDF
        pdf_html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Research Report</title>
            <style>
                body {{ font-family: Georgia, serif; font-size: 12pt; line-height: 1.5; color: #333; }}
                h1, h2, h3 {{ font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
                a {{ color: #2980b9; text-decoration: none; }}
                ul, ol {{ padding-left: 20px; }}
            </style>
        </head>
        <body>
            <h1>Research Report</h1>
            {html_from_markdown}
        </body>
        </html>
        """
        path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
        config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
        pdf = pdfkit.from_string(pdf_html_content, False, configuration=config)

        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=report_{message_id}.pdf'
        return response
    else:
        return "Report not found or invalid request.", 404


if __name__ == '__main__':
    app.run(debug=True)