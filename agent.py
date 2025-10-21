# agent.py

import os
import io
import contextlib
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from flask import Flask, render_template, request, make_response, session, redirect, url_for
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import pdfkit
import sqlite3
import markdown2


app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

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

# --- LLM INITIALIZATION ---
llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0)

# Define the set of tools the agent can use
search_tool = TavilySearch(max_results=5)
tools = [search_tool]

# Initial research prompt (for main topics)
initial_research_prompt = PromptTemplate.from_template(
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

# Follow-up question prompt (for contextual questions)
followup_prompt = PromptTemplate.from_template(
    """
    You are a helpful research assistant answering follow-up questions based on previous research.

    **PREVIOUS CONVERSATION:**
    {chat_history}

    **INSTRUCTIONS:**
    1. Use the previous conversation context to answer the user's follow-up question.
    2. Provide detailed, informative answers WITHOUT searching for new papers or providing new links.
    3. If you need additional information, perform a focused web search, but do NOT list academic papers or links in your response.
    4. Keep your answer conversational, detailed, and directly address the user's question.
    5. Do NOT include paper links, citations, or bibliographies unless specifically asked.

    You have access to the following tools:
    {tools}

    Use the following format:
    Question: {input}
    Thought: Your reasoning here...
    Action: The action to take, should be one of [{tool_names}]
    Action Input: The input to the action
    Observation: The result of the action
    ... (This sequence can repeat)
    Thought: I now have enough information to answer the question.
    Final Answer: [Your conversational answer without paper links]

    Begin!
    Question: {input}
    Thought:{agent_scratchpad}
    """
)

# --- FLASK ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize session memory if it doesn't exist
    if 'conversation_memory' not in session:
        session['conversation_memory'] = []
    if 'current_conversation' not in session:
        session['current_conversation'] = []
    
    # Load the entire conversation history from the database
    conn = sqlite3.connect('conversation_history.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    chat_history = cursor.fetchall()
    conn.close()

    if request.method == 'POST':
        query = request.form['query']
        is_followup = request.form.get('is_followup', 'false') == 'true'
        
        # Format chat history for the agent
        chat_history_str = ""
        for msg in session['conversation_memory'][-6:]:  # Last 3 exchanges
            role = "Human" if msg['role'] == 'user' else "Assistant"
            chat_history_str += f"{role}: {msg['content']}\n"
        
        # Create an instance of our custom handler
        handler = MyCustomCallbackHandler()
        
        # Choose prompt based on whether it's a follow-up
        if is_followup and len(session['conversation_memory']) > 0:
            agent = create_react_agent(llm, tools, followup_prompt)
        else:
            agent = create_react_agent(llm, tools, initial_research_prompt)
            # Clear current conversation for new topic
            session['current_conversation'] = []
        
        # Create agent executor for this request
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=False, 
            handle_parsing_errors=True
        )
        
        # Invoke the agent with chat history
        result = agent_executor.invoke(
            {
                "input": query,
                "chat_history": chat_history_str
            },
            config={"callbacks": [handler]}
        )

        # Get the logs and the final answer
        thinking_process = handler.get_log()
        final_answer = result["output"]

        # Store in database
        conn = sqlite3.connect('conversation_history.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO history (role, content) VALUES (?, ?)", ('user', query))
        cursor.execute("INSERT INTO history (role, content) VALUES (?, ?)", ('assistant', final_answer))
        conn.commit()
        conn.close()

        # Update session memory
        session['conversation_memory'].append({'role': 'user', 'content': query})
        session['conversation_memory'].append({'role': 'assistant', 'content': final_answer})
        
        # Update current conversation (for display)
        session['current_conversation'].append({'role': 'user', 'content': query})
        session['current_conversation'].append({'role': 'assistant', 'content': final_answer})
        
        session.modified = True

        # Reload history
        conn = sqlite3.connect('conversation_history.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history ORDER BY id DESC")
        chat_history = cursor.fetchall()
        conn.close()

    return render_template(
        'index.html', 
        chat_history=chat_history,
        current_conversation=session.get('current_conversation', []),
        has_conversation=len(session.get('current_conversation', [])) > 0
    )

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    """Clear the conversation memory for a fresh start"""
    session['conversation_memory'] = []
    session['current_conversation'] = []
    return redirect(url_for('index'))

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
                font-family: Georgia, serif;
                font-size: 12pt;
                line-height: 1.5;
                color: #333;
            }}
            h1, h2, h3 {{
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                color: #2c3e50;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 10px;
            }}
            img {{
                max-width: 90%;
                height: auto;
                display: block;
                margin: 20px auto;
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

    options = {'enable-local-file-access': None}
    pdf = pdfkit.from_string(html_content, False, options=options)

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
        html_from_markdown = markdown2.markdown(message['content'])

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