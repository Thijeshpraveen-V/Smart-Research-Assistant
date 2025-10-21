# 🧠 Smart Research Assistant

**Smart Research Assistant** is a Flask-based AI web application that helps users perform intelligent academic research.  
It uses **LangChain**, **Groq LLMs**, and **Tavily Search** to generate concise topic summaries, fetch research papers, and handle follow-up questions with context memory.

---

## 🚀 Features

- 🔍 **Smart Research Generation** – Summarizes topics and lists relevant academic papers.
- 💬 **Contextual Memory** – Understands follow-up questions intelligently.
- 🧾 **Downloadable Reports** – Export any research response as a well-formatted PDF.
- 🧠 **Agent Reasoning** – Uses LangChain’s ReAct-style agent for logical reasoning.
- 🧹 **New Topic Option** – Start fresh anytime with a single click.
- 🎨 **Modern UI** – Sleek dark-themed responsive interface.

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| **Backend** | Flask (Python) |
| **AI Model** | Groq LLM – `moonshotai/kimi-k2-instruct-0905` |
| **Search Engine** | Tavily Search API |
| **Framework** | LangChain (ReAct Agent) |
| **Database** | SQLite3 |
| **Frontend** | HTML, CSS, JavaScript |
| **PDF Generator** | pdfkit + wkhtmltopdf |

---

## 🧑‍💻 How to Run the Project

### 1️⃣ Clone this Repository
```bash
git clone https://github.com/your-username/smart-research-assistant.git
cd smart-research-assistant
