# Education Chatbot Assistant

This project is an educational chatbot assistant designed to help undergraduate students understand concepts and definitions through short, memorable stories. The chatbot leverages LangChain for managing interactions and a vector database for efficient data storage and retrieval. This ReadMe provides an overview of the project, setup instructions, and usage guidelines.

## Objective

The goal of this assignment is to develop a domain-specific application that combines the strengths of a Large Language Model (LLM) for understanding and processing natural language queries with the efficiency of a vector database for data storage and retrieval. This application focuses on creating a responsive chatbot for the educational domain, providing personalized recommendations using Retrieval-Augmented Generation (RAG).

## Key Features

- **User Interaction:** Allows users to input their preferences and needs through a conversational interface.
- **RAG-Based Recommendations:** Generates recommendations based on user inputs using RAG.
- **Intelligent Responses:** Provides relevant information and suggestions, ensuring recommendations are accurate to the user's context and preferences.
- **Pre-existing Data:** Recommendations are created only from the existing data in the vector store.
- **Backend Integration:** Utilizes LangChain to manage the flow of interactions, with data stored in a vector database.
- **Data Fetching:** Fetches the top K (K <= 3) data entries based on the user's description using similarity search.

## Getting Started

### Prerequisites

- Python 3.7+
- `chromadb` package
- `streamlit` package
- OpenAI API key

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/education-chatbot.git
   cd education-chatbot
   
2. **Create a virtual environment and activate it:** 
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install the required packages:** 
    ```bash
    pip install -r requirements.txt


4. **Set up your OpenAI API key:**
   [OPENAI API Documentation](https://platform.openai.com/docs/quickstart)
    ```bash
    Create a .env File and set up your openai api key
    OPENAI_API_KEY=your_openai_api_key

### Running the Application

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py


2. **Open your browser:** 
    ```bash
    Navigate to http://localhost:8501 to use the BrainBytes application.

3. **Upload a PDF file:**
   ```bash
   Navigate to the "Upload PDF" tab.
   Choose a PDF file to load into the chatbot.

4. **View Data:**
   ```bash
   Navigate to the "View Data" tab.
   View the vectors stored in the vector database.

6. **Interact with the Chatbot:**
   ```bash
   Navigate to the "Chatbot" tab.
   Enter your question and generate responses from the chatbot.

**Clearing Data:**
    ```bash
  To clear the database, go to the "View Data" tab and click "Clear Database." 
  To clear chat history, go to the "Chatbot" tab and click "Clear Chat History."

## PDF Report
Report: [Report](./RAG_Report.pdf)

## Demo Video Demonstration
Watch the video demonstration to see the application in action: [https://youtu.be/oYt6LAaVhmo](https://youtu.be/uWlA38_H2kE)

## Licensing

Copyright 2024 Ankit Goyal

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
