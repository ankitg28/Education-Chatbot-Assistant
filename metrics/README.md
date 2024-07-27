# Evaluation Metrics

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

2. **cd into the `evaluation_metrics` directory.**
    ```bash
    cd metrics
   
3. **Create a virtual environment and activate it:** 
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

4. **Install the required packages:** 
    ```bash
    pip install -r requirements.txt


5. **Set up your OpenAI API key:**
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

4. **Check the calculated Evaluation Metrics:**

## Retrieval Metrics

| Metric                | Description                                                                            | Value |
|-----------------------|----------------------------------------------------------------------------------------|-------|
| Context Precision     | Measures the proportion of relevant context among the retrieved context                | 0.92  |
| Context Recall        | Measures the proportion of relevant context that is successfully retrieved             | 0.96  |
| Context Relevance     | Assesses the relevance of the retrieved context to the given query or information need | 0.83  |
| Context Entity Recall | Measures the proportion of relevant entities that are successfully retrieved           | 0.33  |
| Noise Robustness      | Evaluates the ability of the retrieval system to handle noisy or irrelevant context    | 0.96  |

## Generation Metrics

| Metric                    | Description                                                                             | Value   |
|---------------------------|-----------------------------------------------------------------------------------------|---------|
| Faithfulness              | Measure the accuracy and reliability of the generated answers.                          | 1.00    |
| Answer Relevance          | Evaluate the relevance of the generated answers to the user's query.                    | 0.67    |
| Information Integration   | Assess the ability to integrate and present information cohesively.                     | 1.00    |
| Counterfactual Robustness | Test the robustness of the system against counterfactual or contradictory queries.      | 1.00    |
| Negative Rejection        | Measure the system's ability to reject and handle negative or inappropriate queries.    | 0.00    |
| Latency                   | Measure the response time of the system from receiving a query to delivering an answer. | 0.00 seconds |

## PDF Report
Report: [EvaluationReport](./EvaluationReport.pdf)

## Demo Video Demonstration
Watch the video demonstration to see the application in action: [https://youtu.be/oYt6LAaVhmo](https://youtu.be/jUzoynBx7gg)

## Licensing

Copyright 2024 Ankit Goyal

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

