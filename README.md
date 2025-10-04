# ProPlay Assistant ChatBot

A Streamlit-based AI chatbot that helps users with sports-related queries, providing real-time information and assistance about various sports, teams, players, and events.

## Features

- Real-time sports information and updates
- Interactive chat interface using Streamlit
- Powered by Groq AI for accurate responses
- Multiple AI model options (llama-3.1-70b, llama-3.1-8b, mixtral-8x7b)
- Fast and Accurate operation modes
- Google search integration for up-to-date information
- Chat session management with save/load capability
- Chat summarization with PDF export
- Support for various sports topics and queries
- Beautiful, responsive UI with dark theme

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MADSKULL21/ProPlay-Assistant.git
cd ProPlay-Assistant-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following variables:
```env
GroqAPIKey=your_groq_api_key
Username=your_preferred_username
Assistantname=Sports Assistant
```

4. Run the application:
```bash
streamlit run src/app.py
```

### Optional CLI utilities

- `ChatBot.py`: simple console chatbot that logs to `Data/ChatLog.json`.
- `RealtimeSearchEngine.py`: console tool that enriches prompts with Google search snippets.

## Usage

1. Launch the application using Streamlit
2. Use the sidebar to select model and creativity, and to clear chat
3. Enter your sports-related question in the chat input
4. Get real-time responses about:
   - Latest sports news and updates
   - Player statistics and information
   - Team performance and rankings
   - Tournament schedules and results
   - Sports rules and regulations
5. Additional features:
   - Save and manage multiple chat sessions
   - Rename chat sessions for better organization
   - Generate and download chat summaries as PDFs
   - Switch between Fast and Accurate modes
   - Adjust AI model creativity with temperature control

## Project Structure

```
├── src/
│   ├── app.py              # Streamlit UI (chat, sidebar, styling)
│   ├── search_engine.py    # Groq + Google search integration
│   └── __pycache__/        # Python bytecode cache
├── Data/
│   ├── ChatLog.json        # Chat history storage
│   ├── Sessions/           # Individual chat session storage
│   │   └── index.json      # Sessions index file
│   └── Summaries/          # Chat summary storage
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
└── .env                   # Environment variables
```

## Contributing

Feel free to submit issues and enhancement requests!