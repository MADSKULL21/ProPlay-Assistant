# 🏆 ProPlay Assistant ChatBot

**ProPlay Assistant** is a **Streamlit-based AI chatbot** that helps users with sports-related queries — offering **real-time updates, player stats, team information**, and more, all powered by **Groq AI** for lightning-fast, accurate responses.

---

## 🚀 Features

- ⚡ **Real-time sports information and updates**  
- 💬 **Interactive chat interface** built with Streamlit  
- 🧠 **Powered by Groq AI** for accurate and fast responses  
- 🔁 **Multiple AI model options** via Groq API:  
  - **LLaMA 3.1 - 70B** (high accuracy)  
  - **LLaMA 3.1 - 8B** (lightweight and fast)  
  - **Mixtral 8x7B** (balanced performance)  
- 🧩 **Fast** and **Accurate** operation modes  
- 🔍 **Google search integration** for up-to-date insights  
- 💾 **Chat session management** — save, load, and rename chats  
- 📝 **Chat summarization** with PDF export  
- 🎯 **Wide sports coverage** — teams, players, schedules, and more  
- 🌙 **Beautiful, responsive dark-themed UI**

---

## 🎮 Live Demo

Try the live Streamlit app here:  
👉 [**ProPlay Assistant on Streamlit**](https://proplay-assistant-sml21.streamlit.app/)

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/MADSKULL21/ProPlay-Assistant.git
cd ProPlay-Assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory and add:
```env
GROQ_API_KEY=your_groq_api_key
USERNAME=your_preferred_username
ASSISTANT_NAME=Sports Assistant
```

### 4. Run the application
```bash
streamlit run src/app.py
```

---

## 💡 Optional CLI Utilities

- **ChatBot.py** – Simple console chatbot that logs interactions to `Data/ChatLog.json`.  
- **RealtimeSearchEngine.py** – Command-line tool that enhances prompts using Google Search snippets.

---

## 🧠 Usage

1. Launch the app using Streamlit.  
2. Use the **sidebar** to:
   - Select an AI model and creativity level (temperature).  
   - Switch between **Fast** and **Accurate** modes.  
   - Clear or rename chat sessions.  
3. Type your **sports-related question** in the input box.  
4. Get real-time insights on:
   - 📰 Latest sports news and events  
   - 🧍 Player stats and biographies  
   - 🏟️ Team rankings and performance  
   - 🏆 Tournament results and schedules  
   - 📜 Rules, formats, and historical data  
5. Additional features:
   - Save, rename, and load chat sessions.  
   - Generate chat summaries and **export them as PDFs**.  
   - Responsive, user-friendly dark theme.

---

## 📁 Project Structure

```
├── src/
│   ├── app.py              # Streamlit app (UI, chat logic, sidebar)
│   ├── search_engine.py    # Groq + Google search integration
│   └── __pycache__/        # Python cache
├── Data/
│   ├── ChatLog.json        # Chat history storage
│   ├── Sessions/           # Individual chat sessions
│   │   └── index.json      # Session index file
│   └── Summaries/          # Chat summaries (PDF/text)
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── .env                     # Environment variables
```

---

## 🧩 Version
**v1.0.0** — Initial release with Groq API integration, Google search support, and Streamlit chat UI.

---

## 🤝 Contributing

Contributions are welcome!  
- Fork the repo  
- Create your feature branch (`git checkout -b feature-name`)  
- Commit your changes  
- Submit a Pull Request 🎉  

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

**Developed with ❤️ using Streamlit & Groq AI**
