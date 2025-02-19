# Custom Search Engine 🚀

A Python-based custom search engine that retrieves and displays search results using the **Gemini API**.

## 🌟 Features
- Accepts user search queries via command line.
- Fetches and processes search results using an API.
- Displays well-formatted search results with titles, descriptions, and links.
- Handles API errors gracefully.
- Supports structured output for easy readability.

## 📌 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Md-Abu-Omayer-Babu/Custom_Search_Engine.git
cd Custom_Search_Engine
```

### 2️⃣ Create & Activate a Virtual Environment *(Optional but Recommended)*
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration
1. Obtain an API key from **Gemini API**.
2. Create a `.env` file and add your API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

## 🚀 Usage
Run the script and enter your search query:
```bash
python main.py
```
Then, input your query when prompted.

## 🛠 Troubleshooting
- **"List is not defined"** ➝ Ensure you have imported `List` from `typing` (`from typing import List`).
- **Invalid JSON response** ➝ The API might be returning unexpected results. Check your API key and request format.
- **GRPC timeout errors** ➝ May indicate network issues or API rate limits.

## 📜 License
This project is licensed under the **MIT License**.

## 🤝 Contributing
Pull requests are welcome! For major changes, open an issue first to discuss your ideas.

## 📬 Contact
For any queries, feel free to reach out via:
- GitHub: [Md-Abu-Omayer-Babu](https://github.com/Md-Abu-Omayer-Babu)

Happy Coding! 🚀

