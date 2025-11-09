# AgriSense

A comprehensive AI-powered multilingual agricultural advisor that provides personalized farming advice, weather insights, market prices, and government policy information in Hindi and English.

> *⚠ Important*: Setup may take 10-15 minutes to load all dependencies and transformer models. Timing depends on your system configuration and internet speed.

> *🔑 API Key Required*: Use your key or if this does not work get your free API key from [Groq Console](https://console.groq.com/keys) and add it to your environment or config file.

## Instant Setup

bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Clone repository
git clone https://github.com/DrunkenMaster2004/DES_6th_Sem

# Run setup
python setup_and_run.py




## Technical Stack

- *Backend*: Python 3.8+, SQLite, FAISS
- *AI/ML*: Groq API (Llama3-8b), Transformers, spaCy
- *Data*: Open-Meteo API, 35K+ price records, 12 policy documents
- *Web*: Streamlit, Plotly
- *NLP*: Sentence Transformers, NLTK

## Current Features

### Multilingual Support
- Hindi and English with automatic language detection
- Handles code-mixed text seamlessly

### Smart Query Processing
- 6 intent categories: Weather, Price, Policy, Technical, Agriculture, General
- Advanced NLP with transformer-based classification

### Price Intelligence
- Real-time mandi prices across India (35,522+ records)
- LLM-based SQL generation for complex queries
- Price trends and market insights

### Weather Analysis
- 7-day weather forecasts with agricultural insights
- Soil moisture and irrigation recommendations
- Location-based weather intelligence

### Policy Guidance
- 12 government policy documents processed
- Vector-based semantic search (973 sections)
- AI-powered policy explanations

### Data Sources
- *Price Data*: mandi_prices.csv (35,522 records)
- *Weather*: Open-Meteo API (free, no key required)
- *Policies*: 12 PDF documents with vector embeddings
- *Soil Health*: 5 districts data for crop recommendations



## Usage

bash
# Command line
python agricultural_advisor_bot.py --interactive

# Web interface
streamlit run streamlit_app.py




## Troubleshooting

*API Key Issues*: Create new key from [Groq Console](https://console.groq.com/keys)

*Installation Timeout*: pip install --timeout 1000 -r requirements.txt

*Database Errors*: python init_mandi_soil.py

*Vector Database Issues*: python improved_policy_chatbot.py

## Future Enhancements

- *Mobile App*: Native Android/iOS applications for farmers
- *Voice Interface*: Speech-to-text and text-to-speech capabilities
- *Image Recognition*: Crop disease detection from photos
- *Real-time Alerts*: Weather warnings and price notifications
- *Multi-language Support*: Additional Indian languages (Punjabi, Tamil, etc.)
- *Offline Mode*: Basic functionality without internet connection
- *Integration*: Connect with government databases and weather stations
- *Analytics Dashboard*: Advanced insights and trend analysis

## Contributing

1. Fork the repository
2. Create feature branch: git checkout -b feature/new-feature
3. Make changes and test
4. Commit and push
5. Create Pull Request

## License

This project is licensed under IIT Kanpur

---

Built with ❤ for the farming community
