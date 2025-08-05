# Create venv (only on initial setup)
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Start local app
streamlit run main.py
