FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "week_4_day_4_interactive_dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]