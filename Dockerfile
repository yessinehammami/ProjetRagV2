FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8088

HEALTHCHECK CMD curl --fail http://localhost:8088/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8088", "--server.address=0.0.0.0"]
