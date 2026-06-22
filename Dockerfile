FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if any (none needed for pure-python pymysql, but curl is good for healthchecks, git is for project history awareness)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
