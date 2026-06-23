FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and Docker CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    gnupg \
    lsb-release \
    ca-certificates \
    libicu-dev \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update && apt-get install -y --no-install-recommends \
    docker-ce-cli \
    docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Install .NET SDK 8.0 via official install script to bypass Debian GPG SHA1 policy issues
RUN curl -fsSL https://dot.net/v1/dotnet-install.sh -o dotnet-install.sh \
    && chmod +x dotnet-install.sh \
    && ./dotnet-install.sh --channel 8.0 --install-dir /usr/share/dotnet \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet \
    && rm dotnet-install.sh

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Publish the dotnet bridge project
RUN dotnet publish -c Release -o /app/bridge_publish /app/dotnet_bridge/dotnet_bridge.csproj

# Save image build timestamp
RUN date +%s > /app/build_time.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
