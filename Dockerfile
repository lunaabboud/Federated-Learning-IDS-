FROM python:3.9

WORKDIR /app

# Copy only requirements first (this layer changes rarely)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Now copy the rest of the code (this layer changes often)
COPY . /app/

CMD ["python", "server.py"]
