FROM python:3.11.5-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENV PORT=8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 0 wsgi:application