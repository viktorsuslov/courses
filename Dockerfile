FROM python:3

WORKDIR /app
COPY torch.py /app/torch.py
CMD ["python", "/app/torch.py"]
