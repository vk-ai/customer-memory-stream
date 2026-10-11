FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY cms ./cms
RUN pip install --no-cache-dir .
CMD ["uvicorn", "cms.api:app", "--host", "0.0.0.0", "--port", "8000"]
