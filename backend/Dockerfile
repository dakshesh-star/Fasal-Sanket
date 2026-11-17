FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv/backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev libopenjp2-7 libopenjp2-7-dev libjpeg62-turbo libtiff6 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app ./app
COPY backend/data ./data
COPY backend/models ./models
COPY frontend /srv/frontend
COPY backend/alembic.ini ./alembic.ini
COPY backend/alembic ./alembic
COPY backend/scripts ./scripts
CMD ["celery", "-A", "app.celery_app:celery_app", "worker", "--beat", "--loglevel=INFO", "--concurrency=1"]
