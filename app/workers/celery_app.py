from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery("resume_parser", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.task_routes = {"app.workers.tasks.*": {"queue": "resume_parser"}}
celery_app.autodiscover_tasks(["app.workers"])
