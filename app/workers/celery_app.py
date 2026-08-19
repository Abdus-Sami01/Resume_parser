from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery("resume_parser", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.task_routes = {"app.workers.tasks.*": {"queue": "resume_parser"}}

# Eager mode runs tasks inline so the API works with no broker running. Failures are
# captured into the result rather than raised, matching how a real worker reports them.
celery_app.conf.task_always_eager = settings.task_backend == "eager"
celery_app.conf.task_eager_propagates = False

celery_app.autodiscover_tasks(["app.workers"])
