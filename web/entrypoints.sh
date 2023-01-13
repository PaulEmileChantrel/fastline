#!/bin/bash
APP_PORT=${PORT:-8000}
cd /app/fastline/
/opt/fastlinevenv3/bin/gunicorn --worker-tmp-dir /dev/shm fastline.wsgi:application --bind "0.0.0.0:${APP_PORT}"