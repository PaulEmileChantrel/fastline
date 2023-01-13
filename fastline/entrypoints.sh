#!/bin/bash
APP_PORT=${PORT:-8000}
cd /app/
cd fastlinevenv3/bin/
ls
cd ../..
/fastlinevenv3/bin/gunicorn --worker-tmp-dir /dev/shm fastline.wsgi:application --blind "0.0.0.0=${APP_PORT}"