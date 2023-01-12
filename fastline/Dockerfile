FROM python:3.10.0-slim

COPY . /app
WORKDIR /app

RUN python3 -m venv /otp/venv

RUN pip install pip --upgrade && /otp/venv/bin/pip install -r requirement.txt && chmod +x entrypoint.sh

CMD ["/app/entrypoints.sh"]
