FROM python:3.10.0-slim

COPY . /app
WORKDIR /app

RUN python3 -m venv /otp/fastlinevenv3

RUN pip install pip --upgrade && \
    /otp/fastlinevenv3/bin/pip install -r requirement.txt && \
    chmod +x entrypoints.sh

CMD ["/app/entrypoints.sh"]
