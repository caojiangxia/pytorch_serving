FROM python:3.6

ADD ./ /opt/pytorch_serving
WORKDIR /opt/pytorch_serving

RUN chmod +x bin/manage \
  && pip install -r requirements/app.txt

CMD bin/manage start --daemon-off
