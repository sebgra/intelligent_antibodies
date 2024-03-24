FROM tensorflow/tensorflow:devel-gpu 

RUN mkdir /src

COPY ./ /src/

RUN ls -la /src/

WORKDIR /src/

RUN pip install -r requirements.txt

ENTRYPOINT [ "flask",  "--app", "app.py", "run" ]