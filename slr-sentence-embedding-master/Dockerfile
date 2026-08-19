FROM tensorflow/tensorflow:2.16.1

WORKDIR /app

RUN apt-get update
RUN apt-get -y install build-essential \
                       gfortran

RUN pip install -U pip setuptools wheel
RUN pip install -U scikit-learn

RUN pip install -U spacy
RUN python -m spacy download pt_core_news_sm

RUN pip install scipy==1.12
RUN pip install imblearn \
                pot \
                pandas \
                gensim \
                keras \
                keras-nlp \
                bs4 \
                transformers \
                tf-keras \
                numpy \
                openpyxl \
                bibtexparser \
                nltk \
                np \
                cython \
                gensim

RUN python -m nltk.downloader punkt \
                              stopwords \
                              averaged_perceptron_tagger \
                              wordnet \
                              omw-1.4

CMD ["bash"]
