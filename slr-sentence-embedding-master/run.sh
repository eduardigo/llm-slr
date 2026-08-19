for theme in `ls bibs`; do
  echo ""
  echo ""
  date
  echo ""
  python main.py $theme svm true tfidf selectkbest
  python main.py $theme dt true tfidf selectkbest
  python main.py $theme svm false tfidf selectkbest
  python main.py $theme dt false tfidf selectkbest
  python main.py $theme svm true tfidf truncatedsvd
  python main.py $theme dt true tfidf truncatedsvd
  python main.py $theme svm false tfidf truncatedsvd
  python main.py $theme dt false tfidf truncatedsvd

  python main.py $theme svm true embeddings_glove selectkbest
  python main.py $theme dt true embeddings_glove selectkbest
  python main.py $theme svm false embeddings_glove selectkbest
  python main.py $theme dt false embeddings_glove selectkbest
  python main.py $theme svm true embeddings_glove truncatedsvd
  python main.py $theme dt true embeddings_glove truncatedsvd
  python main.py $theme svm false embeddings_glove truncatedsvd
  python main.py $theme dt false embeddings_glove truncatedsvd

  python main.py $theme svm true embeddings_se selectkbest
  python main.py $theme dt true embeddings_se selectkbest
  python main.py $theme svm false embeddings_se selectkbest
  python main.py $theme dt false embeddings_se selectkbest
  python main.py $theme svm true embeddings_se truncatedsvd
  python main.py $theme dt true embeddings_se truncatedsvd
  python main.py $theme svm false embeddings_se truncatedsvd
  python main.py $theme dt false embeddings_se truncatedsvd
done
