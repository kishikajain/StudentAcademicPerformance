# Student Academic Performance Prediction using Deep Neural Networks

Dataset: UCI Student Performance (`student-mat.csv` or `student-por.csv`), target G3.
G3 is converted to a category: Low (0-9), Average (10-14), High (15-20).

```bash
pip install -r requirements.txt
python train.py --data student-mat.csv
streamlit run app.py
```

`train.py` covers: preprocessing, EDA, feature selection, DNN design, training,
hyperparameter tuning, evaluation, and comparison with Random Forest.
`app.py` is the Streamlit app.
