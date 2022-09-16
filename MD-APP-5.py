#Project-5- Restaurant Review's Sentiment Analysis -

from flask import Flask,request,render_template,jsonify
import joblib

APP5 = Flask(__name__)
# Load the model pickle from current directory
mnbc = joblib.load('train_md5.pkl')
cv = joblib.load('transform_md5.pkl')

@APP5.route('/')
def home():
    return render_template('index.html')


@APP5.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = mnbc.predict(vect)
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	APP5.run(debug=True)

