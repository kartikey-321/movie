from flask import Flask, app,render_template,redirect,request
import joblib
import nltk

model=joblib.load('movie.pkl')

app=Flask(__name__)

cv=joblib.load('transform.pkl')

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        movie=request.form['movie']
        data=[movie]
        vect=cv.transform(data)
        output=model.predict(vect)
        # output=output[0]
    return render_template('predict.html',value=output)






if __name__=="__main__":
    app.run(debug=True)
