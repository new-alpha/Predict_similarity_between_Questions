from django.shortcuts import render
from  . import ml_predict

def home(request):
    return render(request,'index.html')

def result(request):
    ques1 = request.GET['ques1']
    ques2 = request.GET['ques2']
    prediction = ml_predict.prediction_model(ques1,ques2)
    mydict = {'ques1':ques1,            
                'ques2':ques2,
                 'probability':prediction}
    return render(request,'result.html',mydict)
