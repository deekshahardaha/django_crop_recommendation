from django.shortcuts import render, HttpResponse
import croppredictionsystem
import joblib
# Create your views here.
def index(request):
    return render(request, 'index.html')

def result(request):

    cls = joblib.load('finalized_model.sav')

    lliisstt=[]

    lliisstt.append(request.GET['Nitrogen'])
    lliisstt.append(request.GET['Phosphorus'])
    lliisstt.append(request.GET['Potassium'])
    lliisstt.append(request.GET['Temperature'])
    lliisstt.append(request.GET['Humidity'])
    lliisstt.append(request.GET['pH'])
    lliisstt.append(request.GET['Rainfall'])

    lliisstt=list(round(float(a),2) for a in lliisstt)
    
    ans = cls.predict([lliisstt])
    ans = croppredictionsystem.lblencoder.inverse_transform(ans)
    return render(request, "result.html",{'ans':ans})