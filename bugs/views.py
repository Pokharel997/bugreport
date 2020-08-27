from django.shortcuts import render
import subprocess

def home(request):
    return render(request,'home.html')

def train(request):
    return render(request,'train.html')

def homepage(request):
    return render(request = request, template_name="home.html")

def dataset(request):
    return render(request,'dataset.html')

def about(request):
    return render(request,'about.html')


def trained(request):
    return render(request,'traiin.html')

def make(request):
    if request.method == "POST":
        a = request.POST['input']
        b = "python bugreportjupyter.py '" + a +  "'"
        output = subprocess.check_output(b, shell=True)
        out = output.decode().split('#')
        return render(request,"home1.html",{'var': out})
