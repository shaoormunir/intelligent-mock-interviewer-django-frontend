from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect

import requests

# Create your views here.
from Analysis.interviewAnalysis import InterviewAnalysis


def index(request):
    if request.method == "POST" and request.FILES['interview_video']:
        myfile = request.FILES['interview_video']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)

        analyzer = InterviewAnalysis(filename)
        openness, extraversion, neuroticism, agreeableness, conscientiousness = analyzer.analyzeVideo()

        if openness is not None:
            return render(request, "report.html",
                      {'openness': openness, 'neuroticism': neuroticism, 'extraversion': extraversion,
                       'agreeableness': agreeableness, 'conscientiousness': conscientiousness})
        else:
            raise LookupError


    return render(request, 'index.html')

def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email', None)
        password = request.POST.get('password', None)

        user = authenticate(username=email, password=password)
        login(request, user)

        return render(request, 'index.html')

    return render(request, 'login.html')

def signup_view(request):
    if request.method == 'POST':
        email = request.POST.get('email', None)
        password = request.POST.get('password', None)
        name = request.POST.get('name', None)

        firstname = name.strip().split(' ')[0]
        lastname = ' '.join((name + ' ').split(' ')[1:]).strip()

        if email and password:
            user, created = User.objects.get_or_create(username=email,
                                                       email=email,first_name=firstname, last_name=lastname)

            if created:
                user.set_password(password)
                user.save()

            user = authenticate(username=email, password=password)

            login(request, user)

            return render(request, 'index.html')
    return render(request, 'signup.html')

def logout_view(request):
    logout(request)
    # Redirect to a success page.
    return render(request, 'index.html')
def rooms(request):
    return render(request, 'rooms.html')

def report(request):
    return render(request, 'report.html')