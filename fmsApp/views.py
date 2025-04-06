from django.shortcuts import render,redirect
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from fms_django.settings import MEDIA_ROOT, MEDIA_URL
import json
from django.contrib import messages
from django.contrib.auth.models import User
from django.http import HttpResponse
from fmsApp.forms import UserRegistration, SavePost, UpdateProfile, UpdatePasswords
from fmsApp.models import Post
from cryptography.fernet import Fernet
from django.conf import settings
from django.core.cache import cache
import base64
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from .models import Post
import base64
# from .stegomarkov_old import Encoder, Decoder, file_to_bitstream, bitstream_to_file, build_model
from .stegomarkov import Encoder, Decoder, file_to_bitstream, bitstream_to_file, build_model
import markovify
import os
import time
import logging
# Create your views here.

context = {
    'page_title' : 'File Management System',
}

# Set up basic logging
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)

def get_markov_model():
    try:
        if 'markov_model' in globals() and globals()['markov_model'] is not None:
            logger.info("Successfully retrieved pre-loaded Markov model.")
            return globals()['markov_model']
        else:
            logger.warning("Markov model not loaded or invalid. Returning None.")
            return None
    except Exception as e:
        logger.error(f"Error retrieving Markov model: {str(e)}")
        return None

#login
def login_user(request):
    logout(request)
    resp = {"status":'failed','msg':''}
    username = ''
    password = ''
    if request.POST:
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                resp['status']='success'
            else:
                resp['msg'] = "Incorrect username or password"
        else:
            resp['msg'] = "Incorrect username or password"
    return HttpResponse(json.dumps(resp),content_type='application/json')

#Logout
def logoutuser(request):
    logout(request)
    return redirect('/')

@login_required
def home(request):
    context['page_title'] = 'Home'
    if request.user.is_superuser:
        posts = Post.objects.all()
    else:
        posts = Post.objects.filter(user = request.user).all()
    context['posts'] = posts
    context['postsLen'] = posts.count()
    print(request.build_absolute_uri())
    return render(request, 'home.html',context)

def registerUser(request):
    user = request.user
    if user.is_authenticated:
        return redirect('home-page')
    context['page_title'] = "Register User"
    if request.method == 'POST':
        data = request.POST
        form = UserRegistration(data)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            pwd = form.cleaned_data.get('password1')
            loginUser = authenticate(username= username, password = pwd)
            login(request, loginUser)
            return redirect('home-page')
        else:
            context['reg_form'] = form

    return render(request,'register.html',context)

@login_required
def profile(request):
    context['page_title'] = 'Profile'
    return render(request, 'profile.html',context)

@login_required
def posts_mgt(request):
    context['page_title'] = 'Uploads'

    posts = Post.objects.filter(user = request.user).order_by('title', '-date_created').all()
    context['posts'] = posts
    return render(request, 'posts_mgt.html', context)

@login_required
def manage_post(request, pk=None):
    resp = {'status': 'failed', 'msg': ''}

    if request.method == 'GET':
        user_id = request.user.id
        cache.set(f'decode_progress_{user_id}', 0, timeout=1800)


    try:
        if pk is not None:
            post = get_object_or_404(Post, id=pk)
            user_id = request.user.id

            if post.file_data:
                model = get_markov_model()
                logger.info("Markov model loaded successfully for decoding.")

                decode_start_time = time.time()
                decoder = Decoder(model, post.file_data, logging=True)
                old_progress = 0

                # Stepwise decoding to update progress
                while not decoder.finished:
                    progress = min(int(decoder.step() * 100), 100)
                    if progress - old_progress >= 5:  # Update every 5%
                        cache.set(f'decode_progress_{user_id}', progress)
                        old_progress = progress
                        logger.info(f"Decoding Progress: {progress}%")

                decoded_file_data = decoder.solve()
                decode_duration = time.time() - decode_start_time

                logger.info(f"Decoding completed in {decode_duration:.6f} seconds")
                context['decoded_file_data'] = decoded_file_data
                cache.set(f'decode_progress_{user_id}', 100, timeout=1800)  # Mark decoding as completed
            else:
                logger.warning(f"No file data found for post ID {pk}.")
            context['post'] = post
    except Exception as e:
        logger.error(f"Error managing post {pk}: {str(e)}", exc_info=True)
        messages.error(request, 'An error occurred while processing the post.')

    return render(request, 'manage_post.html', context)

@login_required
def add_post(request):
    context = {'page_title': 'Add New Document'}

    # Only render an empty form for adding a new post
    return render(request, 'manage_post.html', context)

@login_required
def save_post(request):
    resp = {'status': 'failed', 'msg': ''}

    if request.method == 'POST':
        user_id = request.user.id
        cache.set(f'encode_progress_{user_id}', 0, timeout=900)  # Initialize progress

        if request.POST.get('id') and not request.POST['id'] == '':
            post = Post.objects.get(id=request.POST['id'])
            form = SavePost(request.POST, request.FILES, instance=post)
        else:
            form = SavePost(request.POST, request.FILES)

        if form.is_valid():
            saved_post = form.save(commit=False)

            if 'file_path' in request.FILES:
                file = request.FILES['file_path']

                # Save the file
                file_path = default_storage.save(f"uploads/{file.name}", ContentFile(file.read()))
                full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)

                # Convert file to binary bitstream
                bitstream = file_to_bitstream(full_file_path)

                # Load the Markov model
                # Use the pre-loaded Markov model
                model = get_markov_model()
                if not model:
                    logger.error("No pre-loaded Markov model found.")
                    resp['msg'] = 'Error: Markov model not loaded.'
                    return HttpResponse(json.dumps(resp), content_type="application/json")
                logger.info("Using pre-loaded Markov model for encoding.")

                # Encode step-by-step
                encoder = Encoder(model, bitstream, logging=True)
                old_progress = 0
                while not encoder.finished:
                    progress = min(5 * ((5 + encoder.step() * 90) // 5), 100)  # Ensure increments of 5%
                    if progress > old_progress:
                        cache.set(f'encode_progress_{user_id}', progress)
                        old_progress = progress
                        logger.info(f"Encoding Progress: {progress}%")

                saved_post.file_data = encoder.output
                cache.set(f'encode_progress_{user_id}', 100, timeout=900)  # Mark encoding as completed
                logger.info("Encoding completed.")

            saved_post.save()
            # Add success message to the response
            resp['status'] = 'success'
            resp['msg'] = 'Document has been saved successfully.'
        else:
            for field in form:
                for error in field.errors:
                    resp['msg'] += str(error) + '<br/>'

    return HttpResponse(json.dumps(resp), content_type="application/json")

@login_required
def progress_status(request):
    action = request.GET.get('action', None)
    user_id = request.user.id
    progress = 0

    if action == 'encode':
        progress = cache.get(f'encode_progress_{user_id}', 0)
    elif action == 'decode':
        progress = cache.get(f'decode_progress_{user_id}', 0)
    else:
        progress = 0

    return JsonResponse({'progress': progress})

@login_required
def delete_post(request):
    resp = {'status':'failed', 'msg':''}
    if request.method == 'POST':
        try:
            post = Post.objects.get(id = request.POST['id'])
            post.delete()
            resp['status'] = 'success'
            messages.success(request, 'Post has been deleted successfully')
        except:
           resp['msg'] = "Undefined Post ID"
    return HttpResponse(json.dumps(resp),content_type="application/json")

def shareF(request,id=None):
    # print(str("b'UdhnfelTxqj3q6BbPe7H86sfQnboSBzb0irm2atoFUw='").encode())
    context['page_title'] = 'Shared File'
    if not id is None:
        key = settings.ID_ENCRYPTION_KEY
        fernet = Fernet(key)
        id = base64.urlsafe_b64decode(id)
        id = fernet.decrypt(id).decode()
        post = Post.objects.get(id = id)
        context['post'] = post
        context['page_title'] += str(" - " + post.title)
   
    return render(request, 'share-file.html',context)

@login_required
def update_profile(request):
    context['page_title'] = 'Update Profile'
    user = User.objects.get(id = request.user.id)
    if not request.method == 'POST':
        form = UpdateProfile(instance=user)
        context['form'] = form
        print(form)
    else:
        form = UpdateProfile(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile has been updated")
            return redirect("profile")
        else:
            context['form'] = form
            
    return render(request, 'manage_profile.html',context)


@login_required
def update_password(request):
    context['page_title'] = "Update Password"
    if request.method == 'POST':
        form = UpdatePasswords(user = request.user, data= request.POST)
        if form.is_valid():
            form.save()
            messages.success(request,"Your Account Password has been updated successfully")
            update_session_auth_hash(request, form.user)
            return redirect("profile")
        else:
            context['form'] = form
    else:
        form = UpdatePasswords(request.POST)
        context['form'] = form
    return render(request,'update_password.html',context)



