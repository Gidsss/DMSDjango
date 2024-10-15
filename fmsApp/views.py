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
    context = {'page_title': 'Manage Post', 'post': {}}

    if pk is not None:
        post = Post.objects.get(id=pk)

        # Decode the binary data from base64, if the file data exists
        if post.file_data:
            # Load the Markov model (pre-built model)
            model = build_model("markov_models/legal_corpus.json")
            logger.info("Markov model loaded successfully for decoding.")

            # Decode the file_data using the Decoder class
            decoder = Decoder(model, post.file_data, logging=True)
            decoder.solve()

            # The output bitstream will be the decoded binary data
            decoded_file_data = decoder.output  # Ensure correct decoding
            context['decoded_file_data'] = decoded_file_data  # Pass the decoded binary data to the template

        context['post'] = post

    return render(request, 'manage_post.html', context)

@login_required
def view_post(request, pk):
    try:
        # Fetch the post by its primary key (id)
        logger.info(f"Fetching post with ID: {pk}")
        post = get_object_or_404(Post, pk=pk)

        # Check if post has file data
        if post.file_data:
            logger.info(f"Post {pk} contains file_data. Decoding process started.")

            # Load the Markov model
            model = build_model("markov_models/legal_corpus.json")
            logger.info("Markov model loaded successfully.")

            # Decoder class to decode the steganographic text to binary bitstream
            decoder = Decoder(model, post.file_data, logging=True)
            logger.info(f"Decoder initialized for post {pk}. Starting to solve...")

            decoder.solve()
            logger.info(f"Decoding completed for post {pk} using Decoder class.")

            decoded_bitstream = decoder.output  # This will be the decoded bitstream

            # Optionally, save the decoded file or preview
            decoded_file_path = f"{post.title}_decoded.pdf"
            bitstream_to_file(decoded_bitstream, decoded_file_path)
            logger.info(f"Decoded bitstream saved as {decoded_file_path}")

            # Return the decoded data for preview
            response_data = {
                'status': 'success',
                'data': {
                    'file_name': post.title,
                    'binary_data': decoded_bitstream,  
                    'description': post.description
                }
            }
        else:
            logger.warning(f"Post {pk} does not contain file_data.")
            response_data = {'status': 'failed', 'msg': 'No file data available'}

    except Exception as e:
        logger.error(f"Error while processing post {pk}: {str(e)}", exc_info=True)
        response_data = {
            'status': 'failed',
            'msg': str(e)
        }

    return JsonResponse(response_data)

@login_required
def save_post(request):
    resp = {'status': 'failed', 'msg': ''}

    if request.method == 'POST':
        start_time = time.time()  # Start time to measure process duration
        if request.POST.get('id') and not request.POST['id'] == '':
            post = Post.objects.get(id=request.POST['id'])
            form = SavePost(request.POST, request.FILES, instance=post)
        else:
            form = SavePost(request.POST, request.FILES)

        if form.is_valid():
            saved_post = form.save(commit=False)

            if 'file_path' in request.FILES:
                file = request.FILES['file_path']

                # Save the file to the correct directory using default_storage
                file_path = default_storage.save(f"uploads/{file.name}", ContentFile(file.read()))
                full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)

                # Log the file save duration
                logger.info(f"File saved at: {full_file_path}")
                logger.info(f"Time taken to save file: {time.time() - start_time} seconds")

                # Convert file to binary bitstream
                bitstream_start = time.time()  # Start time for bitstream conversion
                bitstream = file_to_bitstream(full_file_path)

                # Log the bitstream conversion duration
                logger.info(f"Time taken to convert file to bitstream: {time.time() - bitstream_start} seconds")

                # Load the Markov model (pre-built model)
                model = build_model("markov_models/legal_corpus.json")
                logger.info("Markov model loaded successfully.")

                # Encode the file to steganographic text
                encode_start = time.time()  # Start time for encoding
                encoder = Encoder(model, bitstream, logging=True)
                encoder.generate()

                # Ensure that the end key was injected
                if encoder.finished:
                    logger.info(f"End key was injected successfully.")
                else:
                    logger.warning(f"End key was not injected.")

                # Log the encoding duration
                logger.info(f"Time taken to encode bitstream: {time.time() - encode_start} seconds")

                stega_text = encoder.output
                saved_post.file_data = stega_text

            # Save the post
            saved_post.save()

            # Log the total duration for the process
            logger.info(f"Total time taken to process file: {time.time() - start_time} seconds")

            messages.success(request, 'File has been saved successfully.')
            resp['status'] = 'success'
        else:
            for field in form:
                for error in field.errors:
                    resp['msg'] += str(error) + '<br/>'

    else:
        resp['msg'] = "No Data sent."

    return HttpResponse(json.dumps(resp), content_type="application/json")

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



