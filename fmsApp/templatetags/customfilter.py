from atexit import register
from django import template
from cryptography.fernet import Fernet
from django.conf import settings
import base64


register = template.Library()

@register.filter
def replaceBlank(value,stringVal = ""):
    value = str(value).replace(stringVal, '')
    return value

@register.filter
def encryptdata(value):
    fernet = Fernet(settings.ID_ENCRYPTION_KEY)
    value = fernet.encrypt(str(value).encode())
    return value

@register.filter
def decode_binary(value):
    """
    Decode base64-encoded binary data to a UTF-8 string.
    In case of decoding errors, the original value is returned.
    """
    try:
        # Decode the base64-encoded value to binary, then to a UTF-8 string
        return base64.b64decode(value).decode('utf-8', errors='replace')
    except Exception as e:
        # If an error occurs, return the original value
        return value
