from .common import *

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "adflaksjdflksdjflksjdfalksdfjsldkfjalskdfjlasdkfjlskdfj"


DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": "django_myboard",
        "USER": "root",
        "PASSWORD": "0000",
        "HOST": "127.0.0.1",
        "PORT": "3306",
    }
}

