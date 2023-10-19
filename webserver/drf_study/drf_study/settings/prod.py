from .common import *

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-&8=6hs2c^9)17(rv#14c%ufurjvh8@f+0rbgix4)uo9bzxn_ow"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}
