from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-=cldztbc4jg&xl0!x673!*v2_=p$$eu)=7*f#d0#zs$44xx-h^'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
ALLOWED_HOSTS = ['*']
WSGI_APPLICATION = 'backend.wsgi.application'

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

DATABASES = {}

# Password validation
# https://docs.djangoproject.com/en/4.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Static files (CSS, JavaScript, Images)
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static", BASE_DIR / "frontend" / "dist"]

# CORS - Configuration différente selon la plateforme
import os

# Détecter la plateforme de déploiement
is_vercel = (
    os.environ.get("VERCEL") == "1" 
    or os.environ.get("VERCEL_ENV")
    or "vercel" in os.environ.get("PATH", "").lower()
    or os.environ.get("VERCEL_URL")
)
is_railway = os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RAILWAY_PROJECT_ID")
cors_origin = os.environ.get("CORS_ALLOWED_ORIGIN", "")

# Debug: afficher la détection (à retirer en prod si besoin)
print(f"[CORS DEBUG] is_vercel={is_vercel}, is_railway={is_railway}, cors_origin={cors_origin}")

if is_vercel:
    # Sur Vercel : CORS ouvert pour les extensions Chrome
    CORS_ALLOW_ALL_ORIGINS = True
    CORS_ALLOW_CREDENTIALS = True
    print("[CORS] Configuration Vercel: CORS_ALLOW_ALL_ORIGINS = True")
elif is_railway:
    # Sur Railway : CORS restrictif (seulement les origines autorisées)
    CORS_ALLOWED_ORIGINS = [
        "http://localhost:8080",
        "http://localhost:5173",
        "https://fakefinder.vercel.app",  # Frontend Vercel
    ]
    if cors_origin and cors_origin != "*":
        # Ajouter les origines spécifiques depuis la variable d'env
        origins = [origin.strip() for origin in cors_origin.split(",")]
        CORS_ALLOWED_ORIGINS.extend(origins)
else:
    # Local : CORS pour développement
    CORS_ALLOWED_ORIGINS = [
        "http://localhost:8080",
        "http://localhost:5173",
    ]
    if cors_origin and cors_origin != "*":
        CORS_ALLOWED_ORIGINS.append(cors_origin)

# Autoriser les headers nécessaires pour les extensions Chrome
CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

# Internationalization
# https://docs.djangoproject.com/en/4.1/topics/i18n/

LANGUAGE_CODE = 'fr-fr'
TIME_ZONE = 'Europe/Paris'
USE_I18N = True
USE_TZ = True

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'