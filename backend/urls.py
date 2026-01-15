from .views import index, inference_api
from django.urls import path, re_path
from django.conf import settings
from django.views.static import serve

urlpatterns = [
    path('api/inference', inference_api),
]

# Servir les fichiers statiques (images, assets JS/CSS)
if len(settings.STATICFILES_DIRS) > 0:
    # Servir les fichiers du répertoire static
    static_dir = settings.STATICFILES_DIRS[0]
    urlpatterns += [
        re_path(r'^static/(?P<path>.*)$', serve, {'document_root': static_dir}),
    ]

    # Servir les assets JS/CSS du frontend
    if len(settings.STATICFILES_DIRS) > 1:
        frontend_static_dir = settings.STATICFILES_DIRS[1]
        urlpatterns += [
            re_path(r'^assets/(?P<path>.*)$', serve, {'document_root': frontend_static_dir / 'assets'}),
        ]

# Catch-all pour servir le frontend React (doit être en dernier)
urlpatterns += [
    re_path(r'^.*$', index),
]
