# core/urls.py
from django.urls import path
from app import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('signup/', views.signup_view, name='signup'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('generate/', views.generate, name='generate'),
    path('generate-prem/', views.generate, name='generate_prem'),
    path('generate-enterprise/', views.generate, name='generated_enterprise'),
    path('clusters/', views.clusters, name='clusters'),
    path('clusters/<int:cluster_id>/communications/', views.cluster_communications, name='cluster_communications'),
    path('journey/', views.journey_builder, name='journey_builder'),
    path('profile/', views.profile, name='profile'),
    path('generated/<str:filename>', views.generated_file, name='generated_file'),
    path('context/', views.context_view, name='context'),
    path('template_preview/<str:template_name>/', views.template_preview, name='template_preview'),
]
