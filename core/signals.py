# signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import UserProfile

@receiver(post_save, sender=User)
def create_or_save_user_profile(sender, instance, created, **kwargs):
    # Si l'usuari és creat, creem el perfil
    if created:
        UserProfile.objects.create(user=instance)
    else:
        # Si l'usuari ja existeix, actualitzem el perfil (si existeix)
        if not hasattr(instance, 'profile'):
            UserProfile.objects.create(user=instance)
        else:
            instance.profile.save()  # Desar el perfil existent
