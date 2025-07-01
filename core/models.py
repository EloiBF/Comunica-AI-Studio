from django.contrib.auth.models import User
from django.db import models

class UserProfile(models.Model):
    SUBSCRIPTION_CHOICES = [
        ('free', 'Free'),
        ('premium', 'Premium'),
        ('enterprise', 'Enterprise'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    subscription = models.CharField(max_length=20, choices=SUBSCRIPTION_CHOICES, default='free')

    def __str__(self):
        return f"{self.user.username} ({self.subscription})"