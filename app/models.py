from django.db import models
from django.contrib.auth.models import User
import os

class UserContext(models.Model):
    """Stores user-specific context in JSON format"""
    CONTEXT_TYPE_CHOICES = [
        ('company', 'Company Information'),
        ('products', 'Products/Services'),
        ('target_audience', 'Target Audience'),
        ('communication_style', 'Communication Style'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, help_text="A descriptive name for this context")
    context_type = models.CharField(
        max_length=50,
        choices=CONTEXT_TYPE_CHOICES,
        default='company',
        help_text="Type of context (determines how it's used in generation)"
    )
    context_data = models.JSONField(
        help_text="Structured JSON data containing the context information"
    )
    is_active = models.BooleanField(
        default=False,
        help_text="Whether this is the currently active context of its type"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-is_active', '-updated_at']
        verbose_name_plural = 'User Contexts'
        unique_together = ['user', 'name']

    def __str__(self):
        return f"{self.name} ({self.get_context_type_display()}) - {self.user.username}"
        
    def save(self, *args, **kwargs):
        # Ensure only one active context per type per user
        if self.is_active:
            UserContext.objects.filter(
                user=self.user, 
                context_type=self.context_type,
                is_active=True
            ).exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)

class GeneratedCommunication(models.Model):
    """Stores generated HTML communications"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    html_content = models.TextField()
    metadata = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class ClusterAnalysis(models.Model):
    """Stores cluster analysis results"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    dataset_name = models.CharField(max_length=255)
    analysis_data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.dataset_name} - {self.created_at.strftime('%Y-%m-%d')}"

def user_upload_path(instance, filename):
    # file will be uploaded to UPLOAD_ROOT/datasets/user_<id>/<filename>
    return os.path.join('datasets', f'user_{instance.user.id}', filename)

class UploadedDataset(models.Model):
    """Stores uploaded datasets"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to=user_upload_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-uploaded_at']
        unique_together = ['user', 'name']
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        # Ensure the upload directory exists
        from django.conf import settings
        os.makedirs(settings.UPLOAD_ROOT, exist_ok=True)
        super().save(*args, **kwargs)