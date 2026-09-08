# workloads/signals.py

from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth import get_user_model
from .models import UserProfile

User = get_user_model()

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance, role=instance.role)
    else:
        # Sync profile role whenever User object is updated
        profile, _ = UserProfile.objects.get_or_create(user=instance)
        if profile.role != instance.role:
            profile.role = instance.role
            profile.save()