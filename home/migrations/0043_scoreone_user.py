# Generated by Django 4.1.7 on 2023-07-18 17:15

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0042_remove_scoreone_user'),
    ]

    operations = [
        migrations.AddField(
            model_name='scoreone',
            name='user',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]