# Generated by Django 4.1.7 on 2023-06-02 10:05

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0023_score'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='formdata',
            name='clusters',
        ),
    ]
