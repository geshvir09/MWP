# Generated by Django 4.1.7 on 2023-05-24 11:25

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0012_question'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='question',
            new_name='QuestionTable',
        ),
    ]
