# Generated by Django 4.1.7 on 2023-05-24 11:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0013_rename_question_questiontable'),
    ]

    operations = [
        migrations.CreateModel(
            name='Table_Question',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question', models.TextField()),
                ('answer', models.TextField()),
                ('mark', models.TextField()),
            ],
        ),
    ]
