# Generated by Django 4.1.7 on 2023-04-10 10:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0006_question_table_delete_questions'),
    ]

    operations = [
        migrations.CreateModel(
            name='PaperDetails_table',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('institution_Name', models.TextField()),
                ('subject_Name', models.TextField()),
                ('level_of_study', models.TextField()),
                ('date', models.DateField()),
            ],
        ),
    ]
