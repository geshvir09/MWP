# Generated by Django 4.1.7 on 2023-06-01 19:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0019_useranswer'),
    ]

    operations = [
        migrations.CreateModel(
            name='student',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question_No', models.IntegerField()),
                ('question', models.CharField(max_length=255)),
                ('teacher_answer', models.TextField()),
                ('total_mark', models.IntegerField()),
                ('student_answer', models.TextField()),
                ('student_mark', models.IntegerField()),
            ],
        ),
    ]