# Generated by Django 4.1.7 on 2023-06-01 21:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0021_studenttable'),
    ]

    operations = [
        migrations.CreateModel(
            name='stu_table',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question', models.CharField(max_length=255)),
                ('student_answer', models.TextField()),
                ('student_mark', models.IntegerField()),
            ],
        ),
    ]
