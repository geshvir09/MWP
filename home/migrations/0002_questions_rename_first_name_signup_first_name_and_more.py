# Generated by Django 4.1.7 on 2023-04-06 08:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Questions',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question_Number', models.IntegerField()),
                ('question', models.TextField()),
                ('answer', models.TextField()),
            ],
            options={
                'db_table': 'Questions_tbl',
            },
        ),
        migrations.RenameField(
            model_name='signup',
            old_name='first_name',
            new_name='first_Name',
        ),
        migrations.RenameField(
            model_name='signup',
            old_name='last_name',
            new_name='last_Name',
        ),
    ]
