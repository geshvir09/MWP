# Generated by Django 4.1.7 on 2023-06-02 10:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0024_remove_formdata_clusters'),
    ]

    operations = [
        migrations.AddField(
            model_name='formdata',
            name='clusters',
            field=models.ManyToManyField(to='home.clusterdb'),
        ),
    ]
