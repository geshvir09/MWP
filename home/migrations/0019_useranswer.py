# Generated by Django 4.1.7 on 2023-05-29 20:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0018_clusterdb_alter_cluster_clusterno_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserAnswer',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question', models.CharField(max_length=255)),
                ('answer', models.TextField()),
            ],
        ),
    ]
