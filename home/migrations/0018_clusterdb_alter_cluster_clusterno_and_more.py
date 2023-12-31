# Generated by Django 4.1.7 on 2023-05-29 09:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0017_cluster'),
    ]

    operations = [
        migrations.CreateModel(
            name='ClusterDB',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField()),
                ('mark', models.IntegerField()),
            ],
        ),
        migrations.AlterField(
            model_name='cluster',
            name='clusterNo',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='cluster',
            name='cluster_mark',
            field=models.CharField(max_length=255),
        ),
        migrations.CreateModel(
            name='FormData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question', models.CharField(max_length=255)),
                ('answer', models.TextField()),
                ('total_mark', models.IntegerField()),
                ('clusters', models.ManyToManyField(to='home.clusterdb')),
            ],
        ),
    ]
