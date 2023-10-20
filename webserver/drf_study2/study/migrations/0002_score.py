# Generated by Django 4.1 on 2023-10-20 01:01

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('study', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Score',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('english', models.IntegerField(default=10)),
                ('math', models.IntegerField(default=10)),
                ('science', models.IntegerField(default=10)),
                ('student', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='score_set', to='study.students')),
            ],
        ),
    ]