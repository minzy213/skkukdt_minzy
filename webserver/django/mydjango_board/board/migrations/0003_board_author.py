# Generated by Django 4.1 on 2023-09-26 07:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('board', '0002_comment'),
    ]

    operations = [
        migrations.AddField(
            model_name='board',
            name='author',
            field=models.CharField(default='', max_length=255),
            preserve_default=False,
        ),
    ]
