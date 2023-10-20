from django.db import models


# Create your models here.
class Students(models.Model):
    name = models.CharField(max_length=10)
    address = models.CharField(max_length=50)
    email = models.CharField(max_length=30)

    def __str__(self):
        return self.name


class Score(models.Model):
    id = models.AutoField(primary_key=True)
    student = models.ForeignKey(Students, on_delete=models.CASCADE, related_name="score_set")
    english = models.IntegerField(default=10)
    math = models.IntegerField(default=10)
    science = models.IntegerField(default=10)
    date = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return self.student.name

