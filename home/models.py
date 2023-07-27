from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    is_teacher = models.BooleanField('Is teacher', default=False)
    is_student = models.BooleanField('Is student', default=True)

class FData(models.Model):
    question = models.TextField()
    answer = models.TextField()
    total_mark = models.IntegerField()
    teacher_clusters = models.TextField(blank=True, null=True)
    objects = models.Manager()


class scoreOne(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    question = models.CharField(max_length=5000)
    teacher_answer = models.TextField()
    teacher_mark = models.IntegerField()
    student_answer = models.TextField()
    student_mark = models.IntegerField()   
    objects = models.Manager()


#########################################################################################################
class ClusterDB(models.Model):
    text = models.TextField()
    mark = models.IntegerField()

    objects = models.Manager()

class FormData(models.Model):
    question = models.CharField(max_length=255)
    answer = models.TextField()
    total_mark = models.IntegerField()
    clusters = models.ManyToManyField(ClusterDB)
    objects = models.Manager()

class stu_table(models.Model):
    question = models.CharField(max_length=255)
    student_answer = models.TextField()
    student_mark = models.IntegerField()   
    objects = models.Manager()

class studentTable(models.Model):
    question_No  = models.IntegerField()
    question = models.CharField(max_length=255)
    teacher_answer = models.TextField()
    total_mark = models.IntegerField()
    student_answer = models.TextField()
    student_mark = models.IntegerField()   
    objects = models.Manager()

class student(models.Model):
    question_No  = models.IntegerField()
    question = models.CharField(max_length=255)
    teacher_answer = models.TextField()
    total_mark = models.IntegerField()
    student_answer = models.TextField()
    student_mark = models.IntegerField()
    
    objects = models.Manager()

class UserAnswer(models.Model):
    question = models.CharField(max_length=255)
    answer = models.TextField()

class Questions(models.Model):
    question = models.TextField()
    answer = models.TextField()
    mark = models.TextField()
    
    objects = models.Manager()
        
class Cluster(models.Model):
    clusterNo = models.CharField(max_length=255)
    clusterLines = models.TextField()
    cluster_mark = models.CharField(max_length=255)
    
    objects = models.Manager()


class score(models.Model):
    question = models.CharField(max_length=255)
    teacher_answer = models.TextField()
    teacher_mark = models.IntegerField()
    student_answer = models.TextField()
    student_mark = models.IntegerField()   
    objects = models.Manager()


class ClusterData(models.Model):
    cluster_number = models.AutoField(primary_key=True)
    cluster_line = models.TextField()
    cluster_mark = models.TextField()
    question = models.ForeignKey(Questions, on_delete=models.CASCADE)
    
    objects = models.Manager()
    
    def __str__(self):
        return f'Cluster {self.cluster_number}'

class Table_Question(models.Model):
    question = models.TextField()
    answer = models.TextField()
    mark = models.TextField()
    

class Question_table(models.Model):
    question = models.TextField()
    answer = models.TextField()

    objects = models.Manager() 

class PaperDetails_table(models.Model):
    institution_Name=models.TextField()
    subject_Name=models.TextField()
    level_of_study=models.TextField()
    date=models.DateField()

    objects = models.Manager() 


class QuestionTable(models.Model):
    question = models.TextField()
    answer = models.TextField()
    mark = models.TextField()

    objects = models.Manager() 