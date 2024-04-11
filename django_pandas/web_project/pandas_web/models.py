from django.db import models
from django.contrib.auth.models import User
from django.contrib import admin


class DataSet(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    json_file = models.FileField(upload_to='json_file/')
    upload_date = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)


class DataColumn(models.Model):
    name = models.CharField(max_length=100)
    data_type = models.CharField(max_length=50)
    data_set = models.ForeignKey(DataSet, on_delete = models.CASCADE)


class DataPoint(models.Model):
    column = models.ForeignKey(DataColumn, on_delete=models.CASCADE)
    value = models.CharField(max_length=500)


class AnalysisResult(models.Model):
    ANALYSIS_TYPES = [
        ('1', 'Descripción estadística'),
        ('2', 'Visualización de datos'),
        ('3', 'Análisis específico'),
    ]

    analysis_type = models.CharField(max_length=1, choices=ANALYSIS_TYPES)
    parameters = models.JSONField()
    result_data = models.TextField()
    analyzed_by = models.ForeignKey(User, on_delete=models.CASCADE)
    analysis_date = models.DateTimeField(auto_now_add=True)
