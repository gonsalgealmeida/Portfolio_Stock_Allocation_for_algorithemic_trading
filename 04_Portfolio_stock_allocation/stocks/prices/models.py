from django.db import models

# Create your models here.


class Stock(models.Model):
    ticker = models.CharField(max_length=10)
    price = models.FloatField()
    date = models.DateField()
