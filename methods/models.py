from django.db import models

class Category(models.Model):
    name = models.CharField(max_length=80, unique=True)
    slug = models.SlugField(max_length=90, unique=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

class Method(models.Model):
    KIND_CHOICES = [
        ("bisection", "Bisección"),
        ("newton", "Newton-Raphson"),
        ("secant", "Secante"),
        ("fixed_point", "Punto Fijo"),
        ("pivot_partial", "Pivoteo Parcial"),
        ("pivot_total", "Pivoteo Total"),
    ]

    name = models.CharField(max_length=120)
    slug = models.SlugField(max_length=130, unique=True)
    description = models.TextField(blank=True)
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, blank=True)
    image = models.ImageField(upload_to="methods/", blank=True, null=True)
    is_featured = models.BooleanField(default=True)

    # 🔽 Campo nuevo para enrutar al ejecutor
    kind = models.CharField(max_length=20, choices=KIND_CHOICES)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name
