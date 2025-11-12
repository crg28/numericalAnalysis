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
        # Raíces
        ("bisection", "Bisection"),
        ("newton", "Newton-Raphson"),
        ("secant", "Secant"),
        ("fixed_point", "Fixed Point"),
        ("false_position", "False Position"),
        ("incremental_search", "Incremental Search"),
        ("multiple_roots", "Multiple Roots"),
        # Sistemas lineales
        ("pivot_partial", "Partial Pivoting"),
        ("pivot_total", "Total Pivoting"),
        ("gaussian_elimination", "Gaussian Elimination"),
        ("jacobi", "Jacobi"),
        ("gauss_seidel", "Gauss-Seidel"),
        ("sor", "SOR"),
        ("crout", "Crout Factorization"),
        ("doolittle", "Doolittle Factorization"),
        ("cholesky", "Cholesky Factorization"),
        ("lu_simple", "Simple LU Factorization"),
        ("lu_pivot", "Pivoting LU Factorization"),
        # Interpolación
        ("lagrange", "Lagrange Interpolation"),
        ("vandermonde", "Vandermonde Interpolation"),
        ("newton_interpolation", "Newton Interpolation"),
        ("spline_linear", "Linear Spline"),
        ("spline_quadratic", "Quadratic Spline"),
        ("spline_cubic", "Cubic Spline"),
    ]


    name = models.CharField(max_length=120)
    slug = models.SlugField(max_length=130, unique=True)
    description = models.TextField(blank=True)
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, blank=True)
    image = models.ImageField(upload_to="methods/", blank=True, null=True)
    is_featured = models.BooleanField(default=True)

    # Para enrutar al ejecutor
    kind = models.CharField(max_length=30, choices=KIND_CHOICES)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name
