from django import forms

# ==============================
# NONLINEAR METHODS
# ==============================
NONLINEAR_CHOICES = [
    ('bisection', 'Bisection'),
    ('newton', 'Newton-Raphson'),
    ('secant', 'Secant'),
    ('fixed_point', 'Fixed Point'),
]

class NonlinearForm(forms.Form):
    method = forms.ChoiceField(choices=NONLINEAR_CHOICES, label='Method')
    function = forms.CharField(
        label='f(x)',
        help_text="Example: sin(x) - x/2",
        widget=forms.TextInput(attrs={'placeholder': 'sin(x) - x/2'})
    )
    g_function = forms.CharField(
        label='g(x) (only for Fixed Point)',
        required=False,
        widget=forms.TextInput(attrs={'placeholder': 'Example: (sin(x)+x)/2'})
    )
    x0 = forms.FloatField(label='x₀')
    x1 = forms.FloatField(label='x₁ (only for Secant and Bisection)', required=False)
    a = forms.FloatField(label='a (only for Bisection)', required=False)
    b = forms.FloatField(label='b (only for Bisection)', required=False)
    tol = forms.FloatField(label='Tolerance', initial=1e-6)
    max_iter = forms.IntegerField(label='Max iterations', initial=100)

# ==============================
# LINEAR SYSTEM METHODS
# ==============================
LINEAR_CHOICES = [
    ('partial', 'Partial Pivoting'),
    ('total', 'Total Pivoting'),
]

class LinearSystemForm(forms.Form):
    method = forms.ChoiceField(choices=LINEAR_CHOICES, label='Method')
    A = forms.CharField(
        label='Matrix A (rows separated by ; and columns by ,)',
        widget=forms.Textarea(attrs={'rows': 4, 'placeholder': '1,2,3; 4,5,6; 7,8,10'})
    )
    b = forms.CharField(
        label='Vector b (comma-separated)',
        widget=forms.TextInput(attrs={'placeholder': '1, 0, 3'})
    )

# ==============================
# USER SIGNUP FORM
# ==============================
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")
