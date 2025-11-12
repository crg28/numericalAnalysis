from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

# ======== AUTH ========
class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)
    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")


# ======== LINEALES: A y b ========
class AbForm(forms.Form):
    """Para: cholesky, crout, doolittle, lu_simple, lu_pivot,
    gaussian_elimination, pivot_partial, pivot_total
    """
    A = forms.CharField(
        label='Matrix A',
        widget=forms.Textarea(attrs={
            'rows': 6,
            'placeholder': 'Ejemplos:\n1 2 3\n4 5 6\n7 8 10\n\nó\n1,2,3; 4,5,6; 7,8,10\nó\n[[1,2],[3,4]]'
        })
    )
    b = forms.CharField(
        label='Vector b',
        widget=forms.Textarea(attrs={
            'rows': 3,
            'placeholder': 'Ejemplos:\n1\n0\n3\n\nó\n1,0,3\nó\n[1,0,3]'
        })
    )


# ======== ITERATIVOS ========
NORM_CHOICES = [('inf', 'Infinity norm (max |·|)'), ('2', 'Euclidean norm (L2)')]

class IterativeForm(forms.Form):
    """Jacobi / Gauss-Seidel."""
    A = forms.CharField(
        label='Matrix A',
        widget=forms.Textarea(attrs={'rows': 6, 'placeholder': '1 2\n3 4'})
    )
    b = forms.CharField(
        label='Vector b',
        widget=forms.Textarea(attrs={'rows': 3, 'placeholder': '1\n2'})
    )
    x0 = forms.CharField(
        label='Initial guess x0',
        required=False,
        widget=forms.Textarea(attrs={'rows': 3, 'placeholder': '0\n0'})
    )
    tol = forms.FloatField(label='Tolerance', initial=1e-6)
    max_iter = forms.IntegerField(label='Max iterations', initial=50)
    norm = forms.ChoiceField(label='Norm', choices=NORM_CHOICES, initial='inf')

class SorForm(IterativeForm):
    """SOR = IterativeForm + w."""
    w = forms.FloatField(label='ω (relaxation factor)', initial=1.0)
