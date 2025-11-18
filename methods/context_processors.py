# methods/context_processors.py
from .models import Category  # <-- usa el nombre real de tu modelo

def method_categories(request):
    """
    Provides 'categories' for the header dropdown in base.html.
    Available in every template.
    """
    try:
        categories = Category.objects.all()
    except Exception:
        categories = []
    return {"categories": categories}
