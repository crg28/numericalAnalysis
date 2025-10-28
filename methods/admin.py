from django.contrib import admin
from .models import Category, Method

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ("name", "slug")
    prepopulated_fields = {"slug": ("name",)}

@admin.register(Method)
class MethodAdmin(admin.ModelAdmin):
    list_display = ("name", "category", "is_featured")
    list_filter = ("category", "is_featured")
    search_fields = ("name", "description")
    prepopulated_fields = {"slug": ("name",)}
