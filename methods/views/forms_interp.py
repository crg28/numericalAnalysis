from django import forms

class SplineBaseForm(forms.Form):
    x_values = forms.CharField(
        label="X values",
        help_text="Separados por coma. Ej: -1, 0, 3, 4",
        widget=forms.TextInput(attrs={"class": "input"})
    )

    y_values = forms.CharField(
        label="Y values",
        help_text="Debe haber la misma cantidad que X",
        widget=forms.TextInput(attrs={"class": "input"})
    )

    # Punto a evaluar (opcional)
    x_point = forms.FloatField(
        label="Evaluar en X (opcional)",
        required=False,
        widget=forms.NumberInput(attrs={"class": "input"})
    )

    def clean(self):
        cleaned_data = super().clean()
        x_raw = cleaned_data.get("x_values")
        y_raw = cleaned_data.get("y_values")

        try:
            x_list = [float(v.strip()) for v in x_raw.split(",")]
            y_list = [float(v.strip()) for v in y_raw.split(",")]
        except Exception:
            raise forms.ValidationError("X y Y deben ser números separados por coma.")

        if len(x_list) != len(y_list):
            raise forms.ValidationError("La cantidad de valores X y Y debe coincidir.")

        # Guardar procesados
        cleaned_data["x_list"] = x_list
        cleaned_data["y_list"] = y_list

        return cleaned_data
    


class LagrangeForm(forms.Form):
    x_values = forms.CharField(
        label="Valores de X",
        required=True,
        widget=forms.TextInput(attrs={
            "placeholder": "Ej: 1, 2, 3, 4"
        })
    )

    y_values = forms.CharField(
        label="Valores de Y",
        required=True,
        widget=forms.TextInput(attrs={
            "placeholder": "Ej: 2.5, 4.1, 6.0, 8.2"
        })
    )

    x_eval = forms.FloatField(
        label="Punto a evaluar (opcional)",
        required=False,
        widget=forms.NumberInput(attrs={
            "placeholder": "Ej: 2.5"
        })
    )

    def clean_x_values(self):
        text = self.cleaned_data["x_values"]
        try:
            return [float(v.strip()) for v in text.split(",")]
        except:
            raise forms.ValidationError("Formato inválido en X. Use números separados por coma.")

    def clean_y_values(self):
        text = self.cleaned_data["y_values"]
        try:
            return [float(v.strip()) for v in text.split(",")]
        except:
            raise forms.ValidationError("Formato inválido en Y. Use números separados por coma.")