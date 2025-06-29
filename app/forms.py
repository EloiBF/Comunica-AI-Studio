# myapp/forms.py
from django import forms

class PromptForm(forms.Form):
    prompt = forms.CharField(widget=forms.Textarea(attrs={'rows': 5, 'placeholder': 'Introdueix el prompt...'}), label='Prompt')
    template = forms.ChoiceField(choices=[('sale.html', 'Sale Template')], label='Plantilla')
