from django import forms
from .models import DataSet, DataColumn, DataPoint


class DataSetForm(forms.ModelForm):
    columns = forms.MultipleChoiceField(choices=[], widget=forms.CheckboxSelectMultiple)

    class Meta:
        model = DataSet
        fields = ['name', 'description', 'json_file']

    def __init__(self, *args, **kwargs):
        columns_choices = kwargs.pop('columns_choices', [])
        super().__init__(*args, **kwargs)
        self.fields['columns'].choices = [(col, col) for col in columns_choices]


class DataColumnForm(forms.ModelForm):
    class Meta:
        model = DataColumn
        fields = ['name', 'data_type', 'data_set']


class DataPointForm(forms.ModelForm):
    class Meta:
        model = DataPoint
        fields = ['column', 'value']
