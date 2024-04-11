from django.shortcuts import render, redirect
from django.views.generic import *
from django.urls import reverse_lazy
import pandas as pd
from jupyterlab_server import translator

from .models import DataSet, DataColumn, DataPoint
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db import transaction
from .forms import DataSetForm, DataColumnForm, DataPointForm
import json
from django.http import JsonResponse, HttpResponseBadRequest
import matplotlib.pyplot as plt
import os
from django.conf import settings

class DataSetListView(ListView):
    model = DataSet
    template_name = 'pandas_web/dataset_list.html'
    context_object_name = 'datasets'

    def get_queryset(self):
        return DataSet.objects.all()


class DataSetCreateView(LoginRequiredMixin, CreateView):
    model = DataSet
    template_name = 'pandas_web/dataset_form.html'
    form_class = DataSetForm
    success_url = reverse_lazy('data_sets')

    def dispatch(self, request, *args, **kwargs):
        if 'json_file' in request.FILES:
            file = request.FILES['json_file']
            try:
                data = json.load(file)
                key = next(iter(data.values()), [])
                self.columns_choices = list(key[0].keys())
            except json.JSONDecodeError:
                pass

        return super().dispatch(request, *args, **kwargs)

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        if hasattr(self, 'columns_choices'):
            kwargs['columns_choices'] = self.columns_choices
            print(self.columns_choices)
        return kwargs

    def post(self, request, *args, **kwargs):
        if 'columns' not in self.request.POST:
            if hasattr(self, 'columns_choices'):
                return JsonResponse({'columns_choices': self.columns_choices})
            else:
                return HttpResponseBadRequest("No se encontraron columnas disponibles.")
        else:
            return super().post(request, *args, **kwargs)

    @transaction.atomic
    def form_valid(self, form):
        form.instance.uploaded_by = self.request.user
        dataset = form.save(commit=False)
        dataset.save()
        json_file = self.request.FILES['json_file']
        print("Nombre del archivo enviado:", json_file.name,"xxxx")
        try:
            json_file.seek(0)
            json_data = json.load(json_file)
        except json.JSONDecodeError as e:
            print (e)
            return HttpResponseBadRequest("Error al cargar el JSON: {}".format(e))

        if 'columns' in self.request.POST:
            print("primero:")
            selected_columns = self.request.POST.getlist('columns')
            print("imprimo:", selected_columns)
        else:
            print("La clave 'columns' no está presente en request.POST")
            return self.form_invalid(form)

        try:
            key = next(iter(json_data.keys()))
            df = pd.DataFrame(json_data[key])
        except json.JSONDecodeError:
            return HttpResponseBadRequest("El archivo no es un JSON válido.")

        for column_name in selected_columns:
            column_data = df[column_name]
            print(column_name, column_data.dtype.name)
            data_column_form = DataColumnForm({
                'name': column_name,
                'data_type': column_data.dtype.name,
                'data_set': dataset
            })
            if data_column_form.is_valid():
                data_column = data_column_form.save(commit=False)
                data_column.save()
            else:
                print("¡Error al guardar la columna {}!".format(column_name))

            for value in column_data.tolist():
                data_point_form = DataPointForm({
                    'column': data_column,
                    'value': value
                })
                if data_point_form.is_valid():
                    data_point = data_point_form.save(commit=False)
                    data_point.save()
                else:
                    print("¡Error al guardar el dato en la columna {}!".format(column_name))

        return super().form_valid(form)

class DataSetDetailView(DetailView):
    model = DataSet
    template_name = 'pandas_web/dataset_detail.html'
    context_object_name = 'dataset'

    def get(self, request, pk):
        dataset = get_object_or_404(DataSet, pk=pk)
        return render(request, 'pandas_web/dataset_detail.html', {'dataset': dataset})


class DataSetUpdateView(UpdateView):
    model = DataSet
    template_name = 'pandas_web/dataset_form.html'
    form_class = DataSetForm

    def form_valid(self, form):
        form.instance.uploaded_by = self.request.user
        return super().form_valid(form)


class DataSetDeleteView(DeleteView):
    model = DataSet
    success_url = reverse_lazy('data_sets')
    template_name = 'pandas_web/dataset_confirm_delete.html'

    def post(self, request, pk):
        dataset = get_object_or_404(DataSet, pk=pk)
        dataset.delete()
        return redirect('data_sets')


def show_json(request, dataset_id):
    dataset = DataSet.objects.get(pk=dataset_id)
    with open(dataset.json_file.path, 'r') as file:
        json_data = json.load(file)
    return render(request, 'show_json.html', {'dataset': dataset, 'json_data': json_data})


class AnalyzeDataView(DetailView):
    model = DataSet
    template_name = 'pandas_web/dataset_analyze.html'
    context_object_name = 'dataset'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        dataset = self.get_object()
        context['columns'] = dataset.datacolumn_set.all()
        return context

    def post(self, request, *args, **kwargs):
        dataset = self.get_object()
        selected_columns = request.POST.getlist('columnas[]')  # Obtener las selecciones de los checkboxes

        # Aquí puedes realizar el procesamiento necesario de las selecciones de las columnas
        # y generar la gráfica con Matplotlib
        # Por ahora, solo se genera una gráfica de ejemplo
        plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
        plt.xlabel('Eje X')
        plt.ylabel('Eje Y')
        plt.title('Ejemplo de Gráfica')

        # Guardar la gráfica como una imagen
        imagen_ruta = os.path.join(settings.MEDIA_ROOT, 'grafica.png')
        plt.savefig(imagen_ruta)
        plt.close()  # Cerrar la figura para liberar recursos

        # Devolver la ruta de la imagen como parte de la respuesta JSON
        return JsonResponse({'imagen_ruta': '/media/grafica.png'})
