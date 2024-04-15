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
import threading
from threading import Semaphore
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import time

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

    max_threads = 5  # Número máximo de hilos permitidos

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semaphore = Semaphore(value=self.max_threads)

    def process_data_points(self, data_points):
        try:
            with transaction.atomic():
                DataPoint.objects.bulk_create(data_points)
        except Exception as e:
            print(f"Error al guardar los puntos de datos: {e}")
        finally:
            self.semaphore.release()

    def form_valid(self, form):
        form.instance.uploaded_by = self.request.user
        dataset = form.save(commit=False)
        dataset.save()
        json_file = self.request.FILES['json_file']
        threads = []

        try:
            json_file.seek(0)
            json_data = json.load(json_file)
        except json.JSONDecodeError as e:
            return HttpResponseBadRequest("Error al cargar el JSON: {}".format(e))

        if 'columns' in self.request.POST:
            selected_columns = self.request.POST.getlist('columns')
        else:
            return self.form_invalid(form)

        try:
            key = next(iter(json_data.keys()))
            df = pd.DataFrame(json_data[key])
        except json.JSONDecodeError:
            return HttpResponseBadRequest("El archivo no es un JSON válido.")

        for column_name in selected_columns:
            column_data = df[column_name]
            data_column_form = DataColumnForm({
                'name': column_name,
                'data_type': column_data.dtype.name,
                'data_set': dataset
            })
            if data_column_form.is_valid():
                data_column = data_column_form.save(commit=False)
                data_column.save()

                # Procesar los puntos de datos en lotes
                data_points = []
                for value in column_data.tolist():
                    data_points.append(DataPoint(column=data_column, value=value))

                # Adquirir el semáforo antes de iniciar un hilo
                self.semaphore.acquire()
                thread = threading.Thread(target=self.process_data_points, args=(data_points,))
                thread.start()
                threads.append(thread)

        # Esperar a que todos los hilos terminen antes de continuar
        for thread in threads:
            thread.join()

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

    def process_data_point(self, data_point, selected_columns, data):
        column_name = data_point.column.name
        if column_name in selected_columns:
            data[column_name].append(data_point.value)

    def post(self, request, *args, **kwargs):
        dataset = self.get_object()
        print("Datos POST recibidos:", request.POST)
        selected_columns = request.POST.getlist('columnas[]')  # Obtener las selecciones de los checkboxes
        print("Columnas seleccionadas:", selected_columns)

        if len(selected_columns) < 2:
            # Si no hay suficientes columnas seleccionadas, devolver un error
            return JsonResponse({'error': 'Debes seleccionar al menos dos columnas.'}, status=400)

        # Obtener los datos del conjunto de datos
        relevant_columns = DataColumn.objects.filter(name__in=selected_columns, data_set=dataset)
        data_points = DataPoint.objects.filter(column__in=relevant_columns).prefetch_related('column')

        # Crear un diccionario para almacenar los datos de las columnas seleccionadas
        data = {column_name: [] for column_name in selected_columns}

        start_time = time.time()

        # Procesar los datos en paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
            futures = []
            for data_point in data_points:
                futures.append(executor.submit(self.process_data_point, data_point, selected_columns, data))
            concurrent.futures.wait(futures)

        duration = time.time() - start_time
        print(duration)
        df = pd.DataFrame(data)

        # Crear un DataFrame de pandas con los datos recolectados
        df = pd.DataFrame(data)

        # Convertir las columnas seleccionadas a tipos numéricos
        df[selected_columns] = df[selected_columns].apply(pd.to_numeric, errors='coerce')

        # Calcular la media
        grouped_data = df.groupby(selected_columns[0]).mean()

        # Generar la gráfica
        plt.plot(grouped_data.index, grouped_data[selected_columns[1]])
        plt.xlabel(selected_columns[0])
        plt.ylabel(selected_columns[1])
        plt.title('Gráfica')

        # Guardar la gráfica como una imagen
        imagen_ruta = os.path.join(settings.MEDIA_ROOT, 'grafica.png')
        plt.savefig(imagen_ruta)
        plt.close()  # Cerrar la figura para liberar recursos

        # Devolver la ruta de la imagen como parte de la respuesta JSON
        return JsonResponse({'imagen_ruta': '/media/grafica.png'})
