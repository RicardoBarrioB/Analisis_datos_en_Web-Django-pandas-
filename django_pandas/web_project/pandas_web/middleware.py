import logging
import time
from django.db import connection

class QueryCountMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.time()
        response = self.get_response(request)
        duration = time.time() - start_time
        query_count = len(connection.queries)
        try:
            print("Hola")
            logging.info(
                f"URL: {request.path}, "
                f"Método HTTP: {request.method}, "
                f"Código de estado: {response.status_code}, "
                f"Consultas SQL realizadas: {query_count}, "
                f"Duración total: {duration} segundos"
            )
        except Exception as e:
            logging.error(f"Error al registrar información de la solicitud: {e}")
        return response
