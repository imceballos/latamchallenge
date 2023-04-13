# FASTAPI Scaffold

## Build

- `make build-nc`
- `make run`

El Jupyter notebook asociado al Challenge se encuentra en HTTPstress\to_expose.ipynb

## Latam Challenge

1. Escoger el modelo que a tu criterio tenga un mejor performance, argumentando la decisión.

A partir de lo presentado en las secciones del Jupyter Notebook, el módelo mejor desarrollado corresponde a XGBoost con reducción de dimensionalidad, considero que las razones que lo hacen el óptimo son:
- Reducción de dimensionalidad: El hecho de disminuir el numero de variables de 37 a 11 permite de forma indirecta reducir la multicolinealidad entre variables, requisito fundamental para el desarrollo de cualquier módelo.
- Mejor tiempo de respuesta: A partir del punto anterior, el tiempo de ejecución de operaciones es mejor, dado que las matrices operadoras tienen menor dimensión y por otro lado, el size del módelo es menor, lo que facilita su carga usando `pickle`.


2. Implementar mejoras sobre el modelo escogiendo la o las técnicas que prefieras.

Dentro de las mejoras aplciadas para mejorar la performance del módelo están:
- Normalizacón de las variables: Aplicación de `StandardScaler` o `MinMaxScaler` a las variables.
- Input format training: El módelo originalmente se entrenaba a partir de un objeto de tipo DataFrame, esto no es recomendable, por lo que se aplico el método `.values` para serializar.
- Multicolinealidad: Factor de Inflación de la varianza (VIF).
- Distribución train test: Los conjuntos de entrenamiento y testeo pueden distribuirse según una proporción 70/30 o 80/20.
- Reducción de dimensionalidad (PCA).


3. Exponer el modelo seleccionado como API REST para ser expuesto.

La exposición de este servicio se realizó usando FASTAPI, para ello se definen tres endpoints que representan los siguientes módelos:

- `/predict_lg`. Logistic R.
- `/predict_xgb`  XGBoost
- `/predict_xgb_opt`  XGBoost optimized

4. Hacer pruebas de estrés a la API con el modelo expuesto con al menos 50.000 requests durante 45 segundos. Para esto debes utilizar esta herramienta y presentar las métricas obtenidas.

Los testeos realizados con la herramienta `wrk` se encuentran documentados en el directorio HTTPstress.

5. ¿Cómo podrías mejorar el performance de las pruebas anteriores?

Considero que para una mejora en el tiempo de respuesta de la aplicación se podria realizar lo siguiente:

- Redis: Implementación de Cache.
- Optimización del módelo de ML.
- Refactorización de código.

## Personal analysis

El siguiente analisis incluye comentarios sobre lo presentado en el Jupyter Notebook.

Comentarios secciòn Análisis:
Observamos una gran participación de mercado  en la industria nacional por parte de Latam Airlines. 
Cabe destacar que la distribución de vuelos según el día es el mismo, todos los días sale el mismo número de vuelos, excepto para los días 29, 30, 31. Esto sucedería dado que no todos los meses tienen la misma duración. Para el caso de los meses, se observa una curva cóncava donde disminuye el número de vuelos para invierno, lo que se considera un fenómeno estacionario. 
Dado que la distribución de vuelos tanto nacionales como internacionales es semejante, seria recomendable incluir la distancia recorrida entre origen y destino como un indicador.
En la sección de vuelos por destinos, el principal destino es Buenos Aires, seguido por Antofagasta y Lima, tengamos en cuenta la desactualización de la información, dado que Latam ya no opera en Argentina.
Para la preparación del modelo y lo restante se definen one-hot methods para categorizar los vuelos según sean en Temporada alta o baja (temporada_alta), se define la variable categórica de atraso (atraso_15) de acuerdo a los minutos de diferencia entre el arribo real y teórico del vuelo. Además de ello se definen los periodos de mañana, tarde y noche según la hora de partida del vuelo para definir variables categóricas que serán agregadas al análisis.
Se definen las tasas de atraso según el destino del vuelo, donde la mayor tasa de atraso se ve en los vuelos internacionales; Houston, Atlanta y Ciudad de Panamá son los destinos con mayor atraso. Sumado a ello, se describe la tasa de atraso según el operador aéreo, donde la mayor tasa de retraso es Aeroméxico, United Airlines y Delta Airlines. Dentro de la categoría de atrasos por mes, observamos que los meses de mayor retraso son Marzo, Abril y Febrero, lo que podria relacionarse con la época estival y la temporada de vacaciones. 
La tasa de retraso por temporada, entre temporada alta y baja se encuentra en torno a un 5%. Para la tasa de retraso según día, los días con mayor tasa corresponden a los del fin de semana (Sábado y Domingo). Para el caso de los vuelos nacionales o internacionales, observamos que hay cerca de un 2% de diferencia entre ambos, lo que podría deberse a la antelación de 2 hrs vuelos nacionales y 4 hrs vuelos internacionales.



Modelo ML:
Modificar la ditrubciòn train test a 70-30 o 80-20
Estandarizar las variables y normalizarlas (MinMax, StandardScaler, Z-Norm)
Multicolinealidad, eliminar variables con alta relaciòn entre ellas

Codigo:
Modificación sns barplot, se agregan los atributos x, y en el llamado.
Modificación del size del plot, bajado de (10, 8) a (8, 4)
El título de los gráficos es cambiado, según sea el caso de Dia, Dia de semana o Mes


## Swagger

- Documentation at 127.0.0.1:80/docs
