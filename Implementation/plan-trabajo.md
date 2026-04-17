# Plan de implementacion: Ensamblador local + Firebase + bot de salida

## Resumen
- Destino del plan: `PrediccionClima/Implementation/`.
- Objetivo operativo: correr el ensamblador intradia en local, publicar sus predicciones a Firebase para revision remota, y activar desde la webapp un bot en VPS que solo gestione la salida de la apuesta.
- Flujo diario v1: prediccion local `11:00` a `17:00` hora de Argentina cada `45` minutos, revision manual en webapp, entrada manual en Polymarket, activacion del monitoreo, salida automatica por stop loss, take profit o corte horario.
- V1 no automatiza la entrada; solo prediccion, visualizacion, activacion del monitoreo y salida.

## Cambios de implementacion
- Paso 1. Crear un runner operativo nuevo para produccion, en lugar de usar `run_hourly_ensemble.py` tal como esta hoy, porque ahora agenda desde la zona horaria local del host y a intervalos relativos. El runner debe usar `America/Argentina/Buenos_Aires` y disparar exactamente estos 9 slots por dia: `11:00`, `11:45`, `12:30`, `13:15`, `14:00`, `14:45`, `15:30`, `16:15`, `17:00`.
- Paso 1. El runner debe envolver `run_daily_ensemble.py` con modo operativo: actualizando datos y reentrenando.  Mantendra los artefactos `latest` para consumo externo.
- Paso 1. Arrancar el runner con una tarea de Windows al iniciar sesion o arrancar la maquina; la logica de horario vive dentro del runner para no depender de la zona horaria del sistema. Guardar lock y state file para evitar doble instancia o doble ejecucion del mismo slot.
- Paso 1. Despues de cada corrida, consolidar `status.json`, `latest_live_forecast.json` y `latest_market_forecast.json` en un snapshot unico con `runId`, `generatedAt`, `forecastOriginDatetime`, temperatura actual, maximo del dia, pico pronosticado, top bins de mercado, probabilidades por threshold y estado de la corrida.
- Paso 2. Exponer una Cloud Function HTTP `ingestForecastSnapshot` en `us-central1`, protegida con secreto compartido. Esta funcion recibe el snapshot del paso 1, valida campos minimos y escribe en Firestore `system/latestForecast` y `forecastRuns/{runId}`.
- Paso 2. Exponer una segunda Cloud Function autenticada `setActiveTrade` para registrar desde la webapp la posicion activa del bot. Debe validar `marketSlug`, `conditionId`, `outcomeTokenId`, `side`, `size`, `stopLossTempC`, `stopLossOperator`, `takeProfitTempC`, `takeProfitOperator`, `forceExitAtArt` y `enabled`.
- Paso 2. Crear una webapp en Firebase Hosting que lea Firestore y muestre: ultimo snapshot, ultimas 9 corridas del dia, probabilidades por threshold, top bins, frescura de datos, estado del bot y un formulario para activar o desactivar la posicion con umbrales editables.
- Paso 2. Proteger la webapp con Firebase Auth por Google para uno o pocos usuarios autorizados y dejar Firestore y Functions cerrados al publico.
- Paso 3. Reutilizar y refactorizar el scraper actual de Weather Underground SAEZ para el VPS. El bot debe muestrear cada 10 segundos, convertir `F -> C` y publicar heartbeat y ultima lectura en `botStatus/current`.
- Paso 3. Ejecutar el bot en un VPS Linux como servicio `systemd`. El bot leera la configuracion activa desde Firestore cada 5 segundos y evaluara salidas en este orden: `forceExitAtArt`, stop loss, take profit.
- Paso 3. Implementar la salida real via Polymarket CLOB API. El bot usara credenciales en variables de entorno, enviara ordenes de cierre hasta dejar la posicion en tamano `0`, y registrara cada trigger, intento y resultado en `botEvents/{eventId}`.
- Paso 3. Para priorizar velocidad, la primera lectura valida que cruce un umbral dispara salida inmediata; no habra debounce en v1. Los operadores `gte` y `lte` definen la direccion del cruce y evitan asumir si la posicion gana con subida o bajada de temperatura.

## APIs e interfaces
- `POST /ingestForecastSnapshot`: JSON con `runId`, `generatedAt`, `forecastOriginDatetime`, `currentTemperatureF`, `currentTemperatureC`, `maxSoFarF`, `maxSoFarC`, `ensemblePeakForecastF`, `ensemblePeakForecastC`, `topMarketBins`, `probabilitiesAtOrAboveC`, `status` y `sourceUrls`.
- `setActiveTrade`: input con `marketSlug`, `conditionId`, `outcomeTokenId`, `side`, `size`, `stopLossTempC`, `stopLossOperator`, `takeProfitTempC`, `takeProfitOperator`, `forceExitAtArt` y `enabled`.
- Firestore `system/latestForecast`: snapshot mas reciente listo para UI.
- Firestore `forecastRuns/{runId}`: historial intradia e historico.
- Firestore `tradeControl/activeTrade`: configuracion vigente del bot.
- Firestore `botStatus/current`: heartbeat, ultima temperatura observada, estado operativo y ultima razon de salida.
- Firestore `botEvents/{eventId}`: auditoria de triggers, ordenes y cierres.

## Plan de pruebas
- Validar que el runner genera exactamente 9 slots diarios en hora de Argentina aunque el host use otra zona horaria.
- Validar que las corridas operativas no reentrenan modelos y siguen generando artefactos `latest` consistentes.
- Probar `ingestForecastSnapshot` con secreto valido e invalido y verificar escritura correcta de `latestForecast` mas historico.
- Verificar en la webapp estados `fresh`, `stale` y `error`, y que el formulario de posicion rechace umbrales incompletos u operadores invalidos.
- Simular bot con `gte` y `lte` para stop loss y take profit, confirmar una sola salida por posicion y cierre forzado a las `17:00` de Argentina si la posicion sigue abierta.
- Probar caida temporal de Weather Underground, caida temporal de Firestore y fallo de orden en CLOB; el bot debe reintentar, conservar estado y dejar trazabilidad.

## Supuestos y defaults
- La maquina local estara encendida y con internet entre `11:00` y `17:00` hora de Argentina.
- La entrada a la apuesta seguira siendo manual; la automatizacion cubre prediccion, visualizacion remota, activacion del monitoreo y salida.
- Los umbrales configurables del bot se guardaran y mostraran en grados Celsius; el scraping seguira leyendo Fahrenheit y convirtiendo internamente.
- La fuente observada para el bot sera Weather Underground estacion SAEZ/Ezeiza, la misma ya usada por el codigo actual.
- Las credenciales de Firebase, CLOB/Polymarket y el secreto de ingesta viviran fuera del repo.
