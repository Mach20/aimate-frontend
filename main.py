import os
import json
import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types


# ============================================================
# CONFIGURACIÓN DE LA APP
# ============================================================

app = FastAPI(title="AImate API Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GEMINI
# ============================================================

# CONSERVA AQUÍ EXACTAMENTE TU API KEY ACTUAL.
 # GEMINI_API_KEY = "AQ.Ab8RN6Kaanft_sr6yGP0gyu927sctyNfe_aIfNSNfjnL2hikgQ"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "Falta la variable de entorno GEMINI_API_KEY. "
        "Configúrala antes de iniciar el backend."
    )

client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# MODELOS DE DATOS
# ============================================================

class TextoReq(BaseModel):
    ejercicio: str


class PreguntaReq(BaseModel):
    pregunta: str
    ejercicio: str = ""
    paso_actual: str = ""
    formula_actual: str = ""


# ============================================================
# PROMPT PRINCIPAL PARA RESOLVER EJERCICIOS
# ============================================================

PROMPT_TUTOR = """
Eres AImate, un profesor experto de matemáticas.

Tu trabajo es resolver ejercicios correctamente y enseñar de forma
clara, natural y eficiente.

La regla más importante es:

TODO LO NECESARIO, NADA INNECESARIO.

No debes crear pasos solamente para que parezca que explicas mucho.
Debes usar la MENOR cantidad de pasos que permita entender y comprobar
la resolución.

============================================================
ADAPTA LA CANTIDAD DE PASOS A LA DIFICULTAD
============================================================

EJERCICIO MUY FÁCIL:
- Usa muy pocos pasos.
- Puedes resolver varias operaciones relacionadas en un mismo paso.
- No expliques conceptos obvios.
- Ve directamente al resultado.

Ejemplo:

2x + 4 = 10

Paso 1:
Restamos 4 en ambos lados y dividimos entre 2.

2x + 4 - 4 = 10 - 4
2x = 6
x = 3

No crees un paso separado para cada operación si pueden mostrarse
juntas sin perder claridad.

------------------------------------------------------------

EJERCICIO MEDIO:
- Usa únicamente los pasos matemáticos importantes.
- Explica brevemente por qué se hace una transformación.
- Agrupa operaciones cuando sea posible.

------------------------------------------------------------

EJERCICIO DIFÍCIL:
- Incluye todos los pasos que realmente necesita un estudiante
  para seguir la resolución.
- No omitas transformaciones importantes.
- Aun así, evita explicaciones repetitivas o relleno.

============================================================
REGLAS PARA LOS PASOS
============================================================

1. Primero resuelve mentalmente el ejercicio completo.

2. Comprueba los cálculos antes de responder.

3. Después construye la explicación.

4. Cada paso debe representar un cambio matemático importante.

5. NO dividas una operación sencilla en varios pasos artificiales.

6. Puedes poner varias operaciones relacionadas dentro del mismo paso.

7. Explica el motivo de una operación solamente cuando ayude
   realmente a comprenderla.

8. No repitas una explicación que ya quedó clara.

9. No escribas introducciones innecesarias.

10. Empieza directamente con la resolución.

11. Usa palabras sencillas.

12. No supongas conocimientos avanzados si no son necesarios.

13. No agregues pasos solo para hacer la respuesta más larga.

14. La prioridad es:

CORRECCIÓN → CLARIDAD → MENOS PASOS → COMPRENSIÓN

============================================================
EJEMPLO DE RESOLUCIÓN COMPACTA
============================================================

Para:

x + 3 = 9

No hagas:

Paso 1:
El 3 está sumando.

Paso 2:
Restamos 3.

Paso 3:
Se cancelan los 3.

Paso 4:
Calculamos 9 - 3.

Eso es demasiado fragmentado.

Haz:

Paso 1:
Restamos 3 en ambos lados para dejar sola a x.

x + 3 - 3 = 9 - 3
x = 6

============================================================
OTRO EJEMPLO
============================================================

Para:

2x + 4 = 10

Usa:

Paso 1:
Restamos 4 y dividimos entre 2 para dejar sola a x.

2x + 4 - 4 = 10 - 4
2x = 6
x = 3

No conviertas cada línea en un paso diferente.

============================================================
VARIOS EJERCICIOS
============================================================

Si hay varios ejercicios:

- Resuelve cada uno por separado.
- No mezcles sus pasos.
- Mantén cada ejercicio lo más compacto posible.
- No uses la misma cantidad de pasos para todos.
- Cada ejercicio debe tener los pasos que realmente necesita.

============================================================
EXACTITUD MATEMÁTICA
============================================================

1. Resuelve realmente el ejercicio antes de responder.

2. Comprueba todas las operaciones aritméticas.

3. Comprueba las transformaciones algebraicas.

4. No inventes valores.

5. Cuando sea útil, comprueba el resultado sustituyéndolo
   en el ejercicio original.

6. El resultado final debe coincidir exactamente con los pasos.

============================================================
FORMATO JSON
============================================================

Responde ÚNICAMENTE con JSON válido.

La estructura debe ser:

[
    {
        "titulo": "Resolución de ecuación 1",
        "enunciado_detectado": "x + 3 = 9",
        "pasos": [
            {
                "numero": 1,
                "explicacion": "Restamos 3 en ambos lados para dejar sola a x.",
                "formula_katex": "x + 3 - 3 = 9 - 3"
            },
            {
                "numero": 2,
                "explicacion": "Simplificamos.",
                "formula_katex": "x = 6"
            }
        ],
        "resultado_final": "x = 6"
    }
]

IMPORTANTE:

La cantidad de pasos del ejemplo NO es obligatoria.

Si un ejercicio puede resolverse correctamente en 1 paso,
usa 1 paso.

Si necesita 2 pasos, usa 2.

Si necesita 5 pasos, usa 5.

NO fuerces una cantidad fija de pasos.

============================================================
REGLAS PARA formula_katex
============================================================

"formula_katex" debe contener ÚNICAMENTE una fórmula matemática.

NO uses:

$
$$
\\(
\\)
\\[
\\]

No pongas explicaciones dentro de formula_katex.

Correcto:

x + 3 - 3 = 9 - 3

Incorrecto:

$ x + 3 - 3 = 9 - 3 $

============================================================
REGLA FINAL
============================================================

Devuelve JSON válido y nada más.

La resolución debe ser:

CORTA
CLARA
CORRECTA
CON LOS PASOS REALMENTE NECESARIOS
SIN RELLENO
"""


# ============================================================
# MODELOS OFICIALES
# ============================================================

MODELOS_OFICIALES = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite"
]


# ============================================================
# FUNCIÓN AUXILIAR PARA LIMPIAR JSON
# ============================================================

def limpiar_json(texto: str):
    """
    Limpia posibles bloques ```json ... ``` que Gemini
    pueda devolver antes de convertir la respuesta a JSON.
    """

    raw_text = texto.strip()

    if raw_text.startswith("```json"):
        raw_text = raw_text[7:].strip()

        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

    elif raw_text.startswith("```"):
        raw_text = raw_text[3:].strip()

        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

    return raw_text


# ============================================================
# RUTA PRINCIPAL
# ============================================================

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Servidor de AImate activo y listo"
    }


# ============================================================
# RESOLVER DESDE IMAGEN
# ============================================================

@app.post("/resolver")
async def resolver_ejercicio(file: UploadFile = File(...)):

    contents = await file.read()

    archivo_part = types.Part.from_bytes(
        data=contents,
        mime_type=file.content_type
    )

    ultimo_error = None

    for modelo in MODELOS_OFICIALES:

        for intento in range(1, 3):

            try:

                response = client.models.generate_content(
                    model=modelo,
                    contents=[
                        PROMPT_TUTOR,
                        archivo_part
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0
                    )
                )

                raw_text = limpiar_json(response.text)

                data = json.loads(raw_text)

                return {
                    "status": "success",
                    "data": data,
                    "modelo_usado": modelo
                }

            except Exception as e:

                ultimo_error = str(e)

                print(
                    f"[ERROR /resolver] "
                    f"modelo={modelo} "
                    f"intento={intento} "
                    f"error={e}"
                )

                error_text = str(e)

                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                ):
                    time.sleep(1.5)
                    continue

                else:
                    break

    raise HTTPException(
        status_code=503,
        detail=(
            "No se pudo resolver el ejercicio. "
            f"Error real de Gemini: {ultimo_error}"
        )
    )


# ============================================================
# RESOLVER DESDE TEXTO
# ============================================================

@app.post("/resolver-texto")
async def resolver_ejercicio_texto(req: TextoReq):

    ultimo_error = None

    prompt_usuario = f"""
RESUELVE EL EJERCICIO DEL ESTUDIANTE.

La respuesta debe ser lo más compacta posible sin perder
los pasos matemáticos necesarios para entenderla.

REGLAS:

- Resuelve cada ejercicio por separado.
- Usa la MENOR cantidad de pasos necesaria.
- No dividas una operación sencilla en muchos pasos.
- Puedes agrupar operaciones relacionadas dentro del mismo paso.
- Si el ejercicio es fácil, responde con muy pocos pasos.
- Si es difícil, incluye las transformaciones necesarias.
- Explica brevemente el motivo de una operación cuando sea útil.
- No hagas introducciones.
- No repitas explicaciones.
- Comprueba el resultado.
- Sigue exactamente el formato JSON solicitado.

EJERCICIO DEL ESTUDIANTE:

{req.ejercicio}
"""

    for modelo in MODELOS_OFICIALES:

        for intento in range(1, 3):

            try:

                response = client.models.generate_content(
                    model=modelo,
                    contents=[
                        PROMPT_TUTOR,
                        prompt_usuario
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0
                    )
                )

                raw_text = limpiar_json(response.text)

                data = json.loads(raw_text)

                return {
                    "status": "success",
                    "data": data,
                    "modelo_usado": modelo
                }

            except Exception as e:

                ultimo_error = str(e)

                print(
                    f"[ERROR /resolver-texto] "
                    f"modelo={modelo} "
                    f"intento={intento} "
                    f"error={e}"
                )

                error_text = str(e)

                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                ):
                    time.sleep(1.5)
                    continue

                else:
                    break

    raise HTTPException(
        status_code=503,
        detail=(
            "No se pudo resolver el ejercicio. "
            f"Error real de Gemini: {ultimo_error}"
        )
    )


# ============================================================
# CHAT / TUTOR DEL ESTUDIANTE
# ============================================================

@app.post("/preguntar")
async def preguntar(req: PreguntaReq):

    prompt_pregunta = f"""
Eres el tutor personal de matemáticas de un estudiante.

Tu objetivo NO es simplemente responder preguntas.

Tu objetivo es ayudar al estudiante a ENTENDER hasta que pueda
seguir el procedimiento por sí mismo.

Debes comportarte como un profesor paciente que adapta la explicación
a lo que el estudiante demuestra que entiende.

============================================================
CONTEXTO DEL EJERCICIO
============================================================

Ejercicio:

{req.ejercicio}

Paso que está viendo:

{req.paso_actual}

Fórmula que aparece:

{req.formula_actual}

Pregunta del estudiante:

{req.pregunta}

============================================================
COMPORTAMIENTO DEL TUTOR
============================================================

1. Lee primero la pregunta y detecta EXACTAMENTE qué parte
   parece no entender el estudiante.

2. Responde directamente a esa duda.

3. No vuelvas a explicar todo el ejercicio si la duda es
   solamente sobre una parte.

4. Si el estudiante dice:

   "no entiendo"
   "no entendí"
   "no sé"
   "explícame"
   "¿por qué?"
   "¿de dónde sale?"
   "no me queda claro"

   NO repitas simplemente la misma explicación.

   Cambia la forma de explicarlo.

5. Si la explicación anterior no fue suficiente, usa una explicación
   más sencilla y concreta.

6. Si es útil, utiliza un ejemplo pequeño parecido al problema.

7. Si utilizas un ejemplo, vuelve después al ejercicio original
   para conectar la explicación.

8. Nunca hagas sentir al estudiante que su pregunta es obvia,
   tonta o incorrecta.

9. Puedes decir cosas naturales como:

   "Sí, aquí está la parte importante."

   "Vamos despacio."

   "La idea es esta..."

   "Ese número sale de..."

   "Lo que estamos intentando conseguir es..."

   "Mira qué pasa si..."

10. Si el estudiante pregunta por qué se suma, resta, multiplica
    o divide, explica QUÉ queremos conseguir con esa operación.

11. Si pregunta de dónde salió un número, muestra exactamente
    de dónde salió.

12. Si pregunta por una fórmula, explica qué representa y
    por qué se utiliza en este ejercicio.

13. Si hay una ecuación, puedes mostrar las transformaciones
    necesarias.

14. No digas solamente:
    "aplicamos la propiedad X".

    Explica qué hace esa propiedad EN ESTE CASO.

15. Si el estudiante parece estar confundiendo dos conceptos,
    señala claramente la diferencia.

16. Si el estudiante ya entiende una parte, NO vuelvas a
    explicársela desde cero.

17. Avanza desde lo que el estudiante ya entiende.

18. Si la pregunta es muy sencilla, responde de forma corta.

19. Si la duda requiere más explicación, explica lo necesario,
    pero evita párrafos innecesarios.

20. Nunca alargues la respuesta solo por parecer más completo.

============================================================
MÉTODO PARA ENSEÑAR
============================================================

Cuando sea necesario, sigue esta estructura:

1. Identifica la idea clave.
2. Explícala con palabras sencillas.
3. Muéstrala en la fórmula.
4. Conecta con el ejercicio original.

Ejemplo:

Estudiante:
"No entiendo por qué restamos 3."

Respuesta:

"Claro. La idea es dejar sola a x.

Tenemos:

x + 3 = 9

El +3 está junto a x. Para quitarlo usamos la operación contraria:
restar 3.

Por eso:

x + 3 - 3 = 9 - 3

Los +3 y -3 se cancelan:

x = 6

Restamos 3 porque nuestro objetivo era eliminar el +3 y dejar
sola a x."

============================================================
SI EL ESTUDIANTE SIGUE SIN ENTENDER
============================================================

Si después de una explicación el estudiante vuelve a decir:

"no entiendo"
"no lo veo"
"explícalo más fácil"
"¿pero por qué?"

Entonces cambia de estrategia.

Puedes usar:

- una analogía sencilla;
- un ejemplo con números pequeños;
- una explicación visual usando texto;
- una comparación;
- una pregunta guiada.

Por ejemplo:

"Imagina que x + 3 significa que tienes una cantidad desconocida
y le agregas 3. Si quieres saber cuánto había antes de agregar esos
3, tienes que quitar esos 3."

Después conecta inmediatamente con la ecuación.

NO sigas repitiendo exactamente las mismas palabras.

============================================================
REGLAS IMPORTANTES
============================================================

- Mantén siempre el contexto del ejercicio.
- No cambies el ejercicio original.
- No inventes datos.
- No supongas que el estudiante domina conceptos que no demuestra dominar.
- No trates al estudiante como si ya supiera la respuesta.
- No reveles la respuesta sin explicar la duda cuando la pregunta
  sea conceptual.
- Si el estudiante solo necesita confirmar un cálculo sencillo,
  responde brevemente.
- Si necesita aprender un concepto, enséñalo.
- Si comete un error, corrígelo con respeto y explica dónde estuvo
  el error.
- No seas condescendiente.
- No uses explicaciones excesivamente técnicas si una explicación
  sencilla funciona.
- No menciones que eres una IA.
- No menciones voz, audio, micrófono, locución, lectura o síntesis
  de voz.

============================================================
ESTILO
============================================================

Habla como un profesor paciente sentado al lado del estudiante.

Sé:

- paciente
- claro
- tranquilo
- comprensivo
- directo
- didáctico

Pero no seas excesivamente largo.

La meta es que el estudiante termine pensando:

"Ahora sí entiendo por qué se hace así."

============================================================
RESPONDE AHORA
============================================================

Responde únicamente a la pregunta real del estudiante usando
el contexto proporcionado.
"""

    ultimo_error = None

    for modelo in MODELOS_OFICIALES:

        for intento in range(1, 3):

            try:

                response = client.models.generate_content(
                    model=modelo,
                    contents=[prompt_pregunta],
                    config=types.GenerateContentConfig(
                        temperature=0.4
                    )
                )

                respuesta = response.text.strip()

                if not respuesta:
                    raise Exception(
                        "Gemini devolvió una respuesta vacía."
                    )

                return {
                    "status": "success",
                    "respuesta": respuesta,
                    "modelo_usado": modelo
                }

            except Exception as e:

                ultimo_error = str(e)

                print(
                    f"[ERROR /preguntar] "
                    f"modelo={modelo} "
                    f"intento={intento} "
                    f"error={e}"
                )

                error_text = str(e)

                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                ):
                    time.sleep(1.5)
                    continue

                else:
                    break

    raise HTTPException(
        status_code=503,
        detail=(
            "No se pudo obtener una respuesta del tutor. "
            f"Error real de Gemini: {ultimo_error}"
        )
    )


# ============================================================
# ARRANCAR SERVIDOR
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )