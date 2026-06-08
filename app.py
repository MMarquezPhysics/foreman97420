# app.py
import streamlit as st
from plotly.graph_objects import Figure, Bar
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.io import decode_jpeg
from tensorflow.image import resize
from tensorflow import expand_dims

CLASSES = ["Glioma", 'Meningioma', 'Sin Tumor (Sano)', 'Pituitario']
TARGET_SIZE = (224, 224)

def preprocess_uploaded_image(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    image = decode_jpeg(file_bytes, channels=3)
    image = resize(image, TARGET_SIZE)
    image = image / 255.0
    image = expand_dims(image, axis=0)
    return image

def predict_tumor_probabilities(model, preprocessed_tensor):
    predictions = model.predict(preprocessed_tensor, verbose=0)[0]
    probability_dict = {
        CLASSES[i]: float(predictions[i]) for i in range(len(CLASSES))}
    return probability_dict
# ==========================================
# CONFIGURACIÓN DE LA PÁGINA Y ESTILO (UI)
# ==========================================
st.set_page_config(
    page_title="Foreman974-20 | Soporte Diagnóstico",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #9AECF9; }
    .stAlert { border-radius: 8px; }
    h1, h2, h3 { color: #1A2B4C !important; font-family: 'Helvetica Neue', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# GESTIÓN PERSISTENTE DEL MODELO
# ==========================================
@st.cache_resource
def load_foreman_model():
    model_path = "./models/model_frac_1.0_layers_20_0.974.keras" 
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"⚠️ Error crítico al cargar el modelo en '{model_path}': {e}")
        return None

model = load_foreman_model()

# ==========================================
# SIDEBAR: CONTEXTO INSTITUCIONAL
# ==========================================
with st.sidebar:
    st.image("foreman_logo.png", width=150)
    st.title("Proyecto Foreman")
    st.markdown("---")
    
    st.subheader("💡 Nuestro modelo experto")
    st.markdown("""
    **Modelo Foreman974-20**
    * **Precisión Global:** 97.4%
    * **Recall Global:** 97.7%
    * **Filtrado de Casos Sanos:** >99% de efectividad.
    * **Mitigación de Ambigüedad:** Reducción de error entre diagnósticos similares hasta un 3.8%. Diagnóstico en segundos.
    """, help="Entrenado en un universo de ~42k imágenes MRI. Métricas validadas con ~2k MRI independientes")
    st.markdown("---")

    #st.subheader("📋 Diplomado en Ciencia de Datos")
    st.markdown("""
    **Foreman es desarrollado por Mateo Márquez Arias**""", help="https://www.linkedin.com/in/mateo-marquez-arias/")
    st.markdown("""
    * **Diplomado en Ciencia de Datos** FES Acatlán UNAM               
    * **Generación:** 31
    * **Módulo:** 5""")
    #st.date_input("Fecha del Estudio")
# ==========================================
# CUERPO PRINCIPAL: STORYTELLING VISUAL
# ==========================================
color_titulo ="#00A896" #"#0379FFB2"
st.markdown(
    f"<h1 style='text-align: left;'>Diagnóstico de neoplasias "
    f"<span style='color:{color_titulo}; font-weight:bold;'>asistido por IA</span></h1>",
    unsafe_allow_html=True
)
#span style='color:{color_diagnostico}; font-weight:bold;'>asistido por IA</span> ({confianza:.1f}%)
#st.title("🧠 Diagnóstico de neoplasias asistido por IA")
st.markdown("#### *Mitigación del error en diagnóstico: Meningiomas - Gliomas - Pituitario*")
#st.markdown("#### *Modelo: Foreman974-20*")
st.markdown("---")

col_izq, col_cent, col_der = st.columns([0.8, 1, 1], gap="large")

with col_izq:
    st.subheader("Imagen de Resonancia Magnética")
    uploaded_file = st.file_uploader(
        "Arrastre o seleccione la imagen de resonancia (MRI) del paciente", 
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        img_display = Image.open(uploaded_file)
        st.image(img_display, caption=f"Estudio cargado exitosamente", use_container_width=True)
    else:
        st.info("Esperando carga del la imagen de resonancia para iniciar la inferencia automática.")

with col_cent: 
    if uploaded_file is not None and model is not None:
        st.subheader("📊 Probabilidades y Diagnóstico")
        
        with st.spinner("Analizando patrones morfológicos en la imagen..."):
            tensor_imagen = preprocess_uploaded_image(uploaded_file)
            resultados_dict = predict_tumor_probabilities(model, tensor_imagen)
        
        clase_predicha = max(resultados_dict, key=resultados_dict.get)
        confianza = resultados_dict[clase_predicha]*100
        
        lista_clases = list(resultados_dict.keys())
        lista_probabilidades = [p*100 for p in resultados_dict.values()]
        max_idx = lista_clases.index(clase_predicha)
        
        # ==========================================
        # GRÁFICO PERSONALIZADO (USANDO LAS IMPORTACIONES COMPACTAS)
        # ==========================================
        colors = ["#C3C3C3"] * len(CLASSES)
        if clase_predicha == "Sin Tumor (Sano)":
            colors[max_idx] = "#07CEBA"
        else:
            colors[max_idx] = "#EB9008"
        
        st.metric(label="Resultado de diagnóstico", value=clase_predicha, delta=f"{confianza:.1f}% Confianza")

        fig = Figure(Bar(
            x=lista_probabilidades,
            y=lista_clases,
            orientation='h',
            marker_color=colors,
            text=[f"{p:.2f}%" for p in lista_probabilidades],
            textposition='auto',
            hovertemplate="Clase: %{y}<br>Probabilidad: %{x:.2}%<extra></extra>"
        ))
        
        fig.update_layout(
            title=dict(text="<b>Distribución de Probabilidades</b>", font=dict(size=18, color="#164D6C")),
            xaxis=dict(title="Nivel de Confianza (0-100%)", range=[0, 100], gridcolor="#EDF2F7"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=50, b=20), height=280,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # ==========================================
        # SECCIÓN CORE: ACCIONABILIDAD
        # ==========================================

with col_der:    
    if uploaded_file is not None and model is not None:
        st.markdown("### Recomendaciones de Acción")
        # primero un if si el resultado es confiable:
        if clase_predicha == "Sin Tumor (Sano)" and confianza > 85:
            st.success(
                f"**Prioridad Baja - Filtrado Seguro Detectado**\n\n"
                "El sistema no detecta anomalías neoplásicas con un alto índice de certeza. "
                "La tasa de efectividad del modelo es superior al 99% en casos sanos, este estudio "
                "podría ser derivado a liberación rutinaria."
            )
        elif clase_predicha in ["Glioma", "Meningioma", "Pituitario"] and confianza > 85:
            st.error(
                f"**Prioridad Alta - Revisión Urgente por Patología ({clase_predicha})**\n\n"
                "**Acción Recomendada:** Revisión prioritaria por neurooncólogo para tratamiento.\n\n"
                "La arquitectura mitiga activamente la ambigüedad histórica entre "
                "estas dos variantes (reducción del error al 3.8%)."
            )
            # un mensaje especial si es una afección de las que el modelo confunde
            if clase_predicha in ["Glioma", "Meningioma"]:
                other_af = "Glioma" if clase_predicha=="Meningioma" else "Meningioma"
                st.warning(
                    f"Se recomienda **verificación por especialista**\n\n"
                    "Ya que nuestro modelo, *en algunas pocas ocasiones*, llega a "
                    f"diagnosticar esta afectación como **{other_af}**"
                )
        else: #
            st.warning(
                f"**Confiabilidad insuficiente**\n\n"
                "La confibilidad de la precisión es particularmente baja. Por favor revise que la imagen "
                "cargada corresponda a una imagen MRI craneal."
            )

        if confianza <= 95:
            st.markdown("---")
            st.markdown("### Confiabilidad de resultados")
            st.warning(
                f"**Confirmación humana recomendada**\n\n"
                f"Se recomienda verificar la predicción de nuestro modelo con un profesional neurooncólogo\n\n"
                f"Foreman974-20 posee una tasa de error máxima de 4.5%, pero el porcentaje de confianza"
                "de este diagnóstico sugiere una segunda opinión\n\n"
                "Foreman es un soporte al diagnóstico y sus resultados **no deben tomarse como infalibles**."
            )
            
    elif uploaded_file is None:
        fig = Figure()
        fig.update_layout(
            title="<b>Esperando análisis...</b>", xaxis=dict(visible=False), yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("El sistema está listo, pero el modelo Foreman974-20 no se ha cargado correctamente.")

st.markdown("---") # Añade una línea divisoria sutil antes del footer

st.markdown(
    """
    <div style="text-align: center; margin-top: 20px;">
        <p style="color: #64748B; font-size: 13px;">
            ¿Quieres conocer más sobre el rigor científico y operativo de Foreman? <br>
            <a href="https://drive.google.com/file/d/1bx_OG6WcdRn3yauUXjVUdyqv0L70UCgu/view?usp=sharing" target="_blank" style="color: #00A896; text-decoration: none; font-weight: 600;">Consulta la documentación técnica aquí</a>.
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)