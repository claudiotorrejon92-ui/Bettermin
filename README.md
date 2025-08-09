# Bettermin Rapid Simulator — Beta IA

Simulador rápido en **Streamlit** para priorizar relaves y estimar desempeño en:
**Preconcentración (TOMRA/gravimetría) → BIOX → Lixiviación/SX-EW**, con métricas
técnico–económicas por relave y recomendaciones operacionales.

## Estructura
- `app.py` → **(copiar desde el canvas de ChatGPT)**. Es el archivo principal de la app.
- `requirements.txt` → Dependencias mínimas.
- `columns_mapping.json` → Diccionario de mapeo de columnas de entrada a esquema estándar.
- `.streamlit/config.toml` → Tema visual.
- `sample_data/` → Carpeta para ejemplos locales.

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre el navegador en la URL que te muestre Streamlit.

## Despliegue gratis (Streamlit Community Cloud)
1. Crea un repositorio en GitHub (p. ej. `bettermin-rapid-simulator`).
2. Sube estos archivos y **agrega `app.py`** (cópialo desde el canvas).
3. Ve a https://share.streamlit.io → New app → conecta el repo → elige rama y `app.py`.
4. Deploy. Cada *push* al repo actualizará la app.

## Entrada esperada
Un archivo CSV/Excel con columnas que el sistema intentará mapear
automáticamente. Esquema estándar recomendado:
- `Nombre`, `Comuna`, `Au_gpt`, `Ag_gpt`, `Cu_pct`, `As_ppm`, `S_tot_pct`, `S_sulf_pct`, `Fe_pct`, `Humedad_pct`, `P80_um`, `Tonelaje_t`

Si usas otros nombres, ajusta el CSV o edita `columns_mapping.json`.

---
_Última actualización: 2025-08-09_
