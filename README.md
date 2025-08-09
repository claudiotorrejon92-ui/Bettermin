# Bettermin Rapid Simulator — Beta IA

Simulador rápido en **Streamlit** para priorizar relaves y estimar desempeño en:
**Preconcentración (TOMRA/gravimetría) → BIOX → Lixiviación/SX-EW**, con métricas
técnico–económicas por relave y recomendaciones operacionales.

## Estructura
- `app.py` → App principal
- `requirements.txt` → Dependencias mínimas
- `columns_mapping.json` → Diccionario de mapeo de columnas (referencia)
- `.streamlit/config.toml` → Tema visual

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud
1. Crea un repositorio en GitHub y sube estos archivos.
2. Entra a https://share.streamlit.io → **New app** → conecta el repo, elige `app.py`.
3. Deploy. Cada *push* al repo actualizará la app.

## Entrada esperada
Un CSV/Excel con columnas que el sistema intentará mapear automáticamente.
Esquema estándar recomendado:
`Nombre, Comuna, Au_gpt, Ag_gpt, Cu_pct, As_ppm, S_tot_pct, S_sulf_pct, Fe_pct, Humedad_pct, P80_um, Tonelaje_t`

Si usas otros nombres, puedes editar tu archivo o ajustar el diccionario en `columns_mapping.json`.

---
_Última actualización: 2025-08-09_
