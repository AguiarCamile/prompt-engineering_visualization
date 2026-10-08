@echo off
setlocal
cd /d C:\Users\Labvis\Downloads\imagens3120
echo ============================================================
echo LOCALIZACAO DE FONTES DO PIPELINE DE CONFORMIDADE VISUAL
echo ============================================================
for %%F in (
visual_features_v4_plot_area_v6.py
axis_conformity_calibration_v3.py
ticklabel_presence_calibration_v4.py
mark_presence_validation_v3.py
chart_type_multiclass_validation_v1.py
chart_type_exclusive_holdout_v2.py
audit_absent_chart_types_v1.py
color_conformity_v4.py
validate_color_bars_core_v5.py
validate_color_lines_v3.py
validate_color_scatter_v7_markercore_final.py
validate_color_scatter_holdout_v8.py
ticklabel_content_extraction_calibration_ba_v7.py
ticklabel_content_extraction_holdout_ba_v7_frozen.py
ticklabel_content_extraction_production_ba_v7_frozen.py
) do (
  echo.
  echo --- %%F ---
  where /r . %%F 2>nul
)
echo.
echo Fim.
pause
