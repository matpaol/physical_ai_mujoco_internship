# Test del progetto

Avviare `/opt/miniconda3/envs/mujoco-tirocinio/bin/python main_test.py` dalla
radice del progetto, oppure `python main_test.py --suite osserva`. Il menu
raggruppa i test per responsabilità senza spostare i singoli file: `scena`,
`osserva`, `validazione_sensori`, `pipeline_dati`, `decidi`, `architettura` e
`tutti`. Ogni nuovo `test_*.py` deve essere assegnato in `tests/suites.py`;
altrimenti il menu lo segnala. `tutti` usa la normale discovery pytest.

`validazione_sensori` prova il CAD STL su 24 pose e cinque percentuali di
superficie sintetica. Il test verifica la copertura e la struttura del report,
non certifica l'accuratezza della posa. Per i valori numerici e un JSON:

```
python -m physical_ai_mujoco.evaluation.sensor_validation --output outputs/cad_validation.json
```

La percentuale è una selezione dei punti CAD per quota, non una misura di
esposizione camera/LiDAR in MuJoCo. Il rapporto dimensionale vale 1 per ogni
fit accettato perché `CADMatcher` assegna la dimensione nominale: non è una
prova di ricostruzione corretta. Per convalidare 1B/1C servono ancora prove
con scansioni LiDAR simulate, ground truth e detector addestrato.
