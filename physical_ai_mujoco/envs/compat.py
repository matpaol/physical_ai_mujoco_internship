"""Shim di compatibilita' MuJoCo <-> Gymnasium.

Da MuJoCo 3.13 la firma di `mjv_moveCamera` e' cambiata: l'argomento `scn`
(MjvScene) e' stato rimosso.

    <= 3.12 : mjv_moveCamera(m, action, reldx, reldy, scn, cam)
    >= 3.13 : mjv_moveCamera(m, action, reldx, reldy, cam)

Gymnasium 1.3.0 chiama ancora la versione a 6 argomenti nei callback del mouse
del viewer `human`, quindi la finestra interattiva solleva
`TypeError: mjv_moveCamera(): incompatible function arguments` al primo
movimento del mouse. In entrambe le firme la camera e' l'ULTIMO argomento
posizionale, quindi basta provare prima la forma nuova.
"""

from __future__ import annotations

import mujoco

_APPLICATO = "_physical_ai_compat"


def applica_shim() -> bool:
    """Installa il wrapper. Ritorna True se e' stato applicato ora."""
    if getattr(mujoco.mjv_moveCamera, _APPLICATO, False):
        return False

    originale = mujoco.mjv_moveCamera

    def mjv_moveCamera(m, action, reldx, reldy, *resto):
        if resto:
            try:
                return originale(m, action, reldx, reldy, resto[-1])
            except TypeError:
                pass
        return originale(m, action, reldx, reldy, *resto)

    setattr(mjv_moveCamera, _APPLICATO, True)
    mujoco.mjv_moveCamera = mjv_moveCamera
    return True


applica_shim()
